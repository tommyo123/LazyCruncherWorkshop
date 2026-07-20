//! Speed scores from 1 to 10 for the cruncher picker.
//!
//! The numbers below were measured by the bench on one fixed 14 KB payload:
//! wall clock packing time, and the 6502 cycles the whole generated SFX takes
//! to unpack on the c64-debug emulator. Unpacking is measured four times per
//! format, for forward and backward layout, each with the balanced decoder and
//! with the fastest decoder that priority speed selects. A backward decoder is
//! a different routine with its own cost, so it never borrows the forward
//! number.
//!
//! Both metrics span about two decades across the formats, so a linear map to
//! 1 to 10 would pile almost every format near 10. The map is logarithmic
//! instead, and the bounds are taken from the constants at runtime so the scale
//! re-normalizes if a format is added or removed.
//!
//! Decode scores share one reference across both directions and both modes, so
//! a forward score and a backward score are directly comparable. A fast decoder
//! is never slower than the balanced one, so turning on priority speed can only
//! raise a score or leave it unchanged.

use lzan_c64::{Direction, Format};

/// `(format, packing ms, forward standard, forward fast, backward standard,
/// backward fast)`. Lower is faster. A fast column equals its standard sibling
/// when the format has no faster decoder in that direction.
const BENCH: &[(Format, f64, f64, f64, f64, f64)] = &[
    (Format::LzanFull, 910.3, 1_285_006.0, 653_218.0, 2_168_212.0, 2_168_212.0),
    (Format::LzanMin, 107.0, 582_416.0, 494_006.0, 1_094_913.0, 1_094_913.0),
    (Format::Exomizer, 540.1, 812_322.0, 812_322.0, 797_476.0, 797_476.0),
    (Format::Subsizer, 275.1, 1_179_343.0, 1_179_343.0, 744_057.0, 744_057.0),
    (Format::Shrinkler, 220.3, 25_467_199.0, 25_467_199.0, 26_851_420.0, 26_851_420.0),
    (Format::Zx02, 297.1, 924_794.0, 924_794.0, 1_085_095.0, 1_085_095.0),
    (Format::Zx0, 104.4, 589_390.0, 589_390.0, 1_052_567.0, 1_052_567.0),
    (Format::Lzsa2, 76.2, 792_226.0, 792_226.0, 784_592.0, 784_592.0),
    (Format::Lzsa1, 79.6, 767_146.0, 399_883.0, 751_226.0, 751_226.0),
    (Format::Aplib, 259.2, 739_369.0, 739_369.0, 1_096_621.0, 1_096_621.0),
    (Format::TsCrunch, 14.1, 368_724.0, 327_836.0, 405_543.0, 405_543.0),
    (Format::ByteBoozer2, 3.8, 575_914.0, 575_914.0, 1_003_083.0, 1_003_083.0),
    (Format::PuCrunch, 163.2, 1_708_180.0, 1_665_311.0, 1_861_707.0, 1_861_707.0),
    (Format::Upkr, 717.8, 10_492_914.0, 10_492_914.0, 10_486_594.0, 10_486_594.0),
    (Format::Bolt, 3.2, 356_042.0, 315_192.0, 299_825.0, 276_410.0),
];

fn lookup(f: Format) -> Option<(f64, f64, f64, f64, f64)> {
    BENCH.iter().find(|e| e.0 == f).map(|e| (e.1, e.2, e.3, e.4, e.5))
}

/// Map a lower is better `value` onto 1 to 10 given the log bounds of its
/// reference field. The fastest maps to 10 and the slowest to 1.
fn score_on(value: f64, lo: f64, hi: f64) -> f64 {
    if hi <= lo {
        return 10.0;
    }
    let l = value.max(1.0).ln();
    (1.0 + 9.0 * (hi - l) / (hi - lo)).clamp(1.0, 10.0)
}

/// Log bounds over all four decode columns of every format. This is the single
/// reference shared by both directions and both modes.
fn decode_bounds() -> (f64, f64) {
    let mut lo = f64::INFINITY;
    let mut hi = f64::NEG_INFINITY;
    for e in BENCH {
        for v in [e.2, e.3, e.4, e.5] {
            let l = v.max(1.0).ln();
            lo = lo.min(l);
            hi = hi.max(l);
        }
    }
    (lo, hi)
}

fn pack_bounds() -> (f64, f64) {
    let mut lo = f64::INFINITY;
    let mut hi = f64::NEG_INFINITY;
    for e in BENCH {
        let l = e.1.max(1.0).ln();
        lo = lo.min(l);
        hi = hi.max(l);
    }
    (lo, hi)
}

/// Packing speed score, 1 to 10, where 10 is fastest to compress.
pub fn pack_speed(f: Format) -> Option<f64> {
    let (ms, _, _, _, _) = lookup(f)?;
    let (lo, hi) = pack_bounds();
    Some(score_on(ms, lo, hi))
}

/// Unpacking speed score, 1 to 10, for the decoder that will actually run:
/// the column for `direction`, fast when `priority_speed` is set.
pub fn decr_speed(f: Format, priority_speed: bool, direction: Direction) -> Option<f64> {
    let (_, fwd_std, fwd_fast, bwd_std, bwd_fast) = lookup(f)?;
    let value = match (direction, priority_speed) {
        (Direction::Forward, false) => fwd_std,
        (Direction::Forward, true) => fwd_fast,
        (Direction::Backward, false) => bwd_std,
        (Direction::Backward, true) => bwd_fast,
    };
    let (lo, hi) = decode_bounds();
    Some(score_on(value, lo, hi))
}

/// `"9.5/10"`, or `"-"` when the format was not benchmarked.
pub fn score_str(score: Option<f64>) -> String {
    match score {
        Some(s) => format!("{:.1}/10", s),
        None => "-".to_string(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const DIRS: [Direction; 2] = [Direction::Forward, Direction::Backward];

    fn all_decode() -> Vec<f64> {
        let mut v = Vec::new();
        for &(f, ..) in BENCH {
            for &d in &DIRS {
                v.push(decr_speed(f, false, d).unwrap());
                v.push(decr_speed(f, true, d).unwrap());
            }
        }
        v
    }

    /// Each metric reaches a 10 for the fastest and a 1 for the slowest.
    #[test]
    fn scale_endpoints() {
        let min = |v: &[f64]| v.iter().cloned().fold(f64::MAX, f64::min);
        let max = |v: &[f64]| v.iter().cloned().fold(f64::MIN, f64::max);

        let pack: Vec<f64> = BENCH.iter().filter_map(|e| pack_speed(e.0)).collect();
        assert!((max(&pack) - 10.0).abs() < 1e-9 && (min(&pack) - 1.0).abs() < 1e-9);

        let dec = all_decode();
        assert!((max(&dec) - 10.0).abs() < 1e-9, "fastest decode must be 10.0");
        assert!((min(&dec) - 1.0).abs() < 1e-9, "slowest decode must be 1.0");
    }

    #[test]
    fn every_format_is_benched() {
        for &(f, ..) in BENCH {
            assert!(pack_speed(f).is_some());
            for &d in &DIRS {
                assert!(decr_speed(f, false, d).is_some());
                assert!(decr_speed(f, true, d).is_some());
            }
        }
    }

    /// Priority speed can only raise a score or leave it unchanged, in either
    /// direction, because a fast decoder is never slower than the balanced one.
    #[test]
    fn priority_speed_never_lowers_any_score() {
        for &(f, ..) in BENCH {
            for &d in &DIRS {
                let std = decr_speed(f, false, d).unwrap();
                let fast = decr_speed(f, true, d).unwrap();
                assert!(fast >= std - 1e-9, "priority speed lowered {f:?} {d:?}");
            }
        }
    }

    /// A backward decode is scored by its own cycles, not the forward number.
    #[test]
    fn backward_has_its_own_score() {
        let f = decr_speed(Format::Bolt, true, Direction::Forward).unwrap();
        let b = decr_speed(Format::Bolt, true, Direction::Backward).unwrap();
        assert!((f - b).abs() > 1e-6, "bolt forward and backward must differ");
    }
}
