//! Non-arbitrary 1–10 speed scores for the cruncher picker.
//!
//! The raw numbers below were measured by `examples/bench_ranks.rs` on one
//! fixed 14 KB payload (forward layout, illegal opcodes allowed): wall-clock
//! packing time and the 6502 cycles the whole generated SFX takes to unpack on
//! the c64-debug emulator. Regenerate them with
//! `cargo run --release --example bench_ranks` and paste the printed table.
//!
//! Both metrics span roughly two decades across the formats (packing 4–940 ms;
//! unpacking 0.37–25 M cycles), so a *linear* map to 1–10 would pile almost
//! every format near 10 and leave only the range-coders (Shrinkler, upkr) at
//! the bottom. We map on a **log scale** instead: the fastest format still
//! scores exactly 10 and the slowest exactly 1, but the middle of the field is
//! legible. Scores are derived from the constants at runtime, so the scale
//! re-normalizes automatically if a format is added or removed.

use lzan_c64::Format;

/// `(format, packing time ms, decrunch 6502 cycles)` — lower is faster in both
/// columns. Source: `examples/bench_ranks.rs`.
// Fully re-measured July 2026 on an idle machine after the boot-glue
// overhaul (payload-at-final-address backward layout, compact copy loops)
// and the deep decoder-body size golf of ALL formats. The golf traded
// cycles for bytes in several decoders (shared JSR bit-fetch paths), which
// these numbers — and thus the GUI's 1-10 unpacking-speed ranks — reflect
// honestly. Cycles are deterministic emulator counts for the whole SFX.
const BENCH: &[(Format, f64, f64)] = &[
    (Format::LzanFull, 879.0, 1_285_006.0),
    (Format::LzanMin, 104.8, 582_416.0),
    (Format::Exomizer, 558.0, 812_322.0),
    (Format::Subsizer, 310.8, 1_179_343.0),
    (Format::Shrinkler, 264.1, 25_467_199.0),
    (Format::Zx02, 295.5, 924_794.0),
    (Format::Zx0, 104.7, 589_390.0),
    (Format::Lzsa2, 73.1, 792_226.0),
    (Format::Lzsa1, 67.6, 767_146.0),
    (Format::Aplib, 262.2, 739_369.0),
    (Format::TsCrunch, 15.8, 368_724.0),
    (Format::ByteBoozer2, 4.0, 575_914.0),
    // PuCrunch re-measured after the 211 B zp-stack forward decoder (selected
    // automatically when it fits the $0100 slot); all other rows are the
    // unaffected raw data from the same idle-machine run.
    (Format::PuCrunch, 152.7, 1_708_180.0),
    (Format::Upkr, 722.2, 10_492_914.0),
];

fn lookup(f: Format) -> Option<(f64, f64)> {
    BENCH.iter().find(|e| e.0 == f).map(|e| (e.1, e.2))
}

/// Map a lower-is-better `value` to a 1..=10 score on a log scale across the
/// full benched field: the smallest (fastest) maps to 10, the largest to 1.
fn score_log(value: f64, column: impl Fn(&(Format, f64, f64)) -> f64) -> f64 {
    let mut lo = f64::INFINITY;
    let mut hi = f64::NEG_INFINITY;
    for e in BENCH {
        let l = column(e).max(1.0).ln();
        lo = lo.min(l);
        hi = hi.max(l);
    }
    if hi <= lo {
        return 10.0;
    }
    let l = value.max(1.0).ln();
    (1.0 + 9.0 * (hi - l) / (hi - lo)).clamp(1.0, 10.0)
}

/// Packing-speed score, 1..=10 (10 = fastest to compress). `None` if unbenched.
pub fn pack_speed(f: Format) -> Option<f64> {
    let (ms, _) = lookup(f)?;
    Some(score_log(ms, |e| e.1))
}

/// Decrunch-speed score, 1..=10 (10 = fewest 6502 cycles to unpack).
pub fn decr_speed(f: Format) -> Option<f64> {
    let (_, cyc) = lookup(f)?;
    Some(score_log(cyc, |e| e.2))
}

/// `"9/10"`, or `"—"` when the format was not benchmarked.
pub fn score_str(score: Option<f64>) -> String {
    match score {
        Some(s) => format!("{}/10", s.round() as u32),
        None => "—".to_string(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The field always contains exactly one 10 (fastest) and one 1 (slowest)
    /// per metric — the property the picker promises the user.
    #[test]
    fn scale_has_a_one_and_a_ten() {
        for col in [
            pack_speed as fn(Format) -> Option<f64>,
            decr_speed as fn(Format) -> Option<f64>,
        ] {
            let scores: Vec<f64> = BENCH.iter().filter_map(|e| col(e.0)).collect();
            let max = scores.iter().cloned().fold(f64::MIN, f64::max);
            let min = scores.iter().cloned().fold(f64::MAX, f64::min);
            assert!(
                (max - 10.0).abs() < 1e-9,
                "fastest must score 10, got {max}"
            );
            assert!((min - 1.0).abs() < 1e-9, "slowest must score 1, got {min}");
        }
    }

    #[test]
    fn every_format_is_benched() {
        // The picker shows all of these; each must have a score.
        for &(f, _, _) in BENCH {
            assert!(pack_speed(f).is_some());
            assert!(decr_speed(f).is_some());
        }
    }
}
