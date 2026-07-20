//! Self-extracting C64 program builder: multi-PRG 64 KB image + in-place layouts.
//!
//! ## Memory image
//!
//! Any number of `.prg` files are added into a 64 KB buffer initialized to
//! zeroes (zeroes in the gaps compress to almost nothing). The compressed span
//! runs from the lowest region start to the highest region end.
//!
//! ## The two layouts
//!
//! **Forward / top** — used whenever the span's last byte is at or below
//! `$FFF0`: the packed stream is moved so its last byte lands on `$FFFF`
//! (under the KERNAL; `$01=$30` makes it plain RAM) and a forward decoder
//! unpacks down at the span start. Since compressed < uncompressed, the read
//! head stays ahead of the write head; the `$FFFF`-alignment leaves a
//! `$10000 - span_end` safety gap at the tightest point (≥ 16 bytes per the
//! `$FFF0` rule).
//!
//! **Backward / bottom** — used when the span reaches past `$FFF0` (so there
//! is no room above it): the packed stream is moved DOWN to start `margin`
//! (default 32) bytes below the span start, and a backward decoder unpacks
//! from `$FFFF` downwards. Write head trails read head with a gap that shrinks
//! to exactly `margin` at the end.
//!
//! In both layouts the decruncher is staged outside the output region
//! (default `$0100` when it fits under `$E0` bytes, else a free low slot); the
//! payload move runs from a relocated mover (default `$02A7`) when its
//! destination overlaps the loaded program image, else inline from the main
//! program. Init is always `SEI` + `LDA #$30` + `STA $01` (all 64 KB RAM).

use lzan_c64::{
    compress_for, pick_routine, pick_speed_routine, pick_zp_stack_routine, Decruncher, Direction,
    Format, PayloadAbi, RoutineSpec, Variant,
};

/// Backward-layout safety margin: the packed stream starts this many bytes
/// below the span start (the final read/write gap).
pub const DEFAULT_MARGIN: u16 = 32;
/// Default clearance for the "no move" in-place layout (see `Placement::clearance`).
pub const DEFAULT_CLEARANCE: u16 = 32;
/// Default relocated-mover address (user memory map: `$02A7-$02FF` is free).
pub const DEFAULT_MOVER: u16 = 0x02A7;
/// `$01` value written at init (all RAM, needed while decrunching). Also the
/// default value to leave it at before the final `JMP`, in which case no
/// extra restore code is needed since `$01` is already at that value.
pub const INIT_BANK: u8 = 0x30;
/// BASIC interpreter main loop (NEWSTT). The RUN option jumps here after
/// `JSR $A659` (CLR) has pointed TXTPTR at the program start, which reliably
/// runs a decompressed BASIC program. (A bare `JMP $A871` — the RUN command
/// itself — is not safe from a cold ML entry: it branches on the caller's Z
/// flag and returns through CLR's `RTS` expecting the interpreter's stack
/// frame.)
pub const RUN_BASIC_LOOP: u16 = 0xA7AE;
/// BASIC CLR routine: sets TXTPTR to the program start and clears
/// variables/arrays/strings from VARTAB. `JSR`ed before jumping to the loop.
const BASIC_CLR: u16 = 0xA659;
/// Banking value the RUN option needs at the final `JMP`: BASIC+KERNAL+IO in.
pub const RUN_BASIC_BANK: u8 = 0x37;
/// The `$02A7` mover slot ends here (exclusive).
const MOVER_SLOT_END: u32 = 0x0300;
/// Forward/top layout applies while the span's last byte is <= $FFF0
/// (exclusive end <= $FFF1); past that there is no top gap and the backward
/// layout is required.
pub const FORWARD_MAX_END: u32 = 0xFFF1;
/// Wrapper bytes added around the routine body when staging (JSR entry +
/// JMP done), with slack. The option-dependent epilogue is counted
/// separately by [`epilogue_len`].
const STAGE_WRAPPER: u32 = 16;
/// The small-decoder stack-page slot is `$0100-$01DF` (leaving `$01E0-$01FF`
/// for the CPU stack). A staged blob no larger than this is placed at `$0100`
/// automatically; a bigger one is relocated.
pub const STACK_PAGE_SLOT: u32 = 0xE0;

// ---------------------------------------------------------------------------
// Memory image
// ---------------------------------------------------------------------------

/// One added `.prg`: its load address and program bytes (load address bytes
/// stripped). Kept verbatim so the 64 KB buffer can be rebuilt after removals.
#[derive(Clone)]
pub struct Region {
    pub name: String,
    pub load: u16,
    pub data: Vec<u8>,
}

impl Region {
    pub fn start(&self) -> u32 {
        self.load as u32
    }
    /// Exclusive end (can be $10000).
    pub fn end(&self) -> u32 {
        self.load as u32 + self.data.len() as u32
    }
}

/// The 64 KB image under construction. Regions are kept in insertion order
/// (later additions overwrite earlier ones where they overlap); use
/// [`MemoryImage::sorted_indices`] for display.
#[derive(Clone, Default)]
pub struct MemoryImage {
    regions: Vec<Region>,
}

impl MemoryImage {
    pub fn add_prg(&mut self, name: &str, bytes: &[u8]) -> Result<(), String> {
        if bytes.len() < 3 {
            return Err(format!(
                "{name}: too short to be a .prg (2-byte load address + data)."
            ));
        }
        let load = u16::from_le_bytes([bytes[0], bytes[1]]);
        let data = bytes[2..].to_vec();
        if load as usize + data.len() > 0x1_0000 {
            return Err(format!(
                "{name}: loads at ${load:04X} with {} bytes — crosses $FFFF.",
                data.len()
            ));
        }
        self.regions.push(Region {
            name: name.to_string(),
            load,
            data,
        });
        Ok(())
    }

    pub fn remove(&mut self, idx: usize) {
        if idx < self.regions.len() {
            self.regions.remove(idx);
        }
    }

    /// Re-window region `idx` to the address range `[new_start, new_end_excl)`.
    /// Bytes shared with the old range are preserved; bytes the new range adds
    /// (growing) are zero-filled, and bytes the new range drops (shrinking) are
    /// discarded — if such dropped bytes sit between other regions they become
    /// part of the zero-filled span gap automatically. This can adjust a
    /// program's load address or trim or extend a chunk. `new_end_excl` may be
    /// `$10000` (a region ending at `$FFFF`).
    pub fn resize_region(
        &mut self,
        idx: usize,
        new_start: u16,
        new_end_excl: u32,
    ) -> Result<(), String> {
        let start = new_start as u32;
        if new_end_excl <= start {
            return Err("the end address must be at or after the start".into());
        }
        if new_end_excl > 0x1_0000 {
            return Err("the end address is above $FFFF".into());
        }
        let r = self.regions.get_mut(idx).ok_or("no such region")?;
        let old_start = r.load as u32;
        let old_end = old_start + r.data.len() as u32;
        let mut new_data = vec![0u8; (new_end_excl - start) as usize];
        // Copy the overlap of the old and new windows into place; the remainder
        // stays zero.
        let ov_start = old_start.max(start);
        let ov_end = old_end.min(new_end_excl);
        if ov_start < ov_end {
            let src = (ov_start - old_start) as usize;
            let dst = (ov_start - start) as usize;
            let n = (ov_end - ov_start) as usize;
            new_data[dst..dst + n].copy_from_slice(&r.data[src..src + n]);
        }
        r.load = new_start;
        r.data = new_data;
        Ok(())
    }

    pub fn is_empty(&self) -> bool {
        self.regions.is_empty()
    }

    pub fn regions(&self) -> &[Region] {
        &self.regions
    }

    /// Region indices sorted by start address (for the always-sorted list).
    pub fn sorted_indices(&self) -> Vec<usize> {
        let mut idx: Vec<usize> = (0..self.regions.len()).collect();
        idx.sort_by_key(|&i| (self.regions[i].start(), self.regions[i].end()));
        idx
    }

    /// Lowest start .. highest end (exclusive) across all regions.
    pub fn span(&self) -> Option<(u32, u32)> {
        let lo = self.regions.iter().map(|r| r.start()).min()?;
        let hi = self.regions.iter().map(|r| r.end()).max()?;
        Some((lo, hi))
    }

    /// The span slice of the zero-initialized 64 KB buffer, regions applied in
    /// insertion order (later overwrites earlier).
    pub fn span_buffer(&self) -> Option<(u32, Vec<u8>)> {
        let (lo, hi) = self.span()?;
        let mut buf = vec![0u8; (hi - lo) as usize];
        for r in &self.regions {
            let at = (r.start() - lo) as usize;
            buf[at..at + r.data.len()].copy_from_slice(&r.data);
        }
        Some((lo, buf))
    }

    /// The full 64 KB address space with all regions painted in insertion
    /// order (later regions win where they overlap, like [`Self::span_buffer`]);
    /// everything else is zero. Backs the memory viewer/editor.
    pub fn full_buffer(&self) -> Vec<u8> {
        let mut buf = vec![0u8; 0x1_0000];
        for r in &self.regions {
            buf[r.start() as usize..r.end() as usize].copy_from_slice(&r.data);
        }
        buf
    }

    /// Apply a memory-editor change set: `old` is the last synced
    /// [`Self::full_buffer`] snapshot, `new` the edited buffer (both 64 KB).
    /// Bytes covered by a region are written into the LAST region painted
    /// there (the visible one); contiguous edited runs outside every region
    /// become new regions named `edit @ $XXXX`, so the change survives the
    /// span rebuild and shows up in the region list. Returns log lines
    /// describing what was applied.
    pub fn apply_edits(&mut self, old: &[u8], new: &[u8]) -> Vec<String> {
        assert_eq!(old.len(), 0x1_0000);
        assert_eq!(new.len(), 0x1_0000);
        // Which region (by index) is visible at each changed address.
        let owner = |addr: usize| -> Option<usize> {
            self.regions
                .iter()
                .rposition(|r| (r.start()..r.end()).contains(&(addr as u32)))
        };
        // Pass 1: maximal changed runs, split wherever the owning region
        // changes (ownership judged against the PRE-edit region set).
        let mut runs: Vec<(usize, usize, Option<usize>)> = Vec::new();
        let mut addr = 0usize;
        while addr < 0x1_0000 {
            if old[addr] == new[addr] {
                addr += 1;
                continue;
            }
            let own = owner(addr);
            let start = addr;
            while addr < 0x1_0000 && old[addr] != new[addr] && owner(addr) == own {
                addr += 1;
            }
            runs.push((start, addr, own));
        }
        // Pass 2: apply.
        let mut log = Vec::new();
        for (start, end, own) in runs {
            match own {
                Some(i) => {
                    let r = &mut self.regions[i];
                    let at = (start as u32 - r.start()) as usize;
                    r.data[at..at + (end - start)].copy_from_slice(&new[start..end]);
                    log.push(format!(
                        "Edited ${start:04X}-${:04X} ({} B) in {}",
                        end - 1,
                        end - start,
                        r.name
                    ));
                }
                None => {
                    let name = format!("edit @ ${start:04X}");
                    self.regions.push(Region {
                        name: name.clone(),
                        load: start as u16,
                        data: new[start..end].to_vec(),
                    });
                    log.push(format!(
                        "New region {name} (${start:04X}-${:04X}, {} B)",
                        end - 1,
                        end - start
                    ));
                }
            }
        }
        log
    }

    /// Pairs of region indices that overlap in memory (for UI warnings).
    pub fn overlap_pairs(&self) -> Vec<(usize, usize)> {
        let mut out = Vec::new();
        for i in 0..self.regions.len() {
            for j in i + 1..self.regions.len() {
                let (a, b) = (&self.regions[i], &self.regions[j]);
                if a.start() < b.end() && b.start() < a.end() {
                    out.push((i, j));
                }
            }
        }
        out
    }
}

// ---------------------------------------------------------------------------
// Planning
// ---------------------------------------------------------------------------

/// Unpack-direction selection: the span rule by default, or a manual
/// override (validated for feasibility).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum DirectionChoice {
    /// Span rule: forward while the last byte is <= $FFF0, else backward.
    #[default]
    Auto,
    Forward,
    Backward,
}

/// Per-crunch decoder-tailoring selection. Only Exomizer streams tailor today;
/// every other format ignores this. The library (lzan-c64) forces an explicit
/// choice per build; this adds the GUI's "try both" convenience on top.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum TailoringChoice {
    /// Build both the standard and the trait-tailored decoder and keep the
    /// smaller (ties keep standard). Never larger than standard.
    #[default]
    Auto,
    /// Always use the routine's standard decoder body.
    Standard,
    /// Always use the trait-tailored body when the stream admits one (falls
    /// back to standard when it does not, or if the tailored body fails to
    /// assemble).
    Tailored,
}

/// User-adjustable placements; `None` = computed default.
#[derive(Clone, Copy, Debug)]
pub struct Placement {
    pub decruncher: Option<u16>,
    pub mover: Option<u16>,
    /// Scratch buffer base for formats that need one (exomizer/subsizer
    /// tables, shrinkler/upkr probability RAM).
    pub scratch: Option<u16>,
    /// Backward-layout safety margin (bytes below span start).
    pub margin: u16,
    pub direction: DirectionChoice,
    /// `$01` value written just before the final `JMP` to the start address.
    /// Defaults to [`INIT_BANK`], in which case no extra code is emitted
    /// (the value is already there from init).
    pub bank_at_jmp: u8,
    /// `CLI` right before the final `JMP`. Interrupts stay disabled (the
    /// init `SEI`) when `false`, since only the caller's start routine knows
    /// what interrupt state it expects.
    pub cli_before_jmp: bool,
    /// When `Some(addr)`, restore the BASIC end-of-program pointer `$2D/$2E`
    /// (VARTAB) to `addr` just before the final `JMP`, so a decompressed
    /// BASIC program can be RUN (its CLR derives the array/string pointers
    /// from VARTAB).
    pub restore_basic_end: Option<u16>,
    /// Emit `JSR $A659` (BASIC CLR) before the final `JMP`. Used by the RUN
    /// option so the interpreter loop (`RUN_BASIC_LOOP`) starts from the
    /// program's first line. Requires ROM banked in (`bank_at_jmp = $37`).
    pub basic_clr: bool,
    /// Allow the embedded decruncher to use undocumented (illegal) 6502
    /// opcodes. `true` (default) picks the size/speed baseline; `false` picks a
    /// legal-only decoder (same compressed stream, slightly larger decoder) so
    /// the output runs on CPUs/emulators without illegal opcodes.
    pub allow_illegal: bool,
    /// TSCrunch in-place layout: bytes the packed stream is placed ABOVE the
    /// end-aligned reference position. Computed per stream by
    /// [`inplace_effective_placement`] (the 6502 token copies overshoot the
    /// reference layout's boundary invariant); not a user setting.
    pub tsc_shift: u16,
    /// Per-crunch decoder tailoring (Exomizer only). `Auto` (default) builds
    /// both the standard and the tailored decoder and keeps the smaller.
    pub tailoring: TailoringChoice,
    /// Clearance in bytes (default [`DEFAULT_CLEARANCE`]) for the "no move"
    /// in-place layout: when the whole program image (decoder + embedded
    /// payload) sits at least this far clear of the output span — and the
    /// scratch buffer clear of both — the payload is read where it was loaded
    /// with no move at all. The gap absorbs any decoder write overshoot so the
    /// decompressed output can never reach the payload or the decoder.
    pub clearance: u16,
    /// Pick the fastest decoder for each format instead of the balanced one.
    /// The packed stream is unchanged; only the embedded decoder body differs.
    pub priority_speed: bool,
}

impl Default for Placement {
    fn default() -> Self {
        Placement {
            decruncher: None,
            mover: None,
            scratch: None,
            margin: DEFAULT_MARGIN,
            direction: DirectionChoice::Auto,
            bank_at_jmp: INIT_BANK,
            cli_before_jmp: false,
            restore_basic_end: None,
            basic_clr: false,
            allow_illegal: true,
            tsc_shift: 0,
            tailoring: TailoringChoice::default(),
            clearance: DEFAULT_CLEARANCE,
            priority_speed: false,
        }
    }
}

/// Span-level feasibility of one direction. Forward needs a top gap (last
/// byte <= $FFF0); backward needs room for the packed stream below the span
/// start (checked precisely in `place`, but a span starting at/below
/// $0200+margin can never work).
fn direction_feasible(
    span: (u32, u32),
    direction: Direction,
    placement: &Placement,
) -> Result<(), String> {
    let (span_start, span_end) = span;
    match direction {
        Direction::Forward => {
            if span_end > FORWARD_MAX_END {
                Err(format!(
                    "forward is impossible: the span ends at ${:04X} (above $FFF0) — there is \
                     no room for packed data above the output",
                    span_end - 1
                ))
            } else {
                Ok(())
            }
        }
        Direction::Backward => {
            if span_start < 0x0200 + placement.margin as u32 {
                Err(format!(
                    "backward is impossible: the span starts at ${span_start:04X} — packed data \
                     must sit {} bytes below the span start and above $0200",
                    placement.margin
                ))
            } else {
                Ok(())
            }
        }
    }
}

/// Resolve the direction from a MANUAL choice (used by the GUI to explain an
/// infeasible selection). `Auto` resolves to the span rule's preference here;
/// the format-aware fallback lives in [`direction_candidates`].
pub fn resolve_direction(span: (u32, u32), placement: &Placement) -> Result<Direction, String> {
    let d = match placement.direction {
        DirectionChoice::Auto => direction_for_span(span.1),
        DirectionChoice::Forward => Direction::Forward,
        DirectionChoice::Backward => Direction::Backward,
    };
    if placement.direction != DirectionChoice::Auto {
        direction_feasible(span, d, placement)?;
    }
    Ok(d)
}

/// Directions to try, in preference order, each pre-checked for span
/// feasibility, routine existence and worst-case placement.
///
/// Manual choice: exactly that direction (with its verbatim error).
/// Auto: the span rule's direction first, then the OTHER direction as a
/// fallback — so a format that only fits one way (e.g. Shrinkler on a big low
/// span, whose 1536-byte buffer only finds room above the span end in the
/// backward layout) is offered with that direction instead of being hidden.
/// Directions the build will try, in order, plus — for Auto — the reason a
/// direction was pruned (surfaced when the surviving candidates also fail,
/// so the pruned direction's problem is never silently masked).
struct DirectionPlan {
    dirs: Vec<Direction>,
    skipped: Vec<String>,
}

fn direction_candidates(
    span: (u32, u32),
    format: Format,
    placement: &Placement,
) -> Result<DirectionPlan, String> {
    let try_one = |d: Direction| -> Result<Direction, String> {
        direction_feasible(span, d, placement)?;
        if !has_routine(format, d, placement.allow_illegal) {
            return Err(format!(
                "{} has no {} {}decoder",
                format.as_str(),
                d.as_str(),
                if placement.allow_illegal {
                    ""
                } else {
                    "legal "
                }
            ));
        }
        let embedded_est = clen_worst_case(span.1 - span.0);
        place(span, embedded_est, format, d, placement, false)?;
        Ok(d)
    };
    match placement.direction {
        DirectionChoice::Forward => Ok(DirectionPlan {
            dirs: vec![try_one(Direction::Forward)?],
            skipped: Vec::new(),
        }),
        DirectionChoice::Backward => Ok(DirectionPlan {
            dirs: vec![try_one(Direction::Backward)?],
            skipped: Vec::new(),
        }),
        DirectionChoice::Auto => {
            let preferred = direction_for_span(span.1);
            let other = match preferred {
                Direction::Forward => Direction::Backward,
                Direction::Backward => Direction::Forward,
            };
            let mut out = Vec::new();
            let mut skipped = Vec::new();
            for d in [preferred, other] {
                match try_one(d) {
                    Ok(d) => out.push(d),
                    Err(e) => skipped.push(format!("{}: {e}", d.as_str())),
                }
            }
            if out.is_empty() {
                Err(skipped.join("; "))
            } else {
                Ok(DirectionPlan { dirs: out, skipped })
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Free-space model
// ---------------------------------------------------------------------------
//
// Reserved during decode (automatic placement never touches these):
//   $0000-$00FF  zero page
//   $0100-$01DF  decoder-only slot (small decoders, per the $E0 rule)
//   $01E0-$01FF  CPU stack working area
//   $0200-$02A6  system area — manual override only
//   $02A7-$02FF  mover slot
//   $0300-$0333  kernal vectors
// The automatic pool is $0334-$03FF plus $0400-$FFFF, minus the output span,
// the packed stream's decode-time position, and the loaded program image
// (blobs are installed while the image is still intact, so nothing may be
// placed on top of it).

/// Half-open address block.
#[derive(Clone, Copy, PartialEq)]
struct Block {
    start: u32,
    end: u32,
}

impl Block {
    fn overlaps(&self, o: &Block) -> bool {
        self.start < o.end && o.start < self.end
    }
}

/// Cut `cut` out of every segment (keeping order and ascending addresses).
fn subtract(segs: Vec<Block>, cut: Block) -> Vec<Block> {
    let mut out = Vec::with_capacity(segs.len() + 1);
    for s in segs {
        if !s.overlaps(&cut) {
            out.push(s);
            continue;
        }
        if s.start < cut.start {
            out.push(Block {
                start: s.start,
                end: cut.start,
            });
        }
        if cut.end < s.end {
            out.push(Block {
                start: cut.end,
                end: s.end,
            });
        }
    }
    out
}

/// The automatic placement pool: $0334-$03FF and $0400 upward, minus the
/// given cut-outs.
fn auto_pool(cuts: &[Block]) -> Vec<Block> {
    let mut segs = vec![
        Block {
            start: 0x0334,
            end: 0x0400,
        },
        Block {
            start: 0x0400,
            end: 0x1_0000,
        },
    ];
    for cut in cuts {
        segs = subtract(segs, *cut);
    }
    segs
}

/// First-fit into the pool, skipping already-placed blocks; `page` rounds the
/// candidate up to a page boundary.
fn first_fit(pool: &[Block], used: &[Block], len: u32, page: bool) -> Option<u32> {
    for seg in pool {
        let mut subs = vec![*seg];
        for u in used {
            subs = subtract(subs, *u);
        }
        for sub in subs {
            let start = if page {
                (sub.start + 0xFF) & !0xFF
            } else {
                sub.start
            };
            if start + len <= sub.end {
                return Some(start);
            }
        }
    }
    None
}

/// Worst-case compressed size (every format stays well under input + 1/16 +
/// 64 even on pure noise). Used pre-compression for availability/preview; the
/// real length is always smaller, so a preview that fits is a build that fits.
fn clen_worst_case(span_len: u32) -> u32 {
    span_len + span_len / 16 + 64
}

/// Conservative end of the loaded program image: stub + init/install code +
/// staged blobs + payload, with slack. Over-estimating is safe (it only makes
/// placement more conservative and the mover trigger earlier).
fn image_end_estimate(staged: u32, embedded_len: u32) -> u32 {
    0x0801 + 12 + 512 + staged + 96 + embedded_len
}

/// Pre-compression plan (everything derivable without running the encoder),
/// shown live in the UI and used to filter the format dropdown.
pub struct PlanPreview {
    pub direction: Direction,
    /// Estimated staged decruncher size (routine + wrapper).
    pub staged_size: u32,
    /// Resolved decruncher address (override or auto).
    pub decruncher_at: Option<u16>,
    /// Resolved mover address (only used if the payload move needs relocation).
    pub mover_at: u16,
    /// Scratch buffer placement, if the format needs one: (address, length).
    pub scratch: Option<(u16, u16)>,
    /// Why this format cannot be used for the current image (None = usable).
    pub unavailable: Option<String>,
}

fn has_routine(format: Format, direction: Direction, allow_illegal: bool) -> bool {
    pick_routine(format, direction, allow_illegal).is_some()
}

/// Direction per the span rule: forward/top while the last byte is <= $FFF0,
/// backward/bottom past that.
pub fn direction_for_span(span_end: u32) -> Direction {
    if span_end <= FORWARD_MAX_END {
        Direction::Forward
    } else {
        Direction::Backward
    }
}

/// Shared placement computation for preview (worst-case packed size,
/// `exact = false`) and build (real packed size, `exact = true`). Size
/// feasibility ("too big") is only decidable with the real length, so those
/// checks run only when `exact`; the preview clamps the estimate instead so a
/// worst-case guess never rejects a format the real build would accept.
fn place(
    span: (u32, u32),
    embedded_len: u32,
    format: Format,
    direction: Direction,
    placement: &Placement,
    exact: bool,
) -> Result<Placed, String> {
    let (span_start, span_end) = span;
    // Priority speed picks the opt-speed decoder where one exists, otherwise
    // the balanced baseline.
    let baseline = if placement.priority_speed {
        pick_speed_routine(format, direction, placement.allow_illegal)
    } else {
        pick_routine(format, direction, placement.allow_illegal)
    }
    .ok_or_else(|| format!("no {} routine for {}", direction.as_str(), format.as_str()))?;
    // The pre-JMP epilogue (banking / CLI / $2D-$2E restore / CLR) is emitted
    // INSIDE the staged blob, between the JSR entry and the decoder body, so
    // its bytes must be reserved too — otherwise a scratch buffer placed just
    // above the decoder can silently overlap the decoder's tail.
    let staged_for =
        |spec: &RoutineSpec| spec.code_bytes as u32 + STAGE_WRAPPER + epilogue_len(placement);
    // The staged blob's EXACT wrapper is JSR entry (3) + JMP done (3);
    // STAGE_WRAPPER carries extra planning slack on top. The zp-stack
    // eligibility check uses the exact number (+2 safety) — the whole point
    // of that variant is squeezing into the $0100 slot, and the slack was
    // costing it the fit (211 + 16 > 224, while the real blob is 217).
    const ZP_STACK_WRAPPER: u32 = 8;
    let staged_zp =
        |spec: &RoutineSpec| spec.code_bytes as u32 + ZP_STACK_WRAPPER + epilogue_len(placement);
    // Prefer the extra-small stack-page variant (pucrunch forward) when the
    // decoder address is AUTO and the whole staged blob fits the $0100 slot —
    // it trades cycles for the bytes that let it live in guaranteed-free RAM.
    // Any manual decoder address, or a blob that does not fit (a big epilogue
    // can push it out), falls back to the standard baseline. Same stream
    // either way, so this is placement-local.
    let spec = match pick_zp_stack_routine(format, direction, placement.allow_illegal) {
        Some(zp) if placement.decruncher.is_none() && staged_zp(zp) <= STACK_PAGE_SLOT => zp,
        _ => baseline,
    };
    let base_staged = if spec.variant == Variant::ZpStack {
        staged_zp(spec)
    } else {
        staged_for(spec)
    };

    // Packed stream position at decode time.
    //
    // TSCrunch's "backward" routine is the format's native in-place mode: it
    // reads and writes FORWARD, and requires the packed blob END-ALIGNED with
    // the output end (`tscrunch -p -i` layout; the read head then always stays
    // ahead of the write head). Bottom-placement would let the writer destroy
    // unread stream bytes.
    let tscrunch_inplace = format == Format::TsCrunch && direction == Direction::Backward;
    let (packed_start, packed_end) = if tscrunch_inplace {
        // The reference `tscrunch -p -i` layout end-aligns the packed blob
        // with the output end (load_to = out_end - clen); the encoder's
        // suffix-trim keeps the read head ahead of the write head at token
        // BOUNDARIES, and the incompressible remainder is stored as a RAW
        // tail that lands (nearly) on itself. The 6502 token copies overshoot
        // that boundary invariant, so `inplace_effective_placement` computes a
        // per-stream upward shift (`tsc_shift`, usually 0) that the layout
        // must honor; the decoder takes explicit addresses, so a shifted
        // stream decodes identically.
        let end = span_end + placement.tsc_shift as u32;
        if exact {
            if end > 0x1_0000 {
                return Err(format!(
                    "tscrunch in-place needs {} bytes above the span end, but the stream \
                     would cross $FFFF",
                    placement.tsc_shift
                ));
            }
            let start = end
                .checked_sub(embedded_len)
                .ok_or_else(|| "packed data is larger than the output span".to_string())?;
            if start < 0x0200 {
                return Err("packed data reaches below $0200".into());
            }
            (start, end)
        } else {
            // Worst-case estimate must not falsely reject; clamp like forward.
            (
                end.saturating_sub(embedded_len).max(0x0200),
                end.min(0x1_0000),
            )
        }
    } else {
        match direction {
            Direction::Forward => {
                let start = 0x1_0000u32.checked_sub(embedded_len);
                if exact {
                    let start = start.ok_or("packed data is larger than the address space")?;
                    if start <= span_start {
                        return Err(format!(
                            "too big: packed data (${start:04X}-$FFFF) reaches down to the span \
                         start ${span_start:04X}"
                        ));
                    }
                    (start, 0x1_0000u32)
                } else {
                    // Clamp: at worst the packed stream reaches just above the
                    // span start; the pool below the span is unaffected either way.
                    (start.unwrap_or(0).max(span_start + 1), 0x1_0000u32)
                }
            }
            Direction::Backward => {
                let start = span_start
                    .checked_sub(placement.margin as u32)
                    .filter(|&s| s >= 0x0200)
                    .ok_or_else(|| {
                        format!(
                            "the span starts at ${span_start:04X} — no room for packed data {} \
                         bytes below the start (above $0200)",
                            placement.margin
                        )
                    })?;
                let end = start + embedded_len;
                if exact && end > span_end {
                    return Err(format!(
                        "too big: packed data (${start:04X}-${:04X}) reaches past the span end",
                        end - 1
                    ));
                }
                (start, end.min(span_end))
            }
        }
    };

    let span_b = Block {
        start: span_start,
        end: span_end,
    };
    let packed_b = Block {
        start: packed_start,
        end: packed_end,
    };
    let image_b = Block {
        start: 0x0801,
        end: image_end_estimate(base_staged, embedded_len).min(0x1_0000),
    };
    // Whether the payload move lands on the program image (so the copy code
    // must survive the overwrite). Default (auto mover): FOLD the moves into
    // the staged decoder blob — one blob, one install loop, no separate mover
    // placement. An explicit mover address keeps the classic relocated mover.
    let mover_needed = packed_b.overlaps(&image_b);
    let mover_folded = mover_needed && placement.mover.is_none();
    // Room for the folded copy loop(s) inside the staged blob (one payload
    // move; the biased/descending loop is <= 23 bytes, keep headroom).
    const MOVER_FOLD: u32 = 28;
    let staged = base_staged + if mover_folded { MOVER_FOLD } else { 0 };
    // Decoder/mover blobs are INSTALLED while the program image is still
    // intact, so their pool must avoid it. The scratch buffer is different:
    // it is first written during DECODE, after the payload move has emptied
    // the image — so its pool only avoids the output span and the packed
    // stream's decode-time position. (build_sfx always moves the payload out
    // of the image before decoding, which this relies on.)
    let pool = auto_pool(&[span_b, packed_b, image_b]);
    let scratch_pool = auto_pool(&[span_b, packed_b]);
    let mut used: Vec<Block> = Vec::new();

    // ---- decoder ----
    let decr_at = match placement.decruncher {
        Some(a) => {
            let b = Block {
                start: a as u32,
                end: a as u32 + staged,
            };
            if b.overlaps(&span_b) || b.overlaps(&packed_b) {
                return Err(format!(
                    "the decoder at ${a:04X} (+{staged} B) collides with the output span or \
                     packed data"
                ));
            }
            if a != 0x0100 && (a as u32) < 0x0200 {
                return Err(format!(
                    "the decoder address ${a:04X} is in the stack/zero page"
                ));
            }
            a as u32
        }
        None => {
            // The stack-page rule for small decoders: the full staged size
            // (decoder + wrapper + option epilogue) must fit $0100-$01DF,
            // otherwise relocate to a free slot.
            if staged <= STACK_PAGE_SLOT {
                0x0100
            } else {
                first_fit(&pool, &used, staged, false)
                    .ok_or_else(|| format!("no free space for the decoder ({staged} B)"))?
            }
        }
    };
    used.push(Block {
        start: decr_at,
        end: decr_at + staged,
    });

    // ---- scratch buffer ----
    let scratch = match spec.scratch.as_ref() {
        None => None,
        Some(sc) => {
            let len = sc.len as u32;
            let at = match placement.scratch {
                Some(a) => {
                    if sc.page_aligned && a & 0xFF != 0 {
                        return Err(format!(
                            "the buffer for {} must be page-aligned (${a:04X} is not)",
                            format.as_str()
                        ));
                    }
                    let b = Block {
                        start: a as u32,
                        end: a as u32 + len,
                    };
                    if b.overlaps(&span_b) || b.overlaps(&packed_b) {
                        return Err(format!(
                            "the buffer at ${a:04X} (+{len} B) collides with the output span or \
                             packed data"
                        ));
                    }
                    if used.iter().any(|u| b.overlaps(u)) {
                        return Err(format!("the buffer at ${a:04X} collides with the decoder"));
                    }
                    if (a as u32) < 0x0100 {
                        return Err("the buffer cannot be in zero page".into());
                    }
                    a as u32
                }
                None => first_fit(&scratch_pool, &used, len, sc.page_aligned).ok_or_else(|| {
                    format!(
                        "{} needs {} bytes of {}buffer, but there is no free space outside \
                         the output span",
                        format.as_str(),
                        len,
                        if sc.page_aligned { "page-aligned " } else { "" }
                    )
                })?,
            };
            used.push(Block {
                start: at,
                end: at + len,
            });
            Some((at as u16, sc.len))
        }
    };

    // ---- mover ----
    // Folded (the default): the moves already live inside the staged blob —
    // no separate placement. An explicit mover address keeps the classic
    // relocated mover with its collision checks.
    let mover_at = if mover_needed && !mover_folded {
        // One relocated copy loop + JMP assembles to <= ~55 bytes; 64 keeps
        // headroom and still fits the $02A7-$02FF slot.
        let mover_est = 64u32;
        let a = placement
            .mover
            .expect("mover_folded is false only with an explicit mover");
        let at = a as u32;
        let b = Block {
            start: at,
            end: at + mover_est,
        };
        if b.overlaps(&packed_b) {
            return Err(format!(
                "the mover at ${at:04X} collides with packed data — choose a \
                 different address"
            ));
        }
        if used.iter().any(|u| b.overlaps(u)) {
            return Err(format!(
                "the mover at ${at:04X} collides with the decoder or buffer"
            ));
        }
        if at == DEFAULT_MOVER as u32 && b.end > MOVER_SLOT_END {
            return Err(format!(
                "the mover does not fit in ${DEFAULT_MOVER:04X}-$02FF"
            ));
        }
        used.push(Block {
            start: at,
            end: at + mover_est,
        });
        Some(a)
    } else {
        None
    };

    Ok(Placed {
        staged,
        packed_start,
        decr_at: decr_at as u16,
        scratch,
        mover_at,
        mover_folded,
        variant: spec.variant,
    })
}

struct Placed {
    staged: u32,
    packed_start: u32,
    decr_at: u16,
    scratch: Option<(u16, u16)>,
    mover_at: Option<u16>,
    /// The payload move overwrites the program image and its copy code is
    /// folded into the staged decoder blob (no separate mover placement).
    mover_folded: bool,
    /// The decoder variant this placement chose (the stack-page-resident
    /// extra-small variant when it fits, otherwise the baseline).
    variant: Variant,
}

/// Compute the live plan for the UI without compressing (worst-case packed
/// size). A format whose plan is `unavailable` is hidden from the dropdown.
pub fn plan_preview(
    image: &MemoryImage,
    format: Format,
    placement: &Placement,
) -> Result<PlanPreview, String> {
    let (span_start, span_end) = image.span().ok_or("no .prg added")?;
    let span = (span_start, span_end);
    let candidates = match direction_candidates(span, format, placement) {
        Ok(c) => c.dirs,
        Err(e) => {
            return Ok(PlanPreview {
                direction: direction_for_span(span_end),
                staged_size: 0,
                decruncher_at: None,
                mover_at: placement.mover.unwrap_or(DEFAULT_MOVER),
                scratch: None,
                unavailable: Some(e),
            })
        }
    };

    // The first candidate is the direction the build will try first; its
    // worst-case placement is already validated by direction_candidates.
    let direction = candidates[0];
    let embedded_est = clen_worst_case(span_end - span_start);
    let p = place(span, embedded_est, format, direction, placement, false)
        .expect("candidate pre-validated");
    Ok(PlanPreview {
        direction,
        staged_size: p.staged,
        decruncher_at: Some(p.decr_at),
        mover_at: p
            .mover_at
            .unwrap_or_else(|| placement.mover.unwrap_or(DEFAULT_MOVER)),
        scratch: p.scratch,
        unavailable: None,
    })
}

// ---------------------------------------------------------------------------
// Building
// ---------------------------------------------------------------------------

#[derive(Debug)]
pub struct SfxResult {
    /// The finished self-extracting `.prg` (2-byte load address + image).
    pub prg: Vec<u8>,
    pub direction: Direction,
    /// Raw compressed stream length (before ABI prefixes / SFX wrapper).
    pub stream_len: usize,
    /// Where the packed stream sits when the decoder runs.
    pub packed_at: u16,
    pub decruncher_at: u16,
    /// Scratch buffer (address, length) for formats that need one.
    pub scratch: Option<(u16, u16)>,
    /// Explicit relocated-mover address, when one was used. `None` = the
    /// payload move runs inline from the main program, or (see
    /// [`SfxResult::mover_folded`]) folded into the staged decoder blob.
    pub mover_at: Option<u16>,
    /// The payload move overwrites the program image and its copy code was
    /// folded into the front of the staged decoder blob (the default; an
    /// explicit mover address keeps the classic separate mover).
    pub mover_folded: bool,
    pub span: (u32, u32),
    pub warnings: Vec<String>,
    /// The embedded decoder was trait-tailored to this stream (a smaller,
    /// per-crunch decoder body; Exomizer only). `false` = the standard body.
    pub decoder_tailored: bool,
    /// Bytes the whole SFX shrank by choosing the tailored decoder over the
    /// standard one (0 when not tailored).
    pub decoder_saved: usize,
    /// Size of the decoder body that was actually built, after any tailoring.
    /// Reflects the variant the placement chose, so callers must not re-derive
    /// it with `pick_routine`.
    pub decoder_bytes: u16,
    /// The raw compressed stream embedded in the SFX — the compressor's output,
    /// before the SFX wrapper and any ABI prefix. Exposed so a host can export
    /// just the packed data.
    pub stream: Vec<u8>,
    /// Whether the payload was moved/copied to its decode position. `false` =
    /// the "no move" in-place layout was used (the image sits clear of the
    /// output span, so the decoder reads the payload where it was loaded).
    pub payload_moved: bool,
}

fn overlaps(a0: u32, a1: u32, b0: u32, b1: u32) -> bool {
    a0 < b1 && b0 < a1
}

/// Exact in-place safety check for a PuCrunch stream in this builder's
/// layouts (see `lzan::pucrunch::container_max_gap`): the write head must
/// never reach an unread stream byte. Forward: container end-aligned at
/// `$10000`, output ascending from the span start. Backward: container at
/// `packed_start`, output descending from the span end.
fn pucrunch_inplace_safe(
    stream: &[u8],
    direction: Direction,
    span: (u32, u32),
    packed_start: u32,
) -> Result<(), String> {
    let (span_start, span_end) = span;
    let len = stream.len() as i64;
    let (gap, bound) = match direction {
        Direction::Forward => (
            lzan_c64::container_max_gap(stream)?,
            (0x1_0000 - len + 19) - span_start as i64,
        ),
        Direction::Backward => (
            lzan_c64::container_max_gap_backward(stream)?,
            (span_end as i64 - packed_start as i64) - (len - 19),
        ),
    };
    if gap >= bound {
        return Err(format!(
            "pucrunch {} in-place decode would overwrite {} unread stream byte(s) \
             (escape-expanded stretch); try the other direction or a larger margin",
            direction.as_str(),
            gap - bound + 1
        ));
    }
    Ok(())
}

/// Effective placement for formats whose compressed stream can be momentarily
/// LARGER than the output it has produced, so a fixed in-place margin is not
/// always enough — apultra (9-bit literals) and ByteBoozer2 (bit-oriented LZ).
/// Only the exact stream reveals how much room is needed, so this runs at build
/// time:
///
/// * **Backward** — raise `margin` to the stream's requirement. `place` then
///   rejects it if the larger margin no longer fits below the span start
///   (Auto falls back to forward; a forced direction errors cleanly).
/// * **Forward** — the top-aligned layout has no margin knob; the safety gap is
///   the free space above the span (`$10000 - span_end`). Reject when it cannot
///   cover the requirement rather than emit a self-corrupting SFX.
///
/// Formats with a self-sufficient fixed margin pass through unchanged.
fn inplace_effective_placement(
    format: Format,
    direction: Direction,
    stream: &[u8],
    span: (u32, u32),
    placement: &Placement,
) -> Result<Placement, String> {
    // Headroom for the decoder's partial-byte bit buffer / any off-by-one in the
    // gap estimate vs. the 6502 pointer arithmetic.
    const SLACK: u32 = 8;

    // PuCrunch's gap is normalized against its container (19-byte header), so its
    // required margin has a different formula than the raw max-gap formats. It
    // used to only REJECT the layout; instead, raise the backward margin to cover
    // the exact overwrite (BACKWARD safe iff `gap < out_len + margin - clen + 19`).
    if format == Format::PuCrunch {
        let out_len = (span.1 - span.0) as i64;
        let clen = stream.len() as i64;
        return match direction {
            Direction::Backward => {
                let gap = lzan_c64::container_max_gap_backward(stream)?;
                let need = (gap - out_len + clen - 18 + SLACK as i64).max(0);
                if (placement.margin as i64) < need {
                    Ok(Placement {
                        margin: need.min(0xFFFF) as u16,
                        ..*placement
                    })
                } else {
                    Ok(*placement)
                }
            }
            Direction::Forward => {
                let gap = lzan_c64::container_max_gap(stream)?;
                let bound = (0x1_0000i64 - clen + 19) - span.0 as i64;
                if gap >= bound {
                    Err(format!(
                        "pucrunch forward in-place would overwrite {} unread stream byte(s) \
                         (escape-expanded stretch); try backward",
                        gap - bound + 1
                    ))
                } else {
                    Ok(*placement)
                }
            }
        };
    }

    // TSCrunch backward is the format's native in-place mode (end-aligned
    // stream, decoded FORWARD). The encoder keeps the read head ahead of the
    // write head at token boundaries, but the 6502 token copies write up to
    // run-1 bytes ABOVE the write head (descending literal loop, RLE/LZ runs)
    // — on a stream whose gap gets tight mid-decode (long incompressible
    // stretches with occasional matches, e.g. already-packed payloads) that
    // overshoot clobbers unread stream bytes. Compute the exact per-stream
    // upward shift and place the stream that much above the span end.
    if format == Format::TsCrunch && direction == Direction::Backward {
        let shift = lzan_c64::tscrunch_inplace_shift(stream) as u32;
        if shift == 0 {
            return Ok(*placement); // reference end-aligned layout is safe
        }
        let need = shift + SLACK;
        let avail = 0x1_0000u32.saturating_sub(span.1);
        if avail < need {
            return Err(format!(
                "tscrunch in-place needs {need} bytes above ${:04X} (its token copies would \
                 overwrite unread stream bytes), but only {avail} are free below $FFFF",
                span.1.saturating_sub(1)
            ));
        }
        return Ok(Placement {
            tsc_shift: need as u16,
            ..*placement
        });
    }

    // (backward gap fn, forward gap fn) for the raw max-gap expanding formats.
    type GapFn = fn(&[u8]) -> usize;
    let (gap_backward, gap_forward): (GapFn, GapFn) = match format {
        Format::Aplib => (lzan_c64::aplib_gap_backward, lzan_c64::aplib_gap_forward),
        Format::ByteBoozer2 => (lzan_c64::bb2_gap_backward, lzan_c64::bb2_gap_forward),
        Format::Subsizer => (
            lzan_c64::subsizer_gap_backward,
            lzan_c64::subsizer_gap_forward,
        ),
        Format::Upkr => (lzan_c64::upkr_gap_backward, lzan_c64::upkr_gap_forward),
        Format::Zx02 => (lzan_c64::zx02_gap_backward, lzan_c64::zx02_gap_forward),
        Format::Zx0 => (lzan_c64::zx0_gap_backward, lzan_c64::zx0_gap_forward),
        Format::Lzsa1 => (lzan_c64::lzsa1_gap_backward, lzan_c64::lzsa1_gap_forward),
        Format::Lzsa2 => (lzan_c64::lzsa2_gap_backward, lzan_c64::lzsa2_gap_forward),
        Format::Exomizer => (lzan_c64::exo_gap_backward, lzan_c64::exo_gap_forward),
        Format::LzanMin => (
            lzan_c64::lzan_min_gap_backward,
            lzan_c64::lzan_min_gap_forward,
        ),
        Format::Bolt => (lzan_c64::bolt_gap_backward, lzan_c64::bolt_gap_forward),
        _ => {
            // No exact gap function (lzan-full's decoder depends on a stripped
            // mode byte; TSCrunch's forward decoder is not read-ahead-safe and
            // corrupts on overlap): be conservative. Forward in-place is safe
            // only when the packed stream cannot overlap the output
            // (clen <= top gap); otherwise fall back to backward (or, when even
            // backward cannot fit, the format is simply unavailable for this
            // span — the same as TSCrunch on a full-64K span).
            if direction == Direction::Forward {
                let clen = stream.len() as u32;
                let top_gap = 0x1_0000u32.saturating_sub(span.1);
                if clen > top_gap {
                    return Err(format!(
                        "{} forward in-place: packed data overlaps the output and its exact \
                         expansion is not modeled; try backward",
                        format.as_str()
                    ));
                }
            }
            return Ok(*placement);
        }
    };
    match direction {
        Direction::Backward => {
            let need = gap_backward(stream) as u32 + SLACK;
            if (placement.margin as u32) < need {
                Ok(Placement {
                    margin: need.min(0xFFFF) as u16,
                    ..*placement
                })
            } else {
                Ok(*placement)
            }
        }
        Direction::Forward => {
            let need = gap_forward(stream) as u32 + SLACK;
            let avail = 0x1_0000u32.saturating_sub(span.1);
            if avail < need {
                Err(format!(
                    "{} forward in-place needs a {need}-byte gap above ${:04X}, but only {avail} \
                     are free below $FFFF (an incompressible tail expands the stream); try backward",
                    format.as_str(),
                    span.1.saturating_sub(1)
                ))
            } else {
                Ok(*placement)
            }
        }
    }
}

/// Bytes the pre-JMP epilogue assembles to. MUST stay in sync with the
/// `post` fragment emitted in [`build_sfx`] — `place` reserves this much
/// extra room inside the staged decoder blob.
fn epilogue_len(p: &Placement) -> u32 {
    (if p.restore_basic_end.is_some() { 8 } else { 0 })   // LDA/STA $2D + LDA/STA $2E
        + (if p.bank_at_jmp != INIT_BANK { 4 } else { 0 }) // LDA #imm / STA $01
        + (if p.basic_clr { 3 } else { 0 })                // JSR $A659
        + (if p.cli_before_jmp { 1 } else { 0 }) // CLI
}

/// Whether the decompressed span overwrites the IRQ vector at `$0314/15`
/// with program data. A `CLI` before the final `JMP` is only safe if this is
/// false; otherwise a stray interrupt could jump through garbage.
pub fn span_covers_irq_vector(span: (u32, u32)) -> bool {
    overlaps(span.0, span.1, 0x0314, 0x0316)
}

/// Build the self-extracting `.prg`. `start_addr` is the JMP target after
/// decrunching.
pub fn build_sfx(
    image: &MemoryImage,
    format: Format,
    start_addr: u16,
    placement: &Placement,
) -> Result<SfxResult, String> {
    let (span_start, data) = image.span_buffer().ok_or("no .prg added")?;
    let span_end = span_start + data.len() as u32;
    if data.len() > 0xFFFF {
        return Err(format!(
            "The span ${span_start:04X}-${:04X} is {} bytes — more than 64 KB cannot be unpacked.",
            span_end - 1,
            data.len()
        ));
    }
    // Direction candidates (Auto falls back to the other direction when the
    // preferred one cannot host the format — same logic that drives the
    // dropdown, so what the GUI shows is what gets built). Compression runs
    // per candidate; the first whose exact placement fits wins.
    let plan = direction_candidates((span_start, span_end), format, placement)?;
    let mut chosen: Option<(Direction, Vec<u8>, Option<u8>, Placed)> = None;
    // Seed with the pruned directions' reasons so a pruned direction's
    // problem is reported alongside the tried candidates' failures.
    let mut errs: Vec<String> = plan.skipped.clone();
    for direction in plan.dirs {
        let variant = pick_routine(format, direction, placement.allow_illegal)
            .ok_or_else(|| {
                format!(
                    "no decoder for {} ({})",
                    format.as_str(),
                    direction.as_str()
                )
            })?
            .variant;
        let spec = Decruncher::with_variant(format, direction, variant)
            .map_err(|e| e.to_string())?
            .spec();
        // ---- compress (the slow part) -------------------------------------
        let out_end_u16 = (span_end & 0xFFFF) as u16; // $10000 wraps to 0 at the top
        let (stream, mode_byte) = compress_for(format, direction, &data, Some(out_end_u16))
            .map_err(|e| format!("Compression failed: {e}"))?;
        let embedded_len = stream.len() as u32
            + if spec.payload == PayloadAbi::DstPrefixed {
                2
            } else {
                0
            };
        // apultra (9-bit literals) and ByteBoozer2 can produce a stream that is
        // momentarily larger than the output decoded so far, so an incompressible
        // run decoded LATE makes the stream expand past the fixed in-place margin
        // — the decoder's write head then clobbers unread compressed bytes and
        // runs away (an infinite loop, not a clean failure). Size the layout to
        // the exact per-stream requirement: raise the BACKWARD margin, or reject
        // a FORWARD top gap that is too small (Auto then tries the other way).
        let eff = match inplace_effective_placement(
            format,
            direction,
            &stream,
            (span_start, span_end),
            placement,
        ) {
            Ok(p) => p,
            Err(e) => {
                errs.push(format!("{}: {e}", direction.as_str()));
                continue;
            }
        };
        // ---- placement (packed / decoder / scratch / mover) -----------------
        match place(
            (span_start, span_end),
            embedded_len,
            format,
            direction,
            &eff,
            true,
        ) {
            Ok(p) => {
                // PuCrunch's escape mechanism can locally EXPAND (escaped
                // literals cost 11+escBits bits for 8 output bits), so the
                // fixed layout margins do not automatically guarantee the
                // in-place invariant like they do for the other formats.
                // Check it exactly against the real stream; on failure this
                // candidate direction is skipped (Auto then tries the other).
                if format == Format::PuCrunch {
                    if let Err(e) = pucrunch_inplace_safe(
                        &stream,
                        direction,
                        (span_start, span_end),
                        p.packed_start,
                    ) {
                        errs.push(format!("{}: {e}", direction.as_str()));
                        continue;
                    }
                }
                chosen = Some((direction, stream, mode_byte, p));
                break;
            }
            Err(e) => errs.push(format!("{}: {e}", direction.as_str())),
        }
    }
    let (direction, stream, mode_byte, placed) = chosen.ok_or_else(|| {
        format!(
            "Placement failed: {}. Try a different format or adjust the addresses.",
            if errs.is_empty() {
                "unknown".to_string()
            } else {
                errs.join("; ")
            }
        )
    })?;
    let stream_len = stream.len();
    // Keep a copy of the raw compressed stream for the "export packed data"
    // feature, before it is moved into the builder below.
    let stream_for_export = stream.clone();
    // Per-crunch tailored decoder (Exomizer only): compose a body trimmed to
    // this exact stream's feature traits, from the stream BEFORE it is moved
    // into the builder. `None` when the stream admits no tailoring (all decoder
    // features used, or an unsupported format). The tailored body decodes the
    // identical stream bytes; correctness is pinned by lzan-c64's anchor /
    // assembly-matrix tests and the decoder_emulator_gate GOLF_TAILORED matrix.
    let tailored_src = if placement.tailoring != TailoringChoice::Standard {
        lzan_c64::tailored_body(format, direction, &stream)
    } else {
        None
    };
    let (packed_start, decr_at, mover_at) = (placed.packed_start, placed.decr_at, placed.mover_at);
    // The variant the placement chose (zp-stack when it fit the $0100 slot,
    // otherwise the baseline) — NOT a fresh pick_routine, which would undo the
    // placement's choice.
    let variant = placed.variant;
    // Static body size of the chosen variant, so the result can report the
    // decoder that actually ran.
    let variant_code_bytes = Decruncher::with_variant(format, direction, variant)
        .map(|d| d.spec().code_bytes)
        .unwrap_or(0);

    // ---- pre-JMP epilogue (shared by every layout) --------------------------
    // Keep its assembled size in sync with `epilogue_len`, which `place` uses to
    // reserve room for it inside the staged blob.
    let mut post = String::new();
    if let Some(end) = placement.restore_basic_end {
        // Restore VARTAB ($2D/$2E) — the end of the loaded BASIC program — so a
        // decompressed BASIC program can be RUN. Zero page is always RAM.
        post.push_str(&format!(
            "        LDA #${:02X}\n        STA $2D\n        LDA #${:02X}\n        STA $2E\n",
            end & 0xFF,
            end >> 8
        ));
    }
    if placement.bank_at_jmp != INIT_BANK {
        post.push_str(&format!(
            "        LDA #${:02X}\n        STA $01\n",
            placement.bank_at_jmp
        ));
    }
    if placement.basic_clr {
        post.push_str(&format!("        JSR ${BASIC_CLR:04X}\n"));
    }
    if placement.cli_before_jmp {
        post.push_str("        CLI\n");
    }

    let prg_from = |built: &lzan_c64::Built| -> Vec<u8> {
        let mut prg = Vec::with_capacity(built.bytes.len() + 2);
        prg.push((built.origin & 0xFF) as u8);
        prg.push((built.origin >> 8) as u8);
        prg.extend_from_slice(&built.bytes);
        prg
    };

    // Given a configured builder, assemble the standard body and (when a
    // tailored body applies) the tailored one, and keep the smaller per the
    // tailoring mode. Returns the chosen `Built` plus tailoring bookkeeping.
    let finalize = |b: Decruncher| -> Result<(lzan_c64::Built, bool, usize, Vec<String>), String> {
        let built_std = b
            .clone()
            .assemble()
            .map_err(|e| format!("Build/assembly failed: {e}"))?;
        let mut warnings: Vec<String> = built_std.warnings.iter().map(|w| w.msg.clone()).collect();
        match &tailored_src {
            Some(src) => match b.body_override(src.clone()).assemble() {
                Ok(bt) => {
                    let take = match placement.tailoring {
                        TailoringChoice::Tailored => true,
                        TailoringChoice::Auto => bt.bytes.len() < built_std.bytes.len(),
                        TailoringChoice::Standard => false,
                    };
                    if take {
                        let saved = built_std.bytes.len().saturating_sub(bt.bytes.len());
                        let w = bt.warnings.iter().map(|w| w.msg.clone()).collect();
                        Ok((bt, true, saved, w))
                    } else {
                        Ok((built_std, false, 0, warnings))
                    }
                }
                Err(_) => Ok((built_std, false, 0, warnings)),
            },
            None => {
                if placement.tailoring == TailoringChoice::Tailored {
                    warnings.push(
                        "Tailored decoder requested, but this stream uses every decoder feature — \
                         the standard body is already minimal."
                            .to_string(),
                    );
                }
                Ok((built_std, false, 0, warnings))
            }
        }
    };

    // The classic layout: payload embedded in the image, then moved (or, for
    // backward auto, its head window folded in) to its decode position; decoder
    // staged clear of the output span. TSCrunch keeps its end-aligned move; an
    // explicit mover address keeps the classic relocated mover.
    let configure_move = || -> Result<Decruncher, String> {
        let mut b = Decruncher::with_variant(format, direction, variant)
            .map_err(|e| e.to_string())?
            .basic_stub()
            .packed_inline(stream.clone())
            .output(span_start as u16)
            .output_len(data.len() as u16)
            .stage_decruncher_at(decr_at)
            .jmp_when_done(start_addr);
        let in_place_used =
            direction == Direction::Backward && format != Format::TsCrunch && placed.mover_folded;
        b = match direction {
            Direction::Forward => b.move_packed_to_top(0xFFFF),
            Direction::Backward if in_place_used => b.payload_in_place(packed_start as u16),
            Direction::Backward => b.move_packed_to(packed_start as u16),
        };
        if let Some((at, _)) = placed.scratch {
            b = b.scratch_address(at);
        }
        b = b.all_ram_with(INIT_BANK, None);
        if !post.is_empty() {
            b = b.custom_post(&post);
        }
        if placed.mover_folded && !in_place_used {
            b = b.fold_mover_into_stage();
        } else if let Some(at) = mover_at {
            b = b.mover_at(at);
        }
        if let Some(m) = mode_byte {
            b = b.mode_byte(m);
        }
        Ok(b)
    };

    // The "no move" in-place layout: keep the payload where it was loaded
    // (embedded in the image) and let the staged decoder read it there — no
    // move, no copy. Valid ONLY when the whole image (decoder + embedded
    // payload) sits clear of the output span by `placement.clearance`, and the
    // scratch buffer is clear of both. Then the decompressed output can never
    // overwrite the payload or the decoder mid-decode. Backward only (forward
    // uses the top-of-memory layout); TSCrunch keeps its special in-place move.
    let no_move: Option<Decruncher> = (|| {
        if direction != Direction::Backward || format == Format::TsCrunch {
            return None;
        }
        let mut nb = Decruncher::with_variant(format, direction, variant)
            .ok()?
            .basic_stub()
            .packed_inline(stream.clone())
            .output(span_start as u16)
            .output_len(data.len() as u16)
            .stage_decruncher_at(decr_at)
            .jmp_when_done(start_addr);
        if let Some((at, _)) = placed.scratch {
            nb = nb.scratch_address(at);
        }
        nb = nb.all_ram_with(INIT_BANK, None);
        if !post.is_empty() {
            nb = nb.custom_post(&post);
        }
        if let Some(m) = mode_byte {
            nb = nb.mode_byte(m);
        }
        // Assemble the STANDARD body to measure the exact image geometry (a
        // tailored body is only smaller, so its payload sits even lower — a
        // standard layout that clears the output clears it for tailored too).
        let built = nb.clone().assemble().ok()?;
        let origin = built.origin as u32;
        let img_end = origin + built.bytes.len() as u32;
        let clr = placement.clearance as u32;
        // The whole image (payload + decoder) must be clear of the output span.
        let image_clear = img_end + clr <= span_start || origin >= span_end + clr;
        if !image_clear {
            return None;
        }
        // Scratch, written during decode, must be clear of BOTH the image (the
        // payload it reads) and the output.
        if let Some((at, len)) = placed.scratch {
            let (s0, s1) = (at as u32, at as u32 + len as u32);
            let clear_img = s1 + clr <= origin || s0 >= img_end + clr;
            let clear_out = s1 + clr <= span_start || s0 >= span_end + clr;
            if !(clear_img && clear_out) {
                return None;
            }
        }
        Some(nb)
    })();

    let payload_moved = no_move.is_none();
    let b = match no_move {
        Some(nb) => nb,
        None => configure_move()?,
    };
    let (built, decoder_tailored, decoder_saved, warnings) = finalize(b)?;
    let prg = prg_from(&built);
    // Where the decoder reads the packed stream at run time: its final decode
    // position (moved layout) or its embedded position at the top of the image
    // (no-move layout).
    let packed_at = if payload_moved {
        packed_start as u16
    } else {
        (built.origin as u32 + built.bytes.len() as u32 - stream_len as u32) as u16
    };

    Ok(SfxResult {
        prg,
        direction,
        stream_len,
        packed_at,
        decruncher_at: decr_at,
        scratch: placed.scratch,
        mover_at: if payload_moved { mover_at } else { None },
        mover_folded: payload_moved && placed.mover_folded,
        span: (span_start, span_end),
        warnings,
        decoder_tailored,
        decoder_saved,
        decoder_bytes: variant_code_bytes.saturating_sub(decoder_saved as u16),
        stream: stream_for_export,
        payload_moved,
    })
}

// ---------------------------------------------------------------------------
// Small helpers
// ---------------------------------------------------------------------------

/// Detect a BASIC `SYS<addr>` autostart stub on the first line of a program
/// loading at $0801, and return the SYS target (e.g. exploding.prg -> $080D).
///
/// Tokenized BASIC lines are: 2-byte next-line pointer, 2-byte line number,
/// token/text body, `$00` terminator. The body is bounded by that terminator
/// so a match can only come from the first line, not later ones; within the
/// body the SYS token ($9E) is searched for directly rather than requiring it
/// to be the very first byte, so a leading REM or POKE statement before the
/// SYS call is tolerated.
pub fn detect_sys_start(load: u16, data: &[u8]) -> Option<u16> {
    if load != 0x0801 || data.len() < 5 {
        return None;
    }
    let body_end = 4 + data[4..].iter().position(|&b| b == 0x00)?;
    let body = &data[4..body_end];
    let sys = body.iter().position(|&b| b == 0x9E)?;
    let mut i = sys + 1;
    while i < body.len() && (body[i] == b' ' || body[i] == 0xA0 || body[i] == b'(') {
        i += 1;
    }
    let mut v: u32 = 0;
    let mut any = false;
    while i < body.len() && body[i].is_ascii_digit() {
        v = v * 10 + (body[i] - b'0') as u32;
        any = true;
        if v > 0xFFFF {
            return None;
        }
        i += 1;
    }
    (any && v > 0).then_some(v as u16)
}

/// Heuristic: does the region at `load` look like a runnable BASIC program (as
/// opposed to an ML program launched by a `SYS` stub, which [`detect_sys_start`]
/// catches first)? True when it loads at `$0801` and the first line's forward
/// link lands exactly on that line's `$00` terminator — a structure raw machine
/// code almost never has by accident. Used to offer "Force basic run".
pub fn looks_like_basic(load: u16, data: &[u8]) -> bool {
    if load != 0x0801 || data.len() < 5 {
        return false;
    }
    // [link_lo, link_hi, line_lo, line_hi, tokens…, $00][next line…]
    let link = u16::from_le_bytes([data[0], data[1]]);
    if link <= load {
        return false;
    }
    let link_off = (link - load) as usize;
    link_off >= 4 && link_off < data.len() && data[link_off - 1] == 0x00
}

/// Parse a user-entered address: `$1000`, `0x1000`, bare hex (`c000`) or decimal.
pub fn parse_addr(s: &str) -> Result<u16, String> {
    let t = s.trim();
    if t.is_empty() {
        return Err("empty address".into());
    }
    let value = if let Some(h) = t.strip_prefix('$') {
        u32::from_str_radix(h, 16)
    } else if let Some(h) = t.strip_prefix("0x").or_else(|| t.strip_prefix("0X")) {
        u32::from_str_radix(h, 16)
    } else if t.chars().all(|c| c.is_ascii_digit()) {
        t.parse::<u32>()
    } else {
        u32::from_str_radix(t, 16)
    }
    .map_err(|_| format!("invalid address: {t}"))?;
    if value > 0xFFFF {
        return Err(format!("address ${value:X} is above $FFFF"));
    }
    Ok(value as u16)
}

/// A cruncher dropdown entry (all 14 formats lzan-c64 can encode). The label is
/// the bare format name; the GUI appends the benchmarked packing/unpacking
/// speed and the live decoder size in parentheses.
pub struct Cruncher {
    pub format: Format,
    pub label: &'static str,
}

pub fn cruncher_list() -> Vec<Cruncher> {
    use Format::*;
    [
        (LzanFull, "LZAN full"),
        (LzanMin, "LZAN min"),
        (Exomizer, "Exomizer"),
        (Subsizer, "Subsizer"),
        (Shrinkler, "Shrinkler"),
        (Zx02, "ZX02"),
        (Zx0, "ZX0"),
        (Lzsa2, "LZSA2"),
        (Lzsa1, "LZSA1"),
        (Aplib, "aPLib"),
        (TsCrunch, "TSCrunch"),
        (ByteBoozer2, "ByteBoozer2"),
        (PuCrunch, "PuCrunch"),
        (Upkr, "upkr"),
        (Bolt, "BoltLZ"),
    ]
    .into_iter()
    .map(|(format, label)| Cruncher { format, label })
    .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use lzan_c64::Format::*;

    fn prg_bytes(load: u16, data: &[u8]) -> Vec<u8> {
        let mut v = vec![(load & 0xFF) as u8, (load >> 8) as u8];
        v.extend_from_slice(data);
        v
    }

    fn compressible(len: usize) -> Vec<u8> {
        (0..len)
            .map(|i| ((i / 64) as u8) ^ ((i % 7) as u8))
            .collect()
    }

    #[test]
    fn image_span_and_zero_gap() {
        let mut img = MemoryImage::default();
        img.add_prg("a.prg", &prg_bytes(0x0801, &[1, 2, 3]))
            .unwrap();
        img.add_prg("b.prg", &prg_bytes(0xC000, &[9, 8])).unwrap();
        assert_eq!(img.span(), Some((0x0801, 0xC002)));
        let (lo, buf) = img.span_buffer().unwrap();
        assert_eq!(lo, 0x0801);
        assert_eq!(buf.len(), 0xC002 - 0x0801);
        assert_eq!(&buf[..3], &[1, 2, 3]);
        assert!(
            buf[3..buf.len() - 2].iter().all(|&b| b == 0),
            "gap must be zeroes"
        );
        assert_eq!(&buf[buf.len() - 2..], &[9, 8]);
    }

    #[test]
    fn image_sorted_view_and_remove() {
        let mut img = MemoryImage::default();
        img.add_prg("high.prg", &prg_bytes(0xC000, &[1])).unwrap();
        img.add_prg("low.prg", &prg_bytes(0x0801, &[2])).unwrap();
        let order = img.sorted_indices();
        assert_eq!(img.regions()[order[0]].load, 0x0801);
        img.remove(1); // removes low.prg (insertion index 1)
        assert_eq!(img.span(), Some((0xC000, 0xC001)));
    }

    #[test]
    fn resize_region_trims_grows_and_shifts() {
        let mut img = MemoryImage::default();
        img.add_prg("a.prg", &prg_bytes(0x1000, &[1, 2, 3, 4, 5, 6, 7, 8]))
            .unwrap();
        // Shrink from the front (fix an imprecise load address): drops 1, 2.
        img.resize_region(0, 0x1002, 0x1008).unwrap();
        assert_eq!(img.regions()[0].start(), 0x1002);
        assert_eq!(img.regions()[0].data, vec![3, 4, 5, 6, 7, 8]);
        // Trim trailing bytes.
        img.resize_region(0, 0x1002, 0x1006).unwrap();
        assert_eq!(img.regions()[0].data, vec![3, 4, 5, 6]);
        // Grow right: the new area is zero-filled.
        img.resize_region(0, 0x1002, 0x100A).unwrap();
        assert_eq!(img.regions()[0].data, vec![3, 4, 5, 6, 0, 0, 0, 0]);
        // Grow left: prepends zeros.
        img.resize_region(0, 0x1000, 0x100A).unwrap();
        assert_eq!(img.regions()[0].start(), 0x1000);
        assert_eq!(img.regions()[0].data, vec![0, 0, 3, 4, 5, 6, 0, 0, 0, 0]);
        // A region may end at $FFFF (exclusive $10000); past that is an error.
        assert!(img.resize_region(0, 0xFFFE, 0x1_0000).is_ok());
        assert!(img.resize_region(0, 0x1000, 0x1000).is_err()); // empty
        assert!(img.resize_region(0, 0x1000, 0x1_0001).is_err()); // above $FFFF
    }

    #[test]
    fn apply_edits_writes_regions_and_creates_patches() {
        let mut img = MemoryImage::default();
        img.add_prg("a.prg", &prg_bytes(0x1000, &[1, 2, 3, 4]))
            .unwrap();
        img.add_prg("b.prg", &prg_bytes(0x1008, &[9, 9])).unwrap();
        let old = img.full_buffer();
        assert_eq!(&old[0x1000..0x100A], &[1, 2, 3, 4, 0, 0, 0, 0, 9, 9]);

        let mut new = old.clone();
        new[0x1001] = 0xAA; // inside a.prg
        new[0x1002] = 0xBB; // inside a.prg (same run)
        new[0x1005] = 0xCC; // gap between the regions -> new region
        new[0x2000] = 0xDD; // far outside everything -> new region
        let log = img.apply_edits(&old, &new);

        assert_eq!(img.regions()[0].data, vec![1, 0xAA, 0xBB, 4]);
        assert_eq!(img.regions().len(), 4);
        assert_eq!(img.regions()[2].start(), 0x1005);
        assert_eq!(img.regions()[2].data, vec![0xCC]);
        assert_eq!(img.regions()[3].start(), 0x2000);
        assert_eq!(img.full_buffer(), new, "edits must round-trip");
        assert_eq!(log.len(), 3);

        // A run crossing a region edge splits: in-region part edits the
        // region, the rest becomes a patch region.
        let old = img.full_buffer();
        let mut new = old.clone();
        new[0x1002..0x1007].fill(0x77);
        img.apply_edits(&old, &new);
        assert_eq!(
            img.full_buffer(),
            new,
            "boundary-crossing edits must round-trip"
        );
        assert_eq!(img.regions()[0].data, vec![1, 0xAA, 0x77, 0x77]);

        // Overlapping regions: the byte goes to the LAST painted (visible) one.
        let mut img = MemoryImage::default();
        img.add_prg("under.prg", &prg_bytes(0x1000, &[1, 1, 1, 1]))
            .unwrap();
        img.add_prg("over.prg", &prg_bytes(0x1001, &[2, 2]))
            .unwrap();
        let old = img.full_buffer();
        let mut new = old.clone();
        new[0x1001] = 0xEE;
        img.apply_edits(&old, &new);
        assert_eq!(
            img.regions()[0].data,
            vec![1, 1, 1, 1],
            "hidden region untouched"
        );
        assert_eq!(img.regions()[1].data, vec![0xEE, 2]);
        assert_eq!(img.full_buffer(), new);
    }

    /// The pucrunch zp-stack variant is selected automatically when the
    /// decoder address is auto and the staged blob fits the $0100 slot; a
    /// manual decoder address falls back to the standard baseline.
    #[test]
    fn zp_stack_variant_selected_when_it_fits() {
        let mut img = MemoryImage::default();
        img.add_prg("a.prg", &prg_bytes(0x0801, &compressible(4096)))
            .unwrap();
        let auto = Placement::default();
        let r = build_sfx(&img, PuCrunch, 0x0801, &auto).unwrap();
        assert_eq!(
            r.decruncher_at, 0x0100,
            "auto must stage the zp-stack variant in page 1"
        );

        let manual = Placement {
            decruncher: Some(0x0400),
            ..Placement::default()
        };
        let m = build_sfx(&img, PuCrunch, 0x0801, &manual).unwrap();
        assert_eq!(m.decruncher_at, 0x0400);
        assert!(
            m.prg.len() > r.prg.len(),
            "manual placement uses the (bigger) standard body: {} vs {} B",
            m.prg.len(),
            r.prg.len()
        );
        // Same stream either way — the variant only swaps the decoder.
        assert_eq!(m.stream_len, r.stream_len);
    }

    #[test]
    fn direction_rule_fff0() {
        assert_eq!(direction_for_span(0xFFF1), Direction::Forward); // last byte $FFF0
        assert_eq!(direction_for_span(0xFFF2), Direction::Backward);
        assert_eq!(direction_for_span(0x1_0000), Direction::Backward);
    }

    #[test]
    fn detect_sys_start_parses_exploding_style_stub() {
        // 0 SYS2061 stub: next-line ptr, line 0, $9E, "2061", EOL, end.
        let stub = [
            0x0B, 0x08, 0x0A, 0x00, 0x9E, 0x32, 0x30, 0x36, 0x31, 0x00, 0x00, 0x00,
        ];
        assert_eq!(detect_sys_start(0x0801, &stub), Some(0x080D));
        assert_eq!(detect_sys_start(0x1000, &stub), None);
        assert_eq!(detect_sys_start(0x0801, &[0; 12]), None);
    }

    #[test]
    fn detect_sys_start_tolerates_a_statement_before_sys() {
        // 1 POKE1,1:SYS2061 — SYS is not the first token on the line, and a
        // second line with its own $9E must not be picked up instead.
        let mut stub = vec![0x00, 0x00, 0x01, 0x00]; // next-line ptr (patched below), line 1
        stub.extend_from_slice(b"POKE1,1:");
        stub.push(0x9E); // SYS
        stub.extend_from_slice(b"2061");
        stub.push(0x00); // end of line 1
        let line1_len = stub.len() as u16;
        stub[0] = (0x0801 + line1_len) as u8;
        stub[1] = ((0x0801 + line1_len) >> 8) as u8;
        // A second line whose own SYS token must be ignored.
        stub.extend_from_slice(&[0x00, 0x00, 0x02, 0x00, 0x9E, b'1', 0x00]);
        assert_eq!(detect_sys_start(0x0801, &stub), Some(0x080D));
    }

    #[test]
    fn looks_like_basic_recognizes_line_structure() {
        // 10 <PRINT>: link → $0807 (the end marker), line 10, token $99, $00,
        // then the $0000 end-of-program. The forward link lands on the line's
        // $00 terminator, so this is BASIC.
        let prog = [0x07, 0x08, 0x0A, 0x00, 0x99, 0x00, 0x00, 0x00];
        assert!(looks_like_basic(0x0801, &prog));
        // Same bytes anywhere but $0801 are not a BASIC start segment.
        assert!(!looks_like_basic(0xC000, &prog));
        // Raw machine code (SEI; LDA #$37; STA $01; JMP $1000): the "link" points
        // far outside the block.
        let ml = [0x78, 0xA9, 0x37, 0x85, 0x01, 0x4C, 0x00, 0x10];
        assert!(!looks_like_basic(0x0801, &ml));
        // Too short to be a line.
        assert!(!looks_like_basic(0x0801, &[0x00, 0x00]));
    }

    #[test]
    fn parse_addr_accepts_the_usual_forms() {
        assert_eq!(parse_addr("$C000").unwrap(), 0xC000);
        assert_eq!(parse_addr("0x1000").unwrap(), 0x1000);
        assert_eq!(parse_addr("c000").unwrap(), 0xC000);
        assert_eq!(parse_addr("2049").unwrap(), 2049);
        assert!(parse_addr("$1FFFF").is_err());
        assert!(parse_addr("").is_err());
    }

    #[test]
    fn forward_layout_builds_for_all_formats() {
        let mut img = MemoryImage::default();
        img.add_prg("a.prg", &prg_bytes(0x0801, &compressible(4096)))
            .unwrap();
        for c in &cruncher_list() {
            let r = build_sfx(&img, c.format, 0x0801, &Placement::default())
                .unwrap_or_else(|e| panic!("{}: {e}", c.label));
            assert_eq!(
                &r.prg[..2],
                &[0x01, 0x08],
                "{}: not $0801 autostart",
                c.label
            );
            assert_eq!(r.direction, Direction::Forward, "{}", c.label);
        }
    }

    #[test]
    fn backward_layout_builds_for_full_range_span() {
        let mut img = MemoryImage::default();
        let len = 0x1_0000 - 0x0801;
        img.add_prg("full.prg", &prg_bytes(0x0801, &compressible(len)))
            .unwrap();
        let r = build_sfx(&img, LzanFull, 0x0801, &Placement::default()).unwrap();
        assert_eq!(r.direction, Direction::Backward);
        assert_eq!(r.packed_at, 0x0801 - DEFAULT_MARGIN);
        // The move lands on the program image; the copy code survives by
        // being folded into the staged decoder blob (the default).
        assert!(
            r.mover_folded,
            "full-range move must survive the image overwrite"
        );
        assert!(r.mover_at.is_none(), "no separate mover when folded");
        // An explicit mover address keeps the classic relocated mover.
        let manual = Placement {
            mover: Some(DEFAULT_MOVER),
            ..Placement::default()
        };
        let r = build_sfx(&img, LzanFull, 0x0801, &manual).unwrap();
        assert!(!r.mover_folded);
        assert_eq!(r.mover_at, Some(DEFAULT_MOVER), "explicit mover honored");
        // These formats support full-range backward decoding.
        for fmt in [Upkr, LzanMin] {
            let r = build_sfx(&img, fmt, 0x0801, &Placement::default())
                .unwrap_or_else(|e| panic!("{fmt:?}: {e}"));
            assert_eq!(r.direction, Direction::Backward, "{fmt:?}");
        }
    }

    /// Exploding-shaped image (58 KB @ $0801): exomizer's 156-byte table fits
    /// the $0334 tape buffer; upkr's page-aligned probs coexist with its
    /// decoder in the $0400 zone; shrinkler's 1536-byte buffer has no room and
    /// the format is reported unavailable (hidden from the dropdown).
    #[test]
    fn scratch_placement_exploding_shape() {
        let mut img = MemoryImage::default();
        img.add_prg("big.prg", &prg_bytes(0x0801, &compressible(58035)))
            .unwrap();
        let p = Placement::default();

        let exo = plan_preview(&img, Exomizer, &p).unwrap();
        assert!(exo.unavailable.is_none(), "{:?}", exo.unavailable);
        assert_eq!(exo.scratch, Some((0x0334, 156)));

        let upkr = plan_preview(&img, Upkr, &p).unwrap();
        assert!(upkr.unavailable.is_none(), "{:?}", upkr.unavailable);
        let (at, len) = upkr.scratch.unwrap();
        assert_eq!(at & 0xFF, 0, "upkr probs must be page-aligned");
        assert_eq!(len, 319);
        // Must not overlap the decoder.
        let d = upkr.decruncher_at.unwrap() as u32;
        assert!(
            at as u32 + 319 <= d || d + upkr.staged_size <= at as u32,
            "scratch ${at:04X} overlaps decoder ${d:04X}"
        );

        // Shrinkler cannot fit forward on a full low span; Auto falls back to
        // backward (see auto_falls_back_to_backward_for_shrinkler).
        let shr = plan_preview(&img, Shrinkler, &p).unwrap();
        assert_eq!(shr.direction, Direction::Backward);
        assert!(shr.unavailable.is_none());
    }

    /// Big low span: forward has no room for shrinkler's 1536-byte buffer
    /// (everything above the span is packed data), but the BACKWARD layout
    /// frees the area above the span end (the packed stream sits at the
    /// bottom, and the scratch pool ignores the program image, which is dead
    /// once the payload has been moved). AUTO must fall back to backward and
    /// keep shrinkler available; forced forward stays rejected.
    #[test]
    fn auto_falls_back_to_backward_for_shrinkler() {
        let mut img = MemoryImage::default();
        img.add_prg("big.prg", &prg_bytes(0x0801, &compressible(58035)))
            .unwrap();
        let span_end = 0x0801 + 58035u32;

        // Auto: available, with the fallback direction chosen automatically.
        let auto = plan_preview(&img, Shrinkler, &Placement::default()).unwrap();
        assert!(auto.unavailable.is_none(), "{:?}", auto.unavailable);
        assert_eq!(
            auto.direction,
            Direction::Backward,
            "auto must pick the only feasible direction"
        );
        let (at, len) = auto.scratch.unwrap();
        assert_eq!(len, 1536);
        assert!(
            (at as u32) >= span_end,
            "scratch ${at:04X} should sit above the span end ${span_end:04X}"
        );

        // Forced forward: honestly rejected.
        let fwd = Placement {
            direction: DirectionChoice::Forward,
            ..Placement::default()
        };
        let p = plan_preview(&img, Shrinkler, &fwd).unwrap();
        assert!(p.unavailable.is_some(), "forced forward must stay hidden");

        // Forced backward: same as auto's fallback.
        let back = Placement {
            direction: DirectionChoice::Backward,
            ..Placement::default()
        };
        let p = plan_preview(&img, Shrinkler, &back).unwrap();
        assert!(p.unavailable.is_none(), "{:?}", p.unavailable);

        // The real build agrees with the preview.
        let r = build_sfx(&img, Shrinkler, 0x0801, &Placement::default()).unwrap();
        assert_eq!(r.direction, Direction::Backward);
        assert_eq!(r.scratch.unwrap().1, 1536);
    }

    /// Small span: the gap between span end and the packed data at top is
    /// large, so even shrinkler's 1536-byte page-aligned buffer finds a home
    /// and the format stays available.
    #[test]
    fn scratch_placement_small_span_allows_shrinkler() {
        let mut img = MemoryImage::default();
        img.add_prg("small.prg", &prg_bytes(0x0801, &compressible(0x3000)))
            .unwrap();
        let p = plan_preview(&img, Shrinkler, &Placement::default()).unwrap();
        assert!(p.unavailable.is_none(), "{:?}", p.unavailable);
        let (at, len) = p.scratch.unwrap();
        assert_eq!(at & 0xFF, 0);
        assert_eq!(len, 1536);
    }

    /// $0200-$02A6 is reserved for automatic placement but usable manually.
    #[test]
    fn manual_scratch_override_allows_system_area() {
        let mut img = MemoryImage::default();
        img.add_prg("a.prg", &prg_bytes(0x0801, &compressible(4096)))
            .unwrap();
        //

        // Auto never lands in $0200-$02A6 (pool starts at $0334).
        let auto = plan_preview(&img, Exomizer, &Placement::default()).unwrap();
        assert!(auto.scratch.unwrap().0 >= 0x0334);
        // A manual override below $02A7 is accepted (exomizer table is not
        // page-aligned, 156 B fits $0200-$029C).
        let manual = Placement {
            scratch: Some(0x0200),
            ..Placement::default()
        };
        let p = plan_preview(&img, Exomizer, &manual).unwrap();
        assert!(p.unavailable.is_none(), "{:?}", p.unavailable);
        assert_eq!(p.scratch, Some((0x0200, 156)));
    }

    /// Manual direction override: forced backward on a mid-range span works;
    /// forced forward on a >$FFF0 span is rejected; forced backward on a span
    /// starting at $0200 (the user's example) is rejected.
    #[test]
    fn direction_override_feasibility() {
        let fwd_only = Placement {
            direction: DirectionChoice::Forward,
            ..Placement::default()
        };
        let back_only = Placement {
            direction: DirectionChoice::Backward,
            ..Placement::default()
        };

        // Mid-range span: both directions feasible.
        let mut img = MemoryImage::default();
        img.add_prg("a.prg", &prg_bytes(0x0801, &compressible(8000)))
            .unwrap();
        let r = build_sfx(&img, LzanFull, 0x0801, &back_only).unwrap();
        assert_eq!(r.direction, Direction::Backward);
        assert_eq!(r.packed_at, 0x0801 - DEFAULT_MARGIN);
        let r = build_sfx(&img, LzanFull, 0x0801, &fwd_only).unwrap();
        assert_eq!(r.direction, Direction::Forward);

        // Span ending at $FFFF: forced forward impossible.
        let mut img = MemoryImage::default();
        img.add_prg("f.prg", &prg_bytes(0xF000, &compressible(0x1000)))
            .unwrap();
        let e = build_sfx(&img, LzanFull, 0xF000, &fwd_only).unwrap_err();
        assert!(e.contains("forward is impossible"), "{e}");

        // Span starting at $0200: forced backward impossible.
        let mut img = MemoryImage::default();
        img.add_prg("low.prg", &prg_bytes(0x0200, &compressible(0x1000)))
            .unwrap();
        let e = build_sfx(&img, LzanFull, 0x0200, &back_only).unwrap_err();
        assert!(e.contains("backward is impossible"), "{e}");
        // The preview reports the same reason (drives the GUI).
        let p = plan_preview(&img, LzanFull, &back_only).unwrap();
        assert!(p.unavailable.unwrap().contains("backward is impossible"));

        // Forced backward decoding is available for these formats.
        let mut img = MemoryImage::default();
        img.add_prg("a.prg", &prg_bytes(0x0801, &compressible(4000)))
            .unwrap();
        for fmt in [Upkr, LzanMin] {
            let r =
                build_sfx(&img, fmt, 0x0801, &back_only).unwrap_or_else(|e| panic!("{fmt:?}: {e}"));
            assert_eq!(r.direction, Direction::Backward, "{fmt:?}");
        }
    }

    #[test]
    fn build_sfx_places_scratch_for_exomizer() {
        let mut img = MemoryImage::default();
        img.add_prg("a.prg", &prg_bytes(0x0801, &compressible(4096)))
            .unwrap();
        let r = build_sfx(&img, Exomizer, 0x0801, &Placement::default()).unwrap();
        assert_eq!(r.scratch, Some((0x0334, 156)));
    }

    #[test]
    fn multi_prg_two_regions_forward() {
        let mut img = MemoryImage::default();
        img.add_prg("main.prg", &prg_bytes(0x0801, &compressible(8000)))
            .unwrap();
        img.add_prg("data.prg", &prg_bytes(0xC000, &compressible(0x1000)))
            .unwrap();
        let r = build_sfx(&img, Lzsa2, 0x0801, &Placement::default()).unwrap();
        assert_eq!(r.direction, Direction::Forward);
        assert_eq!(r.span, (0x0801, 0xD000));
    }

    /// Default Placement leaves $01 and interrupts exactly as init left them
    /// (INIT_BANK, SEI), so no restore code is emitted at all. Overriding
    /// either adds exactly its own bytes: LDA #imm + STA zp (2+2) for a
    /// non-default bank value, CLI (1) for re-enabling interrupts.
    #[test]
    fn bank_and_cli_before_jmp_add_only_the_requested_bytes() {
        let mut img = MemoryImage::default();
        img.add_prg("a.prg", &prg_bytes(0x0801, &compressible(4096)))
            .unwrap();

        let base = build_sfx(&img, LzanFull, 0x0801, &Placement::default()).unwrap();

        let cli_only = Placement {
            cli_before_jmp: true,
            ..Placement::default()
        };
        let r = build_sfx(&img, LzanFull, 0x0801, &cli_only).unwrap();
        assert_eq!(
            r.prg.len(),
            base.prg.len() + 1,
            "CLI alone must add exactly 1 byte"
        );

        let bank_only = Placement {
            bank_at_jmp: 0x37,
            ..Placement::default()
        };
        let r = build_sfx(&img, LzanFull, 0x0801, &bank_only).unwrap();
        assert_eq!(
            r.prg.len(),
            base.prg.len() + 4,
            "LDA #imm + STA zp must add exactly 4 bytes"
        );

        let both = Placement {
            bank_at_jmp: 0x37,
            cli_before_jmp: true,
            ..Placement::default()
        };
        let r = build_sfx(&img, LzanFull, 0x0801, &both).unwrap();
        assert_eq!(
            r.prg.len(),
            base.prg.len() + 5,
            "bank + CLI must add exactly 5 bytes"
        );

        // A bank value equal to INIT_BANK is a no-op even if explicitly set.
        let explicit_default = Placement {
            bank_at_jmp: INIT_BANK,
            ..Placement::default()
        };
        let r = build_sfx(&img, LzanFull, 0x0801, &explicit_default).unwrap();
        assert_eq!(r.prg.len(), base.prg.len());
    }

    #[test]
    fn span_covers_irq_vector_detects_overlap() {
        assert!(!span_covers_irq_vector((0x0200, 0x0314))); // span ends right before the vector
        assert!(span_covers_irq_vector((0x0200, 0x0315))); // span includes the low byte ($0314)
        assert!(span_covers_irq_vector((0x0200, 0x0400))); // span covers the whole vector
        assert!(!span_covers_irq_vector((0x0316, 0x0400))); // span starts right after the vector
    }

    /// Restoring $2D/$2E emits exactly LDA #imm; STA $2D; LDA #imm; STA $2E —
    /// four two-byte instructions = 8 bytes — before the final JMP.
    #[test]
    fn restore_basic_end_adds_eight_bytes() {
        let mut img = MemoryImage::default();
        img.add_prg("a.prg", &prg_bytes(0x0801, &compressible(4096)))
            .unwrap();
        let base = build_sfx(&img, LzanFull, 0x0801, &Placement::default()).unwrap();
        let p = Placement {
            restore_basic_end: Some(0x1234),
            ..Placement::default()
        };
        let r = build_sfx(&img, LzanFull, 0x0801, &p).unwrap();
        assert_eq!(r.prg.len(), base.prg.len() + 8);
    }

    /// The RUN epilogue starts a decompressed BASIC program: restore VARTAB
    /// (8 B) + bank in ROM (4 B) + `JSR $A659` CLR (3 B) + `CLI` (1 B) = 16 B
    /// before the final `JMP $A7AE` (interpreter loop), whose target bytes
    /// are emitted verbatim.
    #[test]
    fn run_epilogue_emits_clr_and_interpreter_jump() {
        let mut img = MemoryImage::default();
        img.add_prg("a.prg", &prg_bytes(0x0801, &compressible(4096)))
            .unwrap();
        let base = build_sfx(&img, LzanFull, 0x0801, &Placement::default()).unwrap();
        let p = Placement {
            restore_basic_end: Some(0x1801),
            bank_at_jmp: RUN_BASIC_BANK,
            cli_before_jmp: true,
            basic_clr: true,
            ..Placement::default()
        };
        let r = build_sfx(&img, LzanFull, RUN_BASIC_LOOP, &p).unwrap();
        assert_eq!(r.prg.len(), base.prg.len() + 16);
        let has = |n: &[u8]| r.prg.windows(n.len()).any(|w| w == n);
        assert!(has(&[0x20, 0x59, 0xA6]), "JSR $A659 (CLR)");
        assert!(has(&[0x4C, 0xAE, 0xA7]), "JMP $A7AE (interpreter loop)");
    }

    /// The staged decoder reserves the RUN epilogue. A non-page-aligned scratch
    /// table placed directly above the decoder must not overlap that reservation.
    #[test]
    fn run_epilogue_reserved_scratch_clears_decoder() {
        let mut img = MemoryImage::default();
        img.add_prg("data.prg", &prg_bytes(0x0340, &compressible(16)))
            .unwrap();
        img.add_prg("basic.prg", &prg_bytes(0x0801, &compressible(4096)))
            .unwrap();
        let end = 0x0801 + 4096u16;
        let run = Placement {
            restore_basic_end: Some(end),
            bank_at_jmp: RUN_BASIC_BANK,
            cli_before_jmp: true,
            basic_clr: true,
            ..Placement::default()
        };
        let plan = plan_preview(&img, Exomizer, &run).unwrap();
        assert!(plan.unavailable.is_none(), "{:?}", plan.unavailable);
        if let (Some(d), Some((s, slen))) = (plan.decruncher_at, plan.scratch) {
            let (d, staged, s, slen) = (d as u32, plan.staged_size, s as u32, slen as u32);
            assert!(
                s + slen <= d || d + staged <= s,
                "scratch ${s:04X}(+{slen}) overlaps reserved decoder ${d:04X}(+{staged})"
            );
        }
        // The real build agrees and succeeds.
        build_sfx(&img, Exomizer, RUN_BASIC_LOOP, &run).unwrap();
    }

    /// `epilogue_len` must equal the exact byte growth the epilogue adds to
    /// the output. Everything else being equal, the only difference from the
    /// default build is the epilogue, so the size delta IS its length — and
    /// `place` reserves `STAGE_WRAPPER + epilogue_len` inside the staged blob,
    /// so as long as this holds the reservation always covers the real blob.
    #[test]
    fn epilogue_len_matches_emitted_bytes() {
        let mut img = MemoryImage::default();
        img.add_prg("a.prg", &prg_bytes(0x0801, &compressible(4096)))
            .unwrap();
        let base = build_sfx(&img, LzanFull, 0x0801, &Placement::default()).unwrap();
        let cases = [
            Placement {
                cli_before_jmp: true,
                ..Placement::default()
            },
            Placement {
                bank_at_jmp: 0x37,
                ..Placement::default()
            },
            Placement {
                restore_basic_end: Some(0x1801),
                ..Placement::default()
            },
            Placement {
                basic_clr: true,
                bank_at_jmp: 0x37,
                ..Placement::default()
            },
            Placement {
                restore_basic_end: Some(0x1801),
                bank_at_jmp: RUN_BASIC_BANK,
                basic_clr: true,
                cli_before_jmp: true,
                ..Placement::default()
            },
        ];
        for p in cases {
            let r = build_sfx(&img, LzanFull, 0x0801, &p).unwrap();
            assert_eq!(
                r.prg.len(),
                base.prg.len() + epilogue_len(&p) as usize,
                "epilogue_len disagrees with emitted bytes for {p:?}",
            );
        }
    }

    /// The staged-size estimate includes the option epilogue, and the auto
    /// stack-page rule never lands an oversized blob at $0100 — even with the
    /// full RUN epilogue added.
    #[test]
    fn staged_size_reflects_options_and_never_overflows_stack_page() {
        let mut img = MemoryImage::default();
        img.add_prg("a.prg", &prg_bytes(0x0801, &compressible(4096)))
            .unwrap();
        let run = Placement {
            restore_basic_end: Some(0x1801),
            bank_at_jmp: RUN_BASIC_BANK,
            basic_clr: true,
            cli_before_jmp: true,
            ..Placement::default()
        };
        // Options grow the staged reservation by exactly their epilogue length.
        let d = plan_preview(&img, LzanFull, &Placement::default()).unwrap();
        let r = plan_preview(&img, LzanFull, &run).unwrap();
        assert_eq!(r.staged_size, d.staged_size + epilogue_len(&run));
        // For every format, an auto decoder at $0100 fits the slot in both
        // the plain and full-RUN placements.
        for c in cruncher_list() {
            for p in [Placement::default(), run] {
                if let Ok(plan) = plan_preview(&img, c.format, &p) {
                    if plan.unavailable.is_none() && plan.decruncher_at == Some(0x0100) {
                        assert!(
                            plan.staged_size <= STACK_PAGE_SLOT,
                            "{}: $0100 decoder staged {} > slot {}",
                            c.label,
                            plan.staged_size,
                            STACK_PAGE_SLOT
                        );
                    }
                }
            }
        }
    }

    /// A manual decruncher override to a safe address is honored even for a
    /// large decoder+epilogue that auto would keep off $0100; a manual $0100
    /// override that would overflow the stack page is caught by the library's
    /// own stack-headroom check (a clear error, never a silent bad file).
    #[test]
    fn manual_decoder_override_honored_or_caught() {
        let mut img = MemoryImage::default();
        img.add_prg("a.prg", &prg_bytes(0x0801, &compressible(4096)))
            .unwrap();
        let opts = Placement {
            restore_basic_end: Some(0x1801),
            bank_at_jmp: RUN_BASIC_BANK,
            basic_clr: true,
            cli_before_jmp: true,
            ..Placement::default()
        };

        // Safe override: LzanFull's ~475-byte decoder relocated to free RAM.
        let safe = Placement {
            decruncher: Some(0xC000),
            ..opts
        };
        let r = build_sfx(&img, LzanFull, RUN_BASIC_LOOP, &safe).unwrap();
        assert_eq!(r.decruncher_at, 0xC000, "safe override honored");

        // Oversized $0100 override: the preview honors it (staged > slot), but
        // the build's stack-headroom check rejects it with a clear error.
        let forced = Placement {
            decruncher: Some(0x0100),
            ..opts
        };
        let plan = plan_preview(&img, LzanFull, &forced).unwrap();
        assert_eq!(plan.decruncher_at, Some(0x0100));
        assert!(plan.staged_size > STACK_PAGE_SLOT);
        let e = build_sfx(&img, LzanFull, RUN_BASIC_LOOP, &forced).unwrap_err();
        assert!(e.contains("headroom") || e.contains("stack"), "{e}");
    }

    /// Legal-only mode (`allow_illegal = false`) builds a valid $0801 autostart
    /// program for the formats whose baseline decoder uses illegal opcodes,
    /// and selects a legal decoder that is never smaller than the baseline.
    /// (The exact growth is not pinned: the standard and legal bodies are
    /// golfed independently, so the delta legitimately drifts.)
    #[test]
    fn legal_only_mode_selects_legal_decoder() {
        let mut img = MemoryImage::default();
        img.add_prg("a.prg", &prg_bytes(0x0801, &compressible(4096)))
            .unwrap();
        let illegal = Placement::default(); // allow_illegal = true
        let legal = Placement {
            allow_illegal: false,
            ..Placement::default()
        };

        for fmt in [Lzsa1, Lzsa2, TsCrunch, Upkr, PuCrunch] {
            let r = build_sfx(&img, fmt, 0x0801, &legal)
                .unwrap_or_else(|e| panic!("{fmt:?} legal build: {e}"));
            assert_eq!(&r.prg[..2], &[0x01, 0x08], "{fmt:?}: not $0801 autostart");

            let p_ill = plan_preview(&img, fmt, &illegal).unwrap();
            let p_leg = plan_preview(&img, fmt, &legal).unwrap();
            assert!(
                p_leg.staged_size >= p_ill.staged_size,
                "{fmt:?}: legal decoder ({} B staged) must not be smaller than the illegal \
                 baseline ({} B staged) — otherwise the baseline should BE the legal body",
                p_leg.staged_size,
                p_ill.staged_size
            );
        }

        // A format that is already legal (zx02) is byte-identical in both modes.
        let a = build_sfx(&img, Zx02, 0x0801, &illegal).unwrap();
        let b = build_sfx(&img, Zx02, 0x0801, &legal).unwrap();
        assert_eq!(a.prg, b.prg, "already-legal format must not change");
    }
}
