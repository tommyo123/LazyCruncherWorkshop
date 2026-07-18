//! LazyCruncher Workshop — egui GUI that packs one or more C64 `.prg` files
//! into a single self-extracting, crunched `.prg`.

#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

use std::panic::{catch_unwind, AssertUnwindSafe};
use std::path::PathBuf;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::mpsc::{Receiver, Sender, TryRecvError};
use std::sync::Arc;
use std::time::Instant;

use eframe::egui;
use lazy_cruncher_workshop::config;
use lazy_cruncher_workshop::import::{import_crunched_with, probe_deeper_layer, ImportedPart};
use lazy_cruncher_workshop::ranks;
use lazy_cruncher_workshop::sfx::{
    build_sfx, cruncher_list, detect_sys_start, looks_like_basic, parse_addr, plan_preview,
    span_covers_irq_vector, Cruncher, DirectionChoice, MemoryImage, Placement, SfxResult,
    TailoringChoice, DEFAULT_CLEARANCE, DEFAULT_MARGIN, DEFAULT_MOVER, INIT_BANK, RUN_BASIC_BANK,
    RUN_BASIC_LOOP, STACK_PAGE_SLOT,
};
use lzan_c64::{pick_routine, Direction, Format};

fn main() -> eframe::Result {
    let mut viewport = egui::ViewportBuilder::default()
        .with_title(format!("{} v{}", config::APP_NAME, config::VERSION))
        .with_inner_size([720.0, 820.0])
        .with_min_inner_size([600.0, 600.0]);
    // Window icon (title bar + taskbar while running). Path is relative to this
    // source file → the project's `icons/` directory.
    if let Ok(icon) = eframe::icon_data::from_png_bytes(include_bytes!("../icons/icon.png")) {
        viewport = viewport.with_icon(icon);
    }
    let options = eframe::NativeOptions {
        viewport,
        ..Default::default()
    };
    eframe::run_native(
        "LazyCruncher Workshop",
        options,
        Box::new(|_cc| Ok(Box::new(App::new()))),
    )
}

// ---------------------------------------------------------------------------
// Status log
// ---------------------------------------------------------------------------

#[derive(Clone, Copy, PartialEq)]
enum LogKind {
    /// Section divider / headline.
    Head,
    /// Neutral fact.
    Info,
    /// Success.
    Good,
    /// Non-fatal problem.
    Warn,
    /// Failure.
    Error,
    /// De-emphasized aside.
    Muted,
}

struct LogLine {
    kind: LogKind,
    text: String,
}

// ---------------------------------------------------------------------------
// Single crunch
// ---------------------------------------------------------------------------

enum Outcome {
    Ok(Report),
    Err(String),
}

struct Report {
    format_label: String,
    input_len: usize,
    output_len: usize,
    stream_len: usize,
    decoder_bytes: u16,
    direction: &'static str,
    packed_at: u16,
    decruncher_at: u16,
    scratch: Option<(u16, u16)>,
    mover_at: Option<u16>,
    mover_folded: bool,
    /// The payload was moved to its decode position (`false` = read in place,
    /// no-move layout).
    payload_moved: bool,
    span: (u32, u32),
    output_path: PathBuf,
    warnings: Vec<String>,
    /// The embedded decoder was trait-tailored to this stream (Exomizer).
    decoder_tailored: bool,
    /// Bytes the whole SFX shrank vs the standard decoder (0 if not tailored).
    decoder_saved: usize,
}

enum Status {
    Idle,
    Working,
    Done(Report),
    Failed(String),
}

// ---------------------------------------------------------------------------
// Compare Crunchers (all fitting formats in parallel)
// ---------------------------------------------------------------------------

/// One format's compare outcome: the payload facts the user asked to see.
#[derive(Clone)]
struct CompareData {
    total_len: usize,
    stream_len: usize,
    decoder_bytes: u16,
    direction: &'static str,
    /// Output size as a percentage of the summed input `.prg` size.
    ratio: f64,
}

#[derive(Clone)]
struct CompareRow {
    label: String,
    #[allow(dead_code)]
    format: Format,
    result: Result<CompareData, String>,
}

enum CompareMsg {
    Row(CompareRow),
    Done,
}

struct CompareRun {
    rx: Receiver<CompareMsg>,
    total: usize,
    done: usize,
    rows: Vec<CompareRow>,
    started: Instant,
}

/// One successfully imported (detected + unpacked) crunched file.
struct ImportedFile {
    name: String,
    path: PathBuf,
    part: ImportedPart,
}

/// A deeper crunch layer found under an imported region, unpacked silently and
/// awaiting the user's "go deeper?" decision. Games often nest crunchers (e.g. an
/// RLE inside a Time Cruncher); this lets the user peel one level at a time.
struct CascadeOffer {
    /// The deeper level, already unpacked.
    next: ImportedPart,
    /// Index of the region holding the current (shallower) level.
    region_idx: usize,
    /// Load + length of that region, to confirm it is still the one probed.
    cur_load: u16,
    cur_len: usize,
    /// Source path (start-default re-tracking) and display name base.
    path: PathBuf,
    name: String,
    /// Depth of the CURRENT level (1 = first import); the offer is for depth + 1.
    depth: usize,
}

/// What the loaded program's entry looks like, for picking the start address.
enum StartKind {
    /// ML program launched by a `SYS <addr>` BASIC stub.
    Sys(u16),
    /// A runnable BASIC program at $0801 (→ Force basic run).
    Basic,
    /// Plain machine code: the depacker's jump target, or the load address.
    Address(u16),
}

struct App {
    crunchers: Vec<Cruncher>,
    format_idx: usize,
    image: MemoryImage,
    load_error: Option<String>,
    output_path: Option<PathBuf>,
    start_auto: bool,
    start_text: String,
    // Placement fields (all overridable).
    decr_auto: bool,
    decr_text: String,
    mover_auto: bool,
    mover_text: String,
    scratch_auto: bool,
    scratch_text: String,
    margin_text: String,
    /// "No move" in-place clearance (bytes) — see `Placement::clearance`.
    clearance_text: String,
    bank_text: String,
    cli_before_jmp: bool,
    // "Restore $2D/$2E" (VARTAB) + the RUN convenience it unlocks.
    restore_2d2e: bool,
    restore_2d2e_text: String,
    restore_2d2e_auto: bool,
    run_basic: bool,
    dir_choice: DirectionChoice,
    /// Allow the embedded decruncher to use undocumented (illegal) 6502
    /// opcodes (default on). Off = legal-only decoder for portability.
    allow_illegal: bool,
    /// Per-crunch decoder tailoring (Exomizer only): Auto builds both the
    /// standard and the trait-tailored decoder and keeps the smaller.
    tailoring_choice: TailoringChoice,
    status: Status,
    rx: Option<Receiver<Outcome>>,
    /// Background "export compressed stream only" job: `(path, bytes written)`.
    export_rx: Option<Receiver<Result<(PathBuf, usize), String>>>,
    started: Option<Instant>,
    // "Import crunched" runs on its own thread (detection emulates the
    // depacker, which can take seconds on unrecognized files).
    import_rx: Option<Receiver<Result<ImportedFile, String>>>,
    import_notes: Vec<String>,
    /// Entry point the last imported file's depacker jumped to; start-address
    /// fallback when the image has no `SYS` stub.
    import_hint: Option<u16>,
    /// A running "is the last unpacked level itself crunched?" probe.
    cascade_rx: Option<Receiver<Option<CascadeOffer>>>,
    /// A deeper crunch layer found and awaiting the user's "go deeper?" answer.
    cascade_offer: Option<CascadeOffer>,
    /// Running "Compare Crunchers" job, if any.
    compare: Option<CompareRun>,
    /// The scrollable status/history log at the bottom of the window.
    log: Vec<LogLine>,
    /// The "really delete everything?" confirmation modal is open.
    confirm_clear: bool,
    /// Region (by index) whose start/end address is being edited inline, plus
    /// the in-progress field texts.
    editing_region: Option<usize>,
    edit_start_text: String,
    edit_end_text: String,
    /// The floating memory viewer/editor window (RetroViewer widget), created
    /// lazily on first open and kept across close/reopen so cursor and view
    /// mode survive.
    mem_view: Option<retroviewer::RetroViewer>,
    mem_view_open: bool,
    /// The last 64 KB [`MemoryImage::full_buffer`] synced into the viewer;
    /// diffed against the viewer's buffer to apply edits back to the regions.
    mem_view_snapshot: Vec<u8>,
    /// Help / About dialog visibility (opened from the Help menu).
    show_help: bool,
    show_about: bool,
}

impl App {
    fn new() -> Self {
        Self {
            crunchers: cruncher_list(),
            format_idx: 0,
            image: MemoryImage::default(),
            load_error: None,
            output_path: None,
            start_auto: true,
            start_text: String::new(),
            decr_auto: true,
            decr_text: String::new(),
            mover_auto: true,
            mover_text: format!("${DEFAULT_MOVER:04X}"),
            scratch_auto: true,
            scratch_text: String::new(),
            margin_text: DEFAULT_MARGIN.to_string(),
            clearance_text: DEFAULT_CLEARANCE.to_string(),
            bank_text: format!("${INIT_BANK:02X}"),
            cli_before_jmp: false,
            restore_2d2e: false,
            restore_2d2e_text: String::new(),
            restore_2d2e_auto: true,
            run_basic: false,
            dir_choice: DirectionChoice::Auto,
            allow_illegal: true,
            tailoring_choice: TailoringChoice::Auto,
            status: Status::Idle,
            rx: None,
            export_rx: None,
            started: None,
            import_rx: None,
            import_notes: Vec::new(),
            import_hint: None,
            cascade_rx: None,
            cascade_offer: None,
            compare: None,
            log: Vec::new(),
            confirm_clear: false,
            editing_region: None,
            edit_start_text: String::new(),
            edit_end_text: String::new(),
            mem_view: None,
            mem_view_open: false,
            mem_view_snapshot: Vec::new(),
            show_help: false,
            show_about: false,
        }
    }

    fn log(&mut self, kind: LogKind, text: impl Into<String>) {
        self.log.push(LogLine {
            kind,
            text: text.into(),
        });
    }

    /// A background job owns the app: block edits while any of them run.
    fn busy(&self) -> bool {
        matches!(self.status, Status::Working)
            || self.import_rx.is_some()
            || self.compare.is_some()
            || self.export_rx.is_some()
            || self.cascade_rx.is_some()
    }

    /// Wipe the whole workspace back to a fresh start. Keeps the status log (it
    /// is the session history) and appends a marker.
    fn clear_workspace(&mut self) {
        let n = self.image.regions().len();
        let notes = self.import_notes.len();
        self.image = MemoryImage::default();
        self.output_path = None;
        self.load_error = None;
        self.import_notes.clear();
        self.import_hint = None;
        self.import_rx = None;
        self.cascade_rx = None;
        self.cascade_offer = None;
        self.rx = None;
        self.compare = None;
        self.status = Status::Idle;
        self.started = None;
        self.editing_region = None;
        // Reset every control to its default so "New" is a true clean slate.
        self.format_idx = 0;
        self.start_auto = true;
        self.start_text.clear();
        self.decr_auto = true;
        self.decr_text.clear();
        self.mover_auto = true;
        self.mover_text = format!("${DEFAULT_MOVER:04X}");
        self.scratch_auto = true;
        self.scratch_text.clear();
        self.margin_text = DEFAULT_MARGIN.to_string();
        self.clearance_text = DEFAULT_CLEARANCE.to_string();
        self.bank_text = format!("${INIT_BANK:02X}");
        self.cli_before_jmp = false;
        self.restore_2d2e = false;
        self.restore_2d2e_text.clear();
        self.restore_2d2e_auto = true;
        self.run_basic = false;
        self.dir_choice = DirectionChoice::Auto;
        self.allow_illegal = true;
        self.tailoring_choice = TailoringChoice::Auto;
        self.mem_view = None;
        self.mem_view_open = false;
        self.mem_view_snapshot.clear();
        let _ = notes;
        self.log(
            LogKind::Head,
            format!("— Workspace cleared ({n} file(s) removed) —"),
        );
    }

    /// Reset to Idle unless a crunch is running. A background crunch owns the
    /// status until it finishes; region edits must not wipe its spinner, which
    /// would also re-enable the Crunch button and let a second thread race the
    /// output file.
    fn clear_status(&mut self) {
        if !matches!(self.status, Status::Working) {
            self.status = Status::Idle;
        }
    }

    fn add_prg(&mut self, path: PathBuf) {
        self.load_error = None;
        self.import_notes.clear();
        self.clear_status();
        let name = file_name(&path);
        match std::fs::read(&path) {
            Ok(bytes) => {
                if let Err(e) = self.image.add_prg(&name, &bytes) {
                    self.load_error = Some(e.clone());
                    self.log(LogKind::Error, format!("{name}: {e}"));
                    return;
                }
                if bytes.len() >= 2 {
                    let load = u16::from_le_bytes([bytes[0], bytes[1]]);
                    let size = bytes.len() - 2;
                    let end = load as u32 + size as u32;
                    self.log(
                        LogKind::Good,
                        format!(
                            "Added {name}: ${load:04X}-${:04X} ({size} B)",
                            end.saturating_sub(1)
                        ),
                    );
                }
                self.after_region_added(&path);
            }
            Err(e) => {
                self.load_error = Some(format!("{name}: could not read the file: {e}"));
                self.log(
                    LogKind::Error,
                    format!("{name}: could not read the file: {e}"),
                );
            }
        }
    }

    /// Bookkeeping shared by the add and import paths once a region landed in
    /// the image: default output name, start default, $2D/$2E re-tracking.
    fn after_region_added(&mut self, source: &std::path::Path) {
        if self.output_path.is_none() {
            let mut out = source.to_path_buf();
            let stem = source
                .file_stem()
                .map(|s| s.to_string_lossy().into_owned())
                .unwrap_or_else(|| "output".into());
            out.set_file_name(format!("{stem}-crunched.prg"));
            self.output_path = Some(out);
        }
        // Re-track the $0801 segment end for the $2D/$2E default
        // (a manual edit is expected after loading, not before).
        self.restore_2d2e_auto = true;
        // Only the first file drives the start default / BASIC detection —
        // later additions must not clobber the user's choices.
        if self.image.regions().len() == 1 {
            self.detect_and_apply_start();
        } else {
            self.refresh_start_default();
        }
    }

    fn push_load_error(&mut self, msg: String) {
        match &mut self.load_error {
            Some(prev) => {
                prev.push('\n');
                prev.push_str(&msg);
            }
            None => self.load_error = Some(msg),
        }
    }

    /// Save the current memory image as a plain, uncompressed `.prg` (load
    /// address = span start, gaps zero-filled). Useful after importing and
    /// unpacking a crunched file: the decrunched program can be written back out
    /// verbatim.
    fn export_uncompressed(&mut self) {
        let Some((lo, buf)) = self.image.span_buffer() else {
            return;
        };
        let mut dialog = rfd::FileDialog::new().add_filter("C64 program", &["prg"]);
        if let Some(p) = &self.output_path {
            if let Some(dir) = p.parent() {
                dialog = dialog.set_directory(dir);
            }
            let stem = p
                .file_stem()
                .map(|s| s.to_string_lossy().into_owned())
                .unwrap_or_else(|| "output".into());
            let stem = stem.strip_suffix("-crunched").unwrap_or(&stem);
            dialog = dialog.set_file_name(format!("{stem}-uncompressed.prg"));
        }
        let Some(path) = dialog.save_file() else {
            return;
        };
        let mut prg = Vec::with_capacity(buf.len() + 2);
        prg.push((lo & 0xFF) as u8);
        prg.push(((lo >> 8) & 0xFF) as u8);
        prg.extend_from_slice(&buf);
        match std::fs::write(&path, &prg) {
            Ok(()) => {
                self.load_error = None;
                // Report the span size (matches the address range and the app's
                // other notes); the .prg on disk is 2 bytes larger (load address).
                let note = format!(
                    "exported uncompressed ${:04X}-${:04X} ({} B) to {}",
                    lo,
                    lo + buf.len() as u32 - 1,
                    buf.len(),
                    file_name(&path)
                );
                self.import_notes.push(note.clone());
                self.log(LogKind::Good, note);
            }
            Err(e) => {
                self.push_load_error(format!("Could not write {}: {e}", file_name(&path)));
                self.log(
                    LogKind::Error,
                    format!("Could not write {}: {e}", file_name(&path)),
                );
            }
        }
    }

    /// Detect + unpack the picked files on a worker thread; each file reports
    /// back individually so the list fills in as results arrive.
    fn start_import(&mut self, paths: Vec<PathBuf>, ctx: &egui::Context) {
        self.load_error = None;
        self.import_notes.clear();
        self.clear_status();
        self.log(
            LogKind::Info,
            format!("Importing {} crunched file(s)…", paths.len()),
        );
        let (tx, rx): (Sender<Result<ImportedFile, String>>, _) = std::sync::mpsc::channel();
        self.import_rx = Some(rx);
        let ctx = ctx.clone();
        std::thread::spawn(move || {
            let ud = unidecrunch::UniDecrunch::new();
            for path in paths {
                let name = file_name(&path);
                let msg = match std::fs::read(&path) {
                    Err(e) => Err(format!("{name}: could not read the file: {e}")),
                    Ok(bytes) => {
                        // A foreign file drives the emulator; a panic in the
                        // engine must not take the thread (and batch) down.
                        match catch_unwind(AssertUnwindSafe(|| import_crunched_with(&ud, &bytes))) {
                            Ok(Ok(part)) => Ok(ImportedFile { name, path, part }),
                            Ok(Err(e)) => Err(format!("{name}: {e}")),
                            Err(_) => Err(format!("{name}: the unpack engine crashed internally.")),
                        }
                    }
                };
                let _ = tx.send(msg);
                ctx.request_repaint();
            }
        });
    }

    fn poll_imports(&mut self, ctx: &egui::Context) {
        // Drain first: applying a message needs `&mut self`, which the held
        // receiver borrow would block.
        let Some(rx) = &self.import_rx else { return };
        let mut messages = Vec::new();
        let mut finished = false;
        loop {
            match rx.try_recv() {
                Ok(m) => messages.push(m),
                Err(TryRecvError::Empty) => break,
                Err(TryRecvError::Disconnected) => {
                    finished = true;
                    break;
                }
            }
        }
        if finished {
            self.import_rx = None;
        }
        for m in messages {
            match m {
                Ok(file) => self.apply_import(file, ctx),
                Err(e) => {
                    self.push_load_error(e.clone());
                    self.log(LogKind::Error, e);
                }
            }
        }
    }

    fn apply_import(&mut self, file: ImportedFile, ctx: &egui::Context) {
        let ImportedFile { name, path, part } = file;
        self.clear_status();
        let listed = format!("{name} ({})", part.cruncher);
        if let Err(e) = self.image.add_prg(&listed, &part.prg) {
            self.push_load_error(e.clone());
            self.log(LogKind::Error, format!("{name}: {e}"));
            return;
        }
        let note = format!(
            "{name}: {} unpacked to ${:04X}-${:04X} ({} B)",
            part.cruncher,
            part.start,
            part.end,
            part.end as u32 - part.start as u32 + 1
        );
        self.import_notes.push(note.clone());
        self.log(LogKind::Good, note);
        self.import_hint = Some(part.jump_start);
        self.after_region_added(&path);
        // Cascade: silently check whether the unpacked program is ITSELF crunched
        // (nested crunchers are common — e.g. RLE inside a Time Cruncher). If so,
        // the user is offered one level deeper at a time.
        let idx = self.image.regions().len() - 1;
        let (cur_load, cur_len) = {
            let r = &self.image.regions()[idx];
            (r.load, r.data.len())
        };
        self.start_cascade_probe(
            part.prg,
            part.jump_start,
            idx,
            cur_load,
            cur_len,
            path,
            name,
            1,
            ctx,
        );
    }

    /// Spawn a background probe: is `prg` (the level-`depth` unpacked program)
    /// itself another crunched layer? Only one cascade runs at a time.
    #[allow(clippy::too_many_arguments)]
    fn start_cascade_probe(
        &mut self,
        prg: Vec<u8>,
        entry: u16,
        region_idx: usize,
        cur_load: u16,
        cur_len: usize,
        path: PathBuf,
        name: String,
        depth: usize,
        ctx: &egui::Context,
    ) {
        if self.cascade_rx.is_some() || self.cascade_offer.is_some() {
            return;
        }
        let (tx, rx): (Sender<Option<CascadeOffer>>, _) = std::sync::mpsc::channel();
        self.cascade_rx = Some(rx);
        let ctx = ctx.clone();
        std::thread::spawn(move || {
            let ud = unidecrunch::UniDecrunch::new();
            let offer = catch_unwind(AssertUnwindSafe(|| {
                probe_deeper_layer(&ud, &prg, Some(entry))
            }))
            .ok()
            .flatten()
            .map(|next| CascadeOffer {
                next,
                region_idx,
                cur_load,
                cur_len,
                path,
                name,
                depth,
            });
            let _ = tx.send(offer);
            ctx.request_repaint();
        });
    }

    fn poll_cascade(&mut self) {
        let Some(rx) = &self.cascade_rx else { return };
        match rx.try_recv() {
            Ok(offer) => {
                self.cascade_rx = None;
                if let Some(o) = offer {
                    self.log(
                        LogKind::Info,
                        format!(
                            "Detected a deeper crunched layer ({}) under the last import.",
                            o.next.cruncher
                        ),
                    );
                    self.cascade_offer = Some(o);
                }
            }
            Err(TryRecvError::Empty) => {}
            Err(TryRecvError::Disconnected) => self.cascade_rx = None,
        }
    }

    /// The "go one level deeper?" prompt shown when the cascade probe finds a
    /// nested crunch layer.
    fn cascade_offer_ui(&mut self, ctx: &egui::Context) {
        let Some(offer) = &self.cascade_offer else {
            return;
        };
        let level = offer.depth + 1;
        let cruncher = offer.next.cruncher.clone();
        let (start, end) = (offer.next.start, offer.next.end);
        let nbytes = end as u32 - start as u32 + 1;
        let kept_depth = offer.depth;
        let mut decision: Option<bool> = None; // Some(true) = deeper, Some(false) = keep
        egui::Modal::new(egui::Id::new("cascade_offer")).show(ctx, |ui| {
            ui.set_width(440.0);
            ui.heading(format!("Another crunched layer (level {level})"));
            ui.add_space(4.0);
            ui.label(format!(
                "The unpacked program is itself crunched with {cruncher}."
            ));
            ui.label(format!(
                "Unpacking one level deeper yields ${start:04X}-${end:04X} ({nbytes} B)."
            ));
            ui.add_space(10.0);
            ui.horizontal(|ui| {
                if ui
                    .button(egui::RichText::new(format!("⬇  Unpack level {level}")).strong())
                    .clicked()
                {
                    decision = Some(true);
                }
                if ui.button("Keep this level").clicked() {
                    decision = Some(false);
                }
            });
        });
        match decision {
            Some(true) => {
                if let Some(o) = self.cascade_offer.take() {
                    self.accept_deeper(o, ctx);
                }
            }
            Some(false) => {
                self.cascade_offer = None;
                self.log(LogKind::Muted, format!("Kept level {kept_depth}."));
            }
            None => {}
        }
    }

    /// Replace the current level's region with the deeper one, then probe for a
    /// level beyond it (so the user can keep peeling as far as the file goes).
    fn accept_deeper(&mut self, offer: CascadeOffer, ctx: &egui::Context) {
        let CascadeOffer {
            next,
            region_idx,
            cur_load,
            cur_len,
            path,
            name,
            depth,
        } = offer;
        let level = depth + 1;
        // Drop the current level's region if it is still the one we probed (the
        // user may have changed the region list while the probe ran).
        if self
            .image
            .regions()
            .get(region_idx)
            .is_some_and(|r| r.load == cur_load && r.data.len() == cur_len)
        {
            self.image.remove(region_idx);
        }
        let listed = format!("{name} (level {level} · {})", next.cruncher);
        if let Err(e) = self.image.add_prg(&listed, &next.prg) {
            self.push_load_error(e.clone());
            self.log(LogKind::Error, format!("{name}: {e}"));
            return;
        }
        let note = format!(
            "{name}: level {level} {} unpacked to ${:04X}-${:04X} ({} B)",
            next.cruncher,
            next.start,
            next.end,
            next.end as u32 - next.start as u32 + 1
        );
        self.import_notes.push(note.clone());
        self.log(LogKind::Good, note);
        self.import_hint = Some(next.jump_start);
        self.clear_status();
        self.after_region_added(&path);
        // Keep peeling: probe the level below this one.
        let new_idx = self.image.regions().len() - 1;
        let (nl, nlen) = {
            let r = &self.image.regions()[new_idx];
            (r.load, r.data.len())
        };
        self.start_cascade_probe(
            next.prg,
            next.jump_start,
            new_idx,
            nl,
            nlen,
            path,
            name,
            level,
            ctx,
        );
    }

    /// Default start = detected `0 SYS...` target of the region at $0801,
    /// else the entry the last imported file's depacker jumped to (when it
    /// still points into the image), else the lowest region's load address.
    fn default_start(&self) -> Option<u16> {
        let regions = self.image.regions();
        if let Some(r) = regions.iter().find(|r| r.load == 0x0801) {
            if let Some(sys) = detect_sys_start(r.load, &r.data) {
                return Some(sys);
            }
        }
        if let Some(jump) = self.import_hint {
            let inside = self
                .image
                .span()
                .is_some_and(|(lo, hi)| (jump as u32) >= lo && (jump as u32) < hi);
            if inside {
                return Some(jump);
            }
        }
        self.image.span().map(|(lo, _)| lo as u16)
    }

    fn refresh_start_default(&mut self) {
        if self.start_auto {
            if let Some(s) = self.default_start() {
                self.start_text = format!("${s:04X}");
            }
        }
    }

    /// Undo the field overrides RUN applied, back to the plain defaults
    /// (auto start, `$01 = INIT_BANK`, SEI). Called when RUN is switched off.
    fn reset_after_run(&mut self) {
        self.start_auto = true;
        self.refresh_start_default();
        self.bank_text = format!("${INIT_BANK:02X}");
        self.cli_before_jmp = false;
    }

    /// Classify the loaded program to pick a start address: a `SYS` stub, a
    /// runnable BASIC program, or plain machine code (depacker jump / load).
    fn detect_start_kind(&self) -> StartKind {
        if let Some(r) = self.image.regions().iter().find(|r| r.load == 0x0801) {
            if let Some(sys) = detect_sys_start(r.load, &r.data) {
                return StartKind::Sys(sys);
            }
            if looks_like_basic(r.load, &r.data) {
                return StartKind::Basic;
            }
        }
        if let Some(j) = self.import_hint {
            let inside = self
                .image
                .span()
                .is_some_and(|(lo, hi)| (j as u32) >= lo && (j as u32) < hi);
            if inside {
                return StartKind::Address(j);
            }
        }
        StartKind::Address(self.image.span().map(|(lo, _)| lo as u16).unwrap_or(0x0801))
    }

    /// Apply the detected start: fill the auto field with a `SYS` target, turn
    /// on "Force basic run" for a BASIC program, or use the extracted ML entry.
    /// Logs what it chose so the info window records the extracted address.
    fn detect_and_apply_start(&mut self) {
        match self.detect_start_kind() {
            StartKind::Sys(addr) => {
                self.set_basic_run(false);
                self.start_auto = true;
                self.start_text = format!("${addr:04X}");
                self.log(
                    LogKind::Info,
                    format!("Start address from BASIC SYS: ${addr:04X}"),
                );
            }
            StartKind::Basic => {
                self.set_basic_run(true);
                self.log(
                    LogKind::Info,
                    format!(
                        "BASIC program detected — Force basic run on ($01=#${RUN_BASIC_BANK:02X}, \
                         CLI, JMP ${RUN_BASIC_LOOP:04X})"
                    ),
                );
            }
            StartKind::Address(a) => {
                self.set_basic_run(false);
                self.start_auto = true;
                self.refresh_start_default();
                self.log(LogKind::Info, format!("Start address: ${a:04X}"));
            }
        }
    }

    /// Enable/disable "Force basic run": bank ROM in (`$01=#$37`), CLI, restore
    /// `$2D/$2E` to the program end, `JSR $A659` (CLR), and `JMP` the BASIC run
    /// loop in ROM. Reuses the tested `run_basic` epilogue path.
    fn set_basic_run(&mut self, on: bool) {
        if on {
            if self.run_basic {
                return; // already on — don't clobber later user edits
            }
            self.restore_2d2e = true;
            self.restore_2d2e_auto = true;
            if let Some(end) = self.basic_end_default() {
                self.restore_2d2e_text = format!("${end:04X}");
            }
            self.run_basic = true;
            self.start_auto = false;
            self.start_text = format!("${RUN_BASIC_LOOP:04X}");
            self.bank_text = format!("${RUN_BASIC_BANK:02X}");
            self.cli_before_jmp = true;
        } else if self.run_basic {
            self.run_basic = false;
            self.reset_after_run();
        }
    }

    /// End address ($2D/$2E / VARTAB) of the BASIC segment loaded at $0801 —
    /// the default value for the "Restore $2D/$2E" option. `end()` can be
    /// $10000 (a program filling to $FFFF), which is not a valid VARTAB, so
    /// such a region yields no default.
    fn basic_end_default(&self) -> Option<u16> {
        self.image
            .regions()
            .iter()
            .find(|r| r.load == 0x0801)
            .and_then(|r| u16::try_from(r.end()).ok())
    }

    fn placement(&self) -> Result<Placement, String> {
        let decruncher = if self.decr_auto {
            None
        } else {
            Some(parse_addr(&self.decr_text).map_err(|e| format!("Decrunch address: {e}"))?)
        };
        let mover = if self.mover_auto {
            None
        } else {
            Some(parse_addr(&self.mover_text).map_err(|e| format!("Mover address: {e}"))?)
        };
        let scratch = if self.scratch_auto {
            None
        } else {
            Some(parse_addr(&self.scratch_text).map_err(|e| format!("Buffer address: {e}"))?)
        };
        let margin = parse_addr(&self.margin_text)
            .map_err(|_| format!("Invalid margin: {}", self.margin_text))?;
        let clearance = parse_addr(&self.clearance_text)
            .map_err(|_| format!("Invalid clearance: {}", self.clearance_text))?;
        let bank = parse_addr(&self.bank_text).map_err(|e| format!("$01 value: {e}"))?;
        let bank_at_jmp = u8::try_from(bank)
            .map_err(|_| format!("$01 value: ${bank:X} is not a single byte (max $FF)"))?;
        let restore_basic_end = if self.restore_2d2e {
            Some(parse_addr(&self.restore_2d2e_text).map_err(|e| format!("Restore $2D/$2E: {e}"))?)
        } else {
            None
        };
        // RUN starts a decompressed BASIC program: it forces the banking and
        // interrupt state BASIC needs plus the CLR call, independent of the
        // (disabled) manual fields. start_crunch sets the JMP target to the
        // interpreter loop to match.
        let (bank_at_jmp, cli_before_jmp, basic_clr) = if self.run_basic {
            (RUN_BASIC_BANK, true, true)
        } else {
            (bank_at_jmp, self.cli_before_jmp, false)
        };
        Ok(Placement {
            decruncher,
            mover,
            scratch,
            margin,
            direction: self.dir_choice,
            bank_at_jmp,
            cli_before_jmp,
            restore_basic_end,
            basic_clr,
            allow_illegal: self.allow_illegal,
            tsc_shift: 0, // computed per stream during the build
            tailoring: self.tailoring_choice,
            clearance,
        })
    }

    /// The JMP-after-unpack target the current settings resolve to.
    fn resolved_start(&self) -> Result<u16, String> {
        if self.run_basic {
            return Ok(RUN_BASIC_LOOP);
        }
        match (self.start_auto, self.default_start()) {
            (true, Some(s)) => Ok(s),
            _ => parse_addr(&self.start_text).map_err(|e| format!("Start address: {e}")),
        }
    }

    /// Staged decruncher size for `fmt` under the current placement: routine +
    /// entry wrapper + the pre-JMP epilogue. Recomputed from the live options,
    /// so it tracks every choice that changes the emitted decruncher —
    /// direction, illegal opcodes, interrupt mode, `$01` banking, `$2D/$2E`
    /// restore and RUN.
    fn decoder_bytes(&self, fmt: Format) -> Option<u32> {
        if let Ok(p) = self.placement() {
            if let Ok(pp) = plan_preview(&self.image, fmt, &p) {
                if pp.unavailable.is_none() {
                    return Some(pp.staged_size);
                }
            }
        }
        // No file loaded yet (or format unavailable): fall back to the raw
        // routine size for the illegal-opcode choice.
        pick_routine(fmt, Direction::Forward, self.allow_illegal).map(|s| s.code_bytes as u32)
    }

    /// Dropdown text for cruncher `i`: the bare name plus benchmarked speeds and
    /// the live decoder size, all inside the parentheses.
    fn cruncher_entry(&self, i: usize) -> String {
        let fmt = self.crunchers[i].format;
        let dec = self
            .decoder_bytes(fmt)
            .map(|b| format!("{b} B"))
            .unwrap_or_else(|| "—".into());
        format!(
            "{}  (packing speed {}, unpacking speed {}, decoder {dec})",
            self.crunchers[i].label,
            ranks::score_str(ranks::pack_speed(fmt)),
            ranks::score_str(ranks::decr_speed(fmt)),
        )
    }

    fn start_crunch(&mut self, ctx: &egui::Context) {
        let Some(output_path) = self.output_path.clone() else {
            return;
        };
        if self.image.is_empty() {
            return;
        }
        let start_addr = match self.resolved_start() {
            Ok(a) => a,
            Err(e) => {
                self.status = Status::Failed(e.clone());
                self.log(LogKind::Error, e);
                return;
            }
        };
        let placement = match self.placement() {
            Ok(p) => p,
            Err(e) => {
                self.status = Status::Failed(e.clone());
                self.log(LogKind::Error, e);
                return;
            }
        };

        let format = self.crunchers[self.format_idx].format;
        let label = short_label(self.crunchers[self.format_idx].label).to_string();
        let allow_illegal = self.allow_illegal;
        let image = self.image.clone();
        let input_len: usize = image.regions().iter().map(|r| r.data.len() + 2).sum();
        self.log(LogKind::Info, format!("Crunching with {label}…"));
        let (tx, rx): (Sender<Outcome>, Receiver<Outcome>) = std::sync::mpsc::channel();
        self.rx = Some(rx);
        self.status = Status::Working;
        self.started = Some(Instant::now());

        let ctx = ctx.clone();
        std::thread::spawn(move || {
            // Encoders can panic on inputs they cannot represent; a library
            // panic must become a clean error, never a stuck spinner.
            let built = catch_unwind(AssertUnwindSafe(|| {
                build_sfx(&image, format, start_addr, &placement)
            }));
            let outcome = match built {
                Ok(Ok(result)) => match std::fs::write(&output_path, &result.prg) {
                    Ok(()) => Outcome::Ok(report_of(
                        result,
                        input_len,
                        output_path,
                        label,
                        format,
                        allow_illegal,
                    )),
                    Err(e) => Outcome::Err(format!("Could not write output file: {e}")),
                },
                Ok(Err(e)) => Outcome::Err(e),
                Err(panic) => Outcome::Err(panic_to_message(&panic, format)),
            };
            let _ = tx.send(outcome);
            ctx.request_repaint();
        });
    }

    /// Export ONLY the compressed stream (the packed data, no SFX wrapper) to a
    /// `.bin` file. Runs the same compression a crunch would, on a background
    /// thread, and writes [`SfxResult::stream`]. Independent of the main crunch
    /// (needs no prior "Save as…" or crunch).
    fn start_export_stream(&mut self, ctx: &egui::Context) {
        if self.image.is_empty() || self.busy() {
            return;
        }
        let start_addr = match self.resolved_start() {
            Ok(a) => a,
            Err(e) => {
                self.log(LogKind::Error, e);
                return;
            }
        };
        let placement = match self.placement() {
            Ok(p) => p,
            Err(e) => {
                self.log(LogKind::Error, e);
                return;
            }
        };
        // Default name: the output stem with a "-crunched.bin" suffix (stripping
        // any existing "-crunched" so it never doubles up), e.g.
        // "game-crunched.prg" -> "game-crunched.bin".
        let mut dialog = rfd::FileDialog::new().add_filter("Compressed stream", &["bin"]);
        if let Some(dir) = self.output_path.as_ref().and_then(|p| p.parent()) {
            dialog = dialog.set_directory(dir);
        }
        let base = self
            .output_path
            .as_ref()
            .and_then(|p| p.file_stem())
            .map(|s| {
                let s = s.to_string_lossy();
                s.strip_suffix("-crunched").unwrap_or(&s).to_string()
            });
        dialog = dialog.set_file_name(format!(
            "{}-crunched.bin",
            base.as_deref().unwrap_or("output")
        ));
        let Some(path) = dialog.save_file() else {
            return;
        };

        let format = self.crunchers[self.format_idx].format;
        let label = short_label(self.crunchers[self.format_idx].label).to_string();
        let image = self.image.clone();
        self.log(
            LogKind::Info,
            format!("Exporting {label} compressed stream…"),
        );
        let (tx, rx): (
            Sender<Result<(PathBuf, usize), String>>,
            Receiver<Result<(PathBuf, usize), String>>,
        ) = std::sync::mpsc::channel();
        self.export_rx = Some(rx);
        let ctx = ctx.clone();
        std::thread::spawn(move || {
            let built = catch_unwind(AssertUnwindSafe(|| {
                build_sfx(&image, format, start_addr, &placement)
            }));
            let result = match built {
                Ok(Ok(r)) => match std::fs::write(&path, &r.stream) {
                    Ok(()) => Ok((path, r.stream.len())),
                    Err(e) => Err(format!("Could not write stream file: {e}")),
                },
                Ok(Err(e)) => Err(e),
                Err(panic) => Err(panic_to_message(&panic, format)),
            };
            let _ = tx.send(result);
            ctx.request_repaint();
        });
    }

    fn poll_export(&mut self) {
        let Some(rx) = &self.export_rx else { return };
        let msg = match rx.try_recv() {
            Ok(m) => Some(m),
            Err(TryRecvError::Disconnected) => {
                Some(Err("The export thread exited without a result.".to_string()))
            }
            Err(TryRecvError::Empty) => None,
        };
        let Some(msg) = msg else { return };
        self.export_rx = None;
        match msg {
            Ok((path, len)) => self.log(
                LogKind::Good,
                format!("Exported {len} B compressed stream → {}", file_name(&path)),
            ),
            Err(e) => self.log(LogKind::Error, format!("Stream export failed: {e}")),
        }
    }

    fn poll(&mut self) {
        let Some(rx) = &self.rx else { return };
        let outcome = match rx.try_recv() {
            Ok(o) => Some(o),
            Err(TryRecvError::Disconnected) => Some(Outcome::Err(
                "The compression thread exited without a result.".into(),
            )),
            Err(TryRecvError::Empty) => None,
        };
        let Some(outcome) = outcome else { return };
        self.rx = None;
        match outcome {
            Outcome::Ok(r) => {
                let saved = r.input_len as i64 - r.output_len as i64;
                let ratio = if r.input_len > 0 {
                    r.output_len as f64 / r.input_len as f64 * 100.0
                } else {
                    0.0
                };
                let decoder_note = if r.decoder_tailored {
                    format!(
                        "decoder {} B (tailored, −{} B)",
                        r.decoder_bytes, r.decoder_saved
                    )
                } else {
                    format!("decoder {} B", r.decoder_bytes)
                };
                self.log(
                    LogKind::Good,
                    format!(
                        "{}: {} → {} B ({ratio:.1}%, saved {saved} B); stream {} B, {}, {}",
                        r.format_label,
                        r.input_len,
                        r.output_len,
                        r.stream_len,
                        decoder_note,
                        r.direction
                    ),
                );
                for w in &r.warnings {
                    self.log(LogKind::Warn, w.clone());
                }
                self.status = Status::Done(r);
            }
            Outcome::Err(e) => {
                self.log(LogKind::Error, format!("Crunch failed: {e}"));
                self.status = Status::Failed(e);
            }
        }
    }

    // -- Compare Crunchers ---------------------------------------------------

    fn start_compare(&mut self, ctx: &egui::Context) {
        if self.image.is_empty() || self.busy() {
            return;
        }
        let start_addr = match self.resolved_start() {
            Ok(a) => a,
            Err(e) => {
                self.log(LogKind::Error, format!("Compare: {e}"));
                return;
            }
        };
        let placement = match self.placement() {
            Ok(p) => p,
            Err(e) => {
                self.log(LogKind::Error, format!("Compare: {e}"));
                return;
            }
        };
        let jobs: Vec<(Format, String)> = self
            .available_crunchers()
            .into_iter()
            .map(|i| {
                (
                    self.crunchers[i].format,
                    short_label(self.crunchers[i].label).to_string(),
                )
            })
            .collect();
        if jobs.is_empty() {
            self.log(
                LogKind::Warn,
                "Compare: no cruncher fits the current layout.",
            );
            return;
        }
        let total = jobs.len();
        let input_len: usize = self.image.regions().iter().map(|r| r.data.len() + 2).sum();
        let cores = std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(1);
        // Leave CPU capacity available for the user interface.
        let workers = ((cores * 2) / 3).max(1).min(total);
        self.log(
            LogKind::Head,
            format!("— Compare Crunchers: {total} formats on {workers} of {cores} cores —"),
        );

        let image = self.image.clone();
        let jobs = Arc::new(jobs);
        let cursor = Arc::new(AtomicUsize::new(0));
        let (tx, rx) = std::sync::mpsc::channel::<CompareMsg>();
        let ctx = ctx.clone();
        std::thread::spawn(move || {
            let mut handles = Vec::new();
            for _ in 0..workers {
                let tx = tx.clone();
                let jobs = jobs.clone();
                let cursor = cursor.clone();
                let image = image.clone();
                let ctx = ctx.clone();
                handles.push(std::thread::spawn(move || loop {
                    let i = cursor.fetch_add(1, Ordering::Relaxed);
                    if i >= jobs.len() {
                        break;
                    }
                    let (fmt, label) = (jobs[i].0, jobs[i].1.clone());
                    let result = compare_build(&image, fmt, start_addr, &placement, input_len);
                    let _ = tx.send(CompareMsg::Row(CompareRow {
                        label,
                        format: fmt,
                        result,
                    }));
                    ctx.request_repaint();
                }));
            }
            for h in handles {
                let _ = h.join();
            }
            let _ = tx.send(CompareMsg::Done);
            ctx.request_repaint();
        });

        self.compare = Some(CompareRun {
            rx,
            total,
            done: 0,
            rows: Vec::new(),
            started: Instant::now(),
        });
    }

    fn poll_compare(&mut self) {
        if self.compare.is_none() {
            return;
        }
        let mut incoming = Vec::new();
        let mut done = false;
        {
            let run = self.compare.as_ref().unwrap();
            loop {
                match run.rx.try_recv() {
                    Ok(CompareMsg::Row(r)) => incoming.push(r),
                    Ok(CompareMsg::Done) => {
                        done = true;
                        break;
                    }
                    Err(TryRecvError::Empty) => break,
                    Err(TryRecvError::Disconnected) => {
                        done = true;
                        break;
                    }
                }
            }
        }
        for row in incoming {
            self.log_compare_row(&row);
            if let Some(run) = &mut self.compare {
                run.done += 1;
                run.rows.push(row);
            }
        }
        if done {
            if let Some(run) = self.compare.take() {
                self.log_compare_summary(&run.rows, run.started.elapsed().as_secs_f64());
            }
        }
    }

    fn log_compare_row(&mut self, row: &CompareRow) {
        match &row.result {
            Ok(d) => self.log(
                LogKind::Info,
                format!(
                    "  {:<11} {:>6} B  {:>5.1}%  stream {:>6} B  decoder {:>3} B  {}",
                    row.label, d.total_len, d.ratio, d.stream_len, d.decoder_bytes, d.direction
                ),
            ),
            // A build failure here means the format simply cannot fit this
            // layout (e.g. its in-place stream expansion needs more room than
            // the span leaves). That is a "skip", not an error — the summary
            // lists which were skipped.
            Err(_) => self.log(
                LogKind::Muted,
                format!("  {:<11} skipped — does not fit this layout", row.label),
            ),
        }
    }

    fn log_compare_summary(&mut self, rows: &[CompareRow], secs: f64) {
        let mut ok: Vec<&CompareRow> = rows.iter().filter(|r| r.result.is_ok()).collect();
        ok.sort_by_key(|r| r.result.as_ref().map(|d| d.total_len).unwrap_or(usize::MAX));
        self.log(LogKind::Head, "Ranking — smallest total .prg first:");
        let best = ok
            .first()
            .and_then(|r| r.result.as_ref().ok())
            .map(|d| d.total_len);
        for (i, r) in ok.iter().enumerate() {
            let d = r.result.as_ref().unwrap();
            let delta = best.map(|b| d.total_len - b).unwrap_or(0);
            let tag = if delta == 0 {
                "best".to_string()
            } else {
                format!("+{delta} B")
            };
            self.log(
                if i == 0 { LogKind::Good } else { LogKind::Info },
                format!(
                    "  {:>2}. {:<11} {:>6} B  {:>5.1}%  stream {:>6} B  decoder {:>3} B  {:<8}  {}",
                    i + 1,
                    r.label,
                    d.total_len,
                    d.ratio,
                    d.stream_len,
                    d.decoder_bytes,
                    d.direction,
                    tag
                ),
            );
        }
        let skipped: Vec<&str> = rows
            .iter()
            .filter(|r| r.result.is_err())
            .map(|r| r.label.as_str())
            .collect();
        if !skipped.is_empty() {
            self.log(
                LogKind::Muted,
                format!(
                    "Skipped {} that do not fit this layout: {}",
                    skipped.len(),
                    skipped.join(", ")
                ),
            );
        }
        self.log(LogKind::Muted, format!("Compare finished in {secs:.1}s."));
    }
}

fn compare_build(
    image: &MemoryImage,
    fmt: Format,
    start_addr: u16,
    placement: &Placement,
    input_len: usize,
) -> Result<CompareData, String> {
    let built = catch_unwind(AssertUnwindSafe(|| {
        build_sfx(image, fmt, start_addr, placement)
    }));
    match built {
        Ok(Ok(r)) => {
            // Report the ACTUAL decoder size: when Auto picked the tailored
            // Exomizer body, subtract what it saved from the static baseline so
            // the column matches the built .prg (same logic as `report_of`).
            let static_bytes = pick_routine(fmt, r.direction, placement.allow_illegal)
                .map(|s| s.code_bytes)
                .unwrap_or(0);
            let decoder_bytes = static_bytes.saturating_sub(r.decoder_saved as u16);
            Ok(CompareData {
                total_len: r.prg.len(),
                stream_len: r.stream_len,
                decoder_bytes,
                direction: dir_str(r.direction),
                ratio: if input_len > 0 {
                    r.prg.len() as f64 / input_len as f64 * 100.0
                } else {
                    0.0
                },
            })
        }
        Ok(Err(e)) => Err(e),
        Err(_) => Err("internal panic".into()),
    }
}

fn dir_str(d: Direction) -> &'static str {
    match d {
        Direction::Forward => "forward",
        Direction::Backward => "backward",
    }
}

/// Drop the parenthetical hint from a dropdown label ("LZAN full  (…)" →
/// "LZAN full") for compact log/report lines.
fn short_label(label: &str) -> &str {
    label.split("  ").next().unwrap_or(label).trim()
}

fn report_of(
    r: SfxResult,
    input_len: usize,
    output_path: PathBuf,
    format_label: String,
    format: Format,
    allow_illegal: bool,
) -> Report {
    let static_bytes = pick_routine(format, r.direction, allow_illegal)
        .map(|s| s.code_bytes)
        .unwrap_or(0);
    // When tailored, the real decoder is smaller than the static baseline by the
    // same bytes the whole SFX shrank (only the body changed).
    let decoder_bytes = static_bytes.saturating_sub(r.decoder_saved as u16);
    Report {
        format_label,
        input_len,
        output_len: r.prg.len(),
        stream_len: r.stream_len,
        decoder_bytes,
        direction: dir_str(r.direction),
        packed_at: r.packed_at,
        decruncher_at: r.decruncher_at,
        scratch: r.scratch,
        mover_at: r.mover_at,
        mover_folded: r.mover_folded,
        payload_moved: r.payload_moved,
        span: r.span,
        output_path,
        warnings: r.warnings,
        decoder_tailored: r.decoder_tailored,
        decoder_saved: r.decoder_saved,
    }
}

fn panic_to_message(panic: &Box<dyn std::any::Any + Send>, format: lzan_c64::Format) -> String {
    let raw = panic
        .downcast_ref::<&str>()
        .map(|s| (*s).to_string())
        .or_else(|| panic.downcast_ref::<String>().cloned())
        .unwrap_or_else(|| "unknown internal error".to_string());
    format!(
        "Compression crashed internally in \"{}\": {raw}. Try a different cruncher.",
        format.as_str()
    )
}

impl eframe::App for App {
    fn ui(&mut self, ui: &mut egui::Ui, _frame: &mut eframe::Frame) {
        self.poll();
        self.poll_imports(ui.ctx());
        self.poll_compare();
        self.poll_export();
        self.poll_cascade();
        if self.busy() {
            ui.ctx().request_repaint();
        }

        // Cascade "go deeper?" prompt (its own foreground modal layer).
        self.cascade_offer_ui(ui.ctx());

        // Confirmation modal (drawn on its own foreground layer, above panels).
        if self.confirm_clear {
            let resp = egui::Modal::new(egui::Id::new("confirm_clear")).show(ui.ctx(), |ui| {
                ui.set_max_width(380.0);
                ui.heading("Clear workspace?");
                ui.add_space(6.0);
                ui.label(
                    "This removes every added and imported file and resets all settings so you \
                     can start fresh. The status log below is kept.",
                );
                ui.add_space(14.0);
                ui.horizontal(|ui| {
                    let danger = egui::Button::new(
                        egui::RichText::new("🗑  Delete everything").color(egui::Color32::WHITE),
                    )
                    .fill(egui::Color32::from_rgb(0xB0, 0x3A, 0x3A));
                    if ui.add(danger).clicked() {
                        self.clear_workspace();
                        self.confirm_clear = false;
                    }
                    if ui.button("Cancel").clicked() {
                        self.confirm_clear = false;
                    }
                });
            });
            if resp.should_close() {
                self.confirm_clear = false;
            }
        }

        // About dialog (own foreground modal layer).
        if self.show_about {
            let resp = egui::Modal::new(egui::Id::new("about")).show(ui.ctx(), |ui| {
                ui.set_max_width(430.0);
                ui.heading(format!("{} v{}", config::APP_NAME, config::VERSION));
                ui.add_space(6.0);
                ui.label(
                    "Packs one or more C64 .prg files into a single self-extracting, \
                     crunched .prg.",
                );
                ui.add_space(8.0);
                ui.label(
                    "Compression by lzan-c64, import and decrunch by UniDecrunch, \
                     memory viewer by RetroViewer.",
                );
                ui.add_space(8.0);
                ui.hyperlink_to(
                    "Project on GitHub",
                    "https://github.com/tommyo123-dev/LazyCruncherWorkshop",
                );
                ui.add_space(4.0);
                ui.label("MIT licensed.");
                ui.add_space(12.0);
                if ui.button("Close").clicked() {
                    self.show_about = false;
                }
            });
            if resp.should_close() {
                self.show_about = false;
            }
        }

        // Help dialog (own foreground modal layer).
        if self.show_help {
            let resp = egui::Modal::new(egui::Id::new("help")).show(ui.ctx(), |ui| {
                ui.set_max_width(470.0);
                ui.heading("Help");
                ui.add_space(6.0);
                ui.label("1. Add one or more .prg files (File menu, or the toolbar).");
                ui.label("2. Pick a cruncher and adjust the placement and start options.");
                ui.label("3. Choose an output file and build the self-extracting .prg.");
                ui.add_space(8.0);
                ui.label("Import crunched: detect and unpack an already-crunched .prg.");
                ui.label("Memory viewer (View menu): inspect and edit the 64 KB image.");
                ui.add_space(8.0);
                ui.hyperlink_to(
                    "Documentation on GitHub",
                    "https://github.com/tommyo123-dev/LazyCruncherWorkshop",
                );
                ui.add_space(12.0);
                if ui.button("Close").clicked() {
                    self.show_help = false;
                }
            });
            if resp.should_close() {
                self.show_help = false;
            }
        }

        // Region start/end editor (own foreground modal).
        self.edit_region_modal(ui);

        // Floating memory viewer/editor (kept in sync with the regions).
        self.mem_view_ui(ui.ctx());

        // Menu bar across the very top.
        egui::Panel::top("menu_bar").show(ui, |ui| self.menu_bar_ui(ui));
        // Toolbar (toolbox) directly under the menu, always visible.
        egui::Panel::top("toolbar").show(ui, |ui| {
            ui.add_space(2.0);
            self.toolbar_ui(ui);
            ui.add_space(2.0);
        });

        // Bottom-of-window log (added first → sits at the very bottom).
        egui::Panel::bottom("status_log")
            .resizable(true)
            .default_size(210.0)
            .min_size(90.0)
            .show(ui, |ui| self.log_ui(ui));

        // Primary actions, always visible above the log.
        egui::Panel::bottom("actions")
            .resizable(false)
            .show(ui, |ui| self.actions_ui(ui));

        // Everything else, scrollable so the window never overflows.
        egui::CentralPanel::default().show(ui, |ui| {
            egui::ScrollArea::vertical()
                .auto_shrink([false, false])
                .show(ui, |ui| {
                    ui.spacing_mut().item_spacing.y = 6.0;
                    self.section(ui, |app, ui| app.region_list_ui(ui));
                    self.section(ui, |app, ui| app.cruncher_ui(ui));
                    self.section(ui, |app, ui| app.placement_ui(ui));
                    self.section(ui, |app, ui| app.start_addr_ui(ui));
                    self.section(ui, |app, ui| app.output_ui(ui));
                    self.section(ui, |app, ui| app.result_ui(ui));
                    ui.add_space(6.0);
                });
        });
    }
}

impl App {
    /// Wrap one UI section with a separator and vertical breathing room so the
    /// window reads as distinct blocks rather than one dense wall of controls.
    fn section(&mut self, ui: &mut egui::Ui, add: impl FnOnce(&mut App, &mut egui::Ui)) {
        ui.add_space(8.0);
        ui.separator();
        ui.add_space(4.0);
        add(self, ui);
    }

    fn menu_bar_ui(&mut self, ui: &mut egui::Ui) {
        let busy = self.busy();
        egui::MenuBar::new().ui(ui, |ui| {
            ui.menu_button("File", |ui| {
                if ui
                    .add_enabled(!busy, egui::Button::new("🗋  New / Clear"))
                    .clicked()
                {
                    self.confirm_clear = true;
                    ui.close();
                }
                if ui
                    .add_enabled(!busy, egui::Button::new("➕  Add .prg…"))
                    .clicked()
                {
                    if let Some(paths) = rfd::FileDialog::new()
                        .add_filter("C64 program", &["prg", "PRG"])
                        .pick_files()
                    {
                        for p in paths {
                            self.add_prg(p);
                        }
                    }
                    ui.close();
                }
                if ui
                    .add_enabled(!busy, egui::Button::new("📦  Import crunched…"))
                    .clicked()
                {
                    if let Some(paths) = rfd::FileDialog::new()
                        .add_filter("C64 program", &["prg", "PRG"])
                        .pick_files()
                    {
                        self.start_import(paths, ui.ctx());
                    }
                    ui.close();
                }
                if ui
                    .add_enabled(
                        !busy && !self.image.is_empty(),
                        egui::Button::new("💾  Export uncompressed…"),
                    )
                    .clicked()
                {
                    self.export_uncompressed();
                    ui.close();
                }
                ui.separator();
                if ui.button("🚪  Exit").clicked() {
                    ui.ctx().send_viewport_cmd(egui::ViewportCommand::Close);
                }
            });
            ui.menu_button("View", |ui| {
                if ui
                    .checkbox(&mut self.mem_view_open, "🔍  Memory viewer")
                    .clicked()
                {
                    ui.close();
                }
                if ui
                    .add_enabled(!self.log.is_empty(), egui::Button::new("🧹  Clear log"))
                    .clicked()
                {
                    self.log.clear();
                    ui.close();
                }
            });
            ui.menu_button("Help", |ui| {
                if ui.button("❓  Help").clicked() {
                    self.show_help = true;
                    ui.close();
                }
                if ui.button("ℹ  About").clicked() {
                    self.show_about = true;
                    ui.close();
                }
            });
        });
    }

    fn toolbar_ui(&mut self, ui: &mut egui::Ui) {
        let busy = self.busy();
        ui.horizontal_wrapped(|ui| {
            if ui
                .add_enabled(!busy, egui::Button::new("🗋  New / Clear"))
                .on_hover_text("Remove all files and reset every setting (asks first).")
                .clicked()
            {
                self.confirm_clear = true;
            }
            ui.separator();
            if ui
                .add_enabled(!busy, egui::Button::new("➕  Add .prg…"))
                .clicked()
            {
                if let Some(paths) = rfd::FileDialog::new()
                    .add_filter("C64 program", &["prg", "PRG"])
                    .pick_files()
                {
                    for p in paths {
                        self.add_prg(p);
                    }
                }
            }
            if ui
                .add_enabled(!busy, egui::Button::new("📦  Import crunched…"))
                .on_hover_text("Detect and unpack an already-crunched .prg, then add the result.")
                .clicked()
            {
                if let Some(paths) = rfd::FileDialog::new()
                    .add_filter("C64 program", &["prg", "PRG"])
                    .pick_files()
                {
                    self.start_import(paths, ui.ctx());
                }
            }
            if ui
                .add_enabled(
                    !busy && !self.image.is_empty(),
                    egui::Button::new("💾  Export uncompressed…"),
                )
                .on_hover_text("Save the current span as a plain .prg (no compression).")
                .clicked()
            {
                self.export_uncompressed();
            }
            if ui
                .selectable_label(self.mem_view_open, "🔍  Memory viewer")
                .on_hover_text(
                    "View and edit the 64 KB memory image (hex + disassembly). Edits inside \
                     a region change it; edits elsewhere become new regions.",
                )
                .clicked()
            {
                self.mem_view_open = !self.mem_view_open;
            }
        });
    }

    fn region_list_ui(&mut self, ui: &mut egui::Ui) {
        ui.strong("Memory regions");
        ui.add_space(2.0);
        if let Some(err) = &self.load_error {
            ui.colored_label(egui::Color32::from_rgb(0xE0, 0x50, 0x50), err);
        }
        if self.import_rx.is_some() {
            ui.horizontal(|ui| {
                ui.spinner();
                ui.weak("detecting and unpacking…");
            });
        }
        if self.image.is_empty() {
            ui.weak("No files added. Use “Add .prg…” or “Import crunched…”.");
            return;
        }

        let overlap: std::collections::HashSet<usize> = self
            .image
            .overlap_pairs()
            .into_iter()
            .flat_map(|(a, b)| [a, b])
            .collect();

        let busy = self.busy();
        let editing_active = self.editing_region.is_some();
        let mut delete: Option<usize> = None;
        let mut start_edit: Option<usize> = None;
        egui::Grid::new("regions")
            .num_columns(4)
            .spacing([14.0, 4.0])
            .show(ui, |ui| {
                for idx in self.image.sorted_indices() {
                    let r = &self.image.regions()[idx];
                    ui.monospace(format!("${:04X}-${:04X}", r.start(), r.end() - 1));
                    ui.monospace(format!("{} B", r.end() - r.start()));
                    ui.label(&r.name);
                    ui.horizontal(|ui| {
                        if overlap.contains(&idx) {
                            ui.colored_label(
                                egui::Color32::from_rgb(0xD0, 0xB0, 0x50),
                                "⚠ overlap",
                            );
                        }
                        if ui
                            .add_enabled(!busy && !editing_active, egui::Button::new("✏").small())
                            .on_hover_text(
                                "Edit start/end address — shrinking trims bytes (a gap between \
                             chunks becomes zero-fill), growing zero-fills the new area.",
                            )
                            .clicked()
                        {
                            start_edit = Some(idx);
                        }
                        if ui
                            .add_enabled(!busy, egui::Button::new("🗑").small())
                            .clicked()
                        {
                            delete = Some(idx);
                        }
                    });
                    ui.end_row();
                }
            });
        if let Some(idx) = start_edit {
            let r = &self.image.regions()[idx];
            self.edit_start_text = format!("${:04X}", r.start());
            self.edit_end_text = format!("${:04X}", r.end() - 1);
            self.editing_region = Some(idx);
        }
        if let Some(idx) = delete {
            self.image.remove(idx);
            // An imported region's leftovers must not outlive it: a stale
            // depacker jump would otherwise become the auto start address for
            // an unrelated region, and the success notes would describe a
            // region that is gone.
            self.import_hint = None;
            self.import_notes.clear();
            self.editing_region = None;
            self.refresh_start_default();
            self.clear_status();
            self.log(LogKind::Info, "Removed a region.");
        }
        if let Some((lo, hi)) = self.image.span() {
            ui.add_space(2.0);
            ui.horizontal(|ui| {
                ui.label("Compressed span:");
                ui.monospace(format!(
                    "${lo:04X}-${:04X}  ({} B, gaps zero-filled)",
                    hi - 1,
                    hi - lo
                ));
            });
        }
    }

    /// Apply an accepted start/end edit to region `idx`, then re-track the
    /// defaults the moved span affects.
    /// The floating memory viewer/editor. Two-way sync each frame while open:
    /// bytes typed in the viewer are diffed against the last synced snapshot
    /// and written back into the regions (edits outside every region become
    /// new `edit @ $XXXX` regions); any external region change (import,
    /// resize, remove, clear) is copied into the viewer's buffer in place, so
    /// cursor and scroll position survive.
    fn mem_view_ui(&mut self, ctx: &egui::Context) {
        if !self.mem_view_open {
            return;
        }
        if self.mem_view.is_none() {
            // Read-write with staged edits: the viewer collects changes and only
            // writes them back on "Apply changes". Apply/Cancel close the window;
            // the ✖ prompts about pending edits (all handled in the widget).
            let buf = self.image.full_buffer();
            let mut v = retroviewer::RetroViewer::new(buf.clone())
                .with_title("Memory viewer")
                .mode(retroviewer::EditMode::ReadWrite)
                .close_on_apply(true)
                .prompt_on_close(true);
            if let Some((lo, _)) = self.image.span() {
                v.execute_command(&format!("m ${lo:04X}"));
            }
            self.mem_view_snapshot = buf;
            self.mem_view = Some(v);
        }

        // Image -> viewer: absorb external changes (import / resize / remove /
        // clear) ONLY when the viewer has no pending edits, so a background
        // change never clobbers the user's staged work. `resync` keeps the
        // cursor and re-baselines, so a later Cancel reverts to the fresh buffer.
        let cur = self.image.full_buffer();
        if let Some(v) = self.mem_view.as_mut() {
            if !v.is_dirty() && v.data() != cur.as_slice() {
                v.resync(&cur);
                self.mem_view_snapshot = cur;
            }
        }

        // Draw the window; act on the frame's outcome.
        let mut open = self.mem_view_open;
        let action = self
            .mem_view
            .as_mut()
            .map(|v| v.show(ctx, &mut open))
            .unwrap_or(retroviewer::ViewerAction::None);
        self.mem_view_open = open;

        match action {
            retroviewer::ViewerAction::Applied => {
                // Commit the staged edits back into the regions.
                let new = self
                    .mem_view
                    .as_ref()
                    .map(|v| v.data().to_vec())
                    .unwrap_or_default();
                let regions_before = self.image.regions().len();
                let lines = self.image.apply_edits(&self.mem_view_snapshot, &new);
                for l in lines {
                    self.log(LogKind::Info, l);
                }
                self.clear_status();
                if self.image.regions().len() != regions_before {
                    // New patch regions can move the span: re-track the derived
                    // defaults, exactly like a region resize.
                    self.import_notes.clear();
                    self.restore_2d2e_auto = true;
                    self.refresh_start_default();
                }
                self.mem_view_snapshot = new;
            }
            // Cancelled: the viewer already reverted its buffer — nothing to write.
            retroviewer::ViewerAction::Cancelled | retroviewer::ViewerAction::None => {}
        }
    }

    fn apply_region_resize(&mut self, idx: usize, s: u16, e: u32) {
        let (os, oe, name) = {
            let r = &self.image.regions()[idx];
            (r.start(), r.end(), r.name.clone())
        };
        match self.image.resize_region(idx, s, e) {
            Ok(()) => {
                self.editing_region = None;
                // The span moved: re-track the auto start / $2D-$2E defaults and
                // drop import notes that now describe stale addresses.
                self.import_notes.clear();
                self.restore_2d2e_auto = true;
                self.refresh_start_default();
                self.clear_status();
                self.log(
                    LogKind::Good,
                    format!(
                        "Resized {name}: ${os:04X}-${:04X} → ${s:04X}-${:04X} ({} B)",
                        oe - 1,
                        e - 1,
                        e - s as u32,
                    ),
                );
            }
            Err(msg) => self.log(LogKind::Error, format!("Resize failed: {msg}")),
        }
    }

    /// Modal to edit the selected region's start/end address. A modal (rather
    /// than inline fields) gives the fields full, unconstrained width — the
    /// region grid squeezes a cell that holds two text boxes.
    fn edit_region_modal(&mut self, ui: &mut egui::Ui) {
        let Some(idx) = self.editing_region else {
            return;
        };
        if idx >= self.image.regions().len() {
            self.editing_region = None;
            return;
        }
        let name = self.image.regions()[idx].name.clone();
        let mut set: Option<(u16, u32)> = None;
        let mut close = false;
        let resp = egui::Modal::new(egui::Id::new("edit_region")).show(ui.ctx(), |ui| {
            ui.set_max_width(440.0);
            ui.heading("Edit region");
            ui.add_space(4.0);
            ui.monospace(&name);
            ui.add_space(10.0);
            egui::Grid::new("edit_region_grid")
                .num_columns(2)
                .spacing([12.0, 8.0])
                .show(ui, |ui| {
                    ui.label("Start address:");
                    ui.add(
                        egui::TextEdit::singleline(&mut self.edit_start_text)
                            .desired_width(120.0)
                            .hint_text("$0801"),
                    );
                    ui.end_row();
                    ui.label("End address:");
                    ui.add(
                        egui::TextEdit::singleline(&mut self.edit_end_text)
                            .desired_width(120.0)
                            .hint_text("$3FFF"),
                    );
                    ui.end_row();
                });
            ui.weak("End is the last byte (inclusive), matching the list display.");
            ui.add_space(8.0);
            let parsed = parse_region_range(&self.edit_start_text, &self.edit_end_text);
            match &parsed {
                Ok((s, e)) => {
                    ui.strong(format!("New size: {} B", e - *s as u32));
                }
                Err(e) => {
                    ui.colored_label(egui::Color32::from_rgb(0xE0, 0x80, 0x50), e);
                }
            }
            ui.add_space(6.0);
            ui.add(
                egui::Label::new(
                    egui::RichText::new(
                        "Shrinking trims bytes (a gap between chunks becomes zero-fill); \
                         growing zero-fills the new area.",
                    )
                    .weak(),
                )
                .wrap(),
            );
            ui.add_space(12.0);
            ui.horizontal(|ui| {
                if ui
                    .add_enabled(parsed.is_ok(), egui::Button::new("Set"))
                    .clicked()
                {
                    if let Ok((s, e)) = parsed {
                        set = Some((s, e));
                    }
                }
                if ui.button("Cancel").clicked() {
                    close = true;
                }
            });
        });
        if resp.should_close() {
            close = true;
        }
        if let Some((s, e)) = set {
            self.apply_region_resize(idx, s, e);
        } else if close {
            self.editing_region = None;
        }
    }

    /// Per-cruncher availability for the current image/placement: `None` when
    /// the format fits, `Some(reason)` when it does not (no room for its
    /// decoder/buffer, or the span rules out its only direction). This is the
    /// fast, stream-INDEPENDENT check; a format whose in-place feasibility
    /// depends on how well the data compresses stays "available" here and is
    /// resolved at build time (Compare skips one that then cannot fit).
    fn cruncher_availability(&self) -> Vec<Option<String>> {
        if self.image.is_empty() {
            return vec![None; self.crunchers.len()];
        }
        let Ok(placement) = self.placement() else {
            return vec![None; self.crunchers.len()];
        };
        (0..self.crunchers.len())
            .map(
                |i| match plan_preview(&self.image, self.crunchers[i].format, &placement) {
                    Ok(p) => p.unavailable,
                    Err(e) => Some(e),
                },
            )
            .collect()
    }

    /// Indices of crunchers usable for the current image (the ones Compare and
    /// the build actually attempt).
    fn available_crunchers(&self) -> Vec<usize> {
        let avail = self.cruncher_availability();
        (0..self.crunchers.len())
            .filter(|&i| avail[i].is_none())
            .collect()
    }

    fn cruncher_ui(&mut self, ui: &mut egui::Ui) {
        let avail = self.cruncher_availability();
        // If the current selection is not usable (e.g. after adding a .prg that
        // shrinks the free space), move to the first usable format.
        if avail.get(self.format_idx).is_some_and(|a| a.is_some()) {
            if let Some(first) = (0..self.crunchers.len()).find(|&i| avail[i].is_none()) {
                self.format_idx = first;
            }
        }
        ui.strong("Cruncher");
        ui.add_space(2.0);
        // Precompute the entry strings: `selectable_value` borrows
        // `self.format_idx` mutably, so we cannot also call `self` (immutably)
        // inside the combo closure.
        let selected_text = self.cruncher_entry(self.format_idx);
        let entries: Vec<(usize, String, Option<String>)> = (0..self.crunchers.len())
            .map(|i| (i, self.cruncher_entry(i), avail[i].clone()))
            .collect();
        ui.horizontal(|ui| {
            ui.label("Format:");
            egui::ComboBox::from_id_salt("cruncher")
                .width(520.0)
                .selected_text(selected_text)
                .show_ui(ui, |ui| {
                    // Show every format; grey out (disable) the ones that do not
                    // fit the current layout, with the reason on hover.
                    for (i, entry, unavail) in &entries {
                        match unavail {
                            None => {
                                ui.selectable_value(&mut self.format_idx, *i, entry.as_str());
                            }
                            Some(reason) => {
                                ui.add_enabled_ui(false, |ui| {
                                    ui.selectable_value(&mut self.format_idx, *i, entry.as_str());
                                })
                                .response
                                .on_hover_text(format!("Does not fit this layout: {reason}"));
                            }
                        }
                    }
                });
            let greyed = entries.iter().filter(|(_, _, u)| u.is_some()).count();
            if greyed > 0 {
                ui.weak(format!("({greyed} greyed — no room for this layout)"));
            }
        });
        ui.add_space(2.0);
        ui.add(
            egui::Label::new(
                egui::RichText::new(
                    "Packing and unpacking speed are 1–10 benchmark scores (10 = fastest). \
                 Decoder is the staged decruncher size for the current options.",
                )
                .weak(),
            )
            .wrap(),
        );

        ui.add_space(2.0);
        ui.horizontal(|ui| {
            ui.checkbox(&mut self.allow_illegal, "Use illegal opcodes")
                .on_hover_text(
                    "On: smallest/fastest decruncher (may use undocumented NMOS 6502 opcodes).\n\
                     Off: legal-only decoder — same compressed data, a slightly larger decoder \
                     that runs on CPUs/emulators without illegal opcodes.",
                );
            if !self.allow_illegal {
                ui.weak("legal-only decoder (no undocumented opcodes)");
            }
        });
    }

    /// Live plan block + placement overrides.
    fn placement_ui(&mut self, ui: &mut egui::Ui) {
        // Keep the $2D/$2E restore value tracking the $0801 segment end until
        // the user edits it by hand.
        if self.restore_2d2e_auto {
            if let Some(end) = self.basic_end_default() {
                self.restore_2d2e_text = format!("${end:04X}");
            }
        }
        let placement = self.placement().ok();
        // Compute the live plan once; reused for the plan block and the
        // per-field size estimate/warnings below.
        let plan = if self.image.is_empty() {
            None
        } else {
            placement.as_ref().and_then(|p| {
                plan_preview(&self.image, self.crunchers[self.format_idx].format, p).ok()
            })
        };
        // Whether the selected cruncher uses a scratch buffer at all. Prefer the
        // live plan (reflects the chosen direction/variant); fall back to the
        // format's standard routine when no image is loaded yet. Drives the
        // Buffer row's enabled state — a format with no buffer greys it out.
        let needs_buffer = match plan.as_ref().filter(|p| p.unavailable.is_none()) {
            Some(p) => p.scratch.is_some(),
            None => pick_routine(
                self.crunchers[self.format_idx].format,
                Direction::Forward,
                self.allow_illegal,
            )
            .map(|s| s.scratch.is_some())
            .unwrap_or(false),
        };
        ui.strong("Placement");
        if let Some(plan) = &plan {
            if let Some(msg) = &plan.unavailable {
                ui.add_space(2.0);
                ui.add(
                    egui::Label::new(
                        egui::RichText::new(msg).color(egui::Color32::from_rgb(0xE0, 0x80, 0x50)),
                    )
                    .wrap(),
                );
            } else {
                // Display the placement plan as wrapped text.
                let dir = match plan.direction {
                    Direction::Forward => "forward unpacking, packed data moved to $FFFF",
                    Direction::Backward => "backward unpacking, packed data moved below the span",
                };
                let decr = plan
                    .decruncher_at
                    .map(|a| format!("${a:04X}"))
                    .unwrap_or_else(|| "set manually!".into());
                ui.add_space(2.0);
                ui.indent("plan", |ui| {
                    ui.spacing_mut().item_spacing.y = 2.0;
                    plan_line(ui, "Direction", dir);
                    plan_line(
                        ui,
                        "Decrunch",
                        &format!("{} B staged @ {decr}", plan.staged_size),
                    );
                    if let Some((a, l)) = plan.scratch {
                        plan_line(ui, "Buffer", &format!("{l} B @ ${a:04X}"));
                    }
                    if self.mover_auto {
                        plan_line(
                            ui,
                            "Mover",
                            "folded into the decoder blob if the payload move needs to survive",
                        );
                    } else {
                        plan_line(
                            ui,
                            "Mover",
                            &format!(
                                "${:04X} (used only if the payload move needs relocation)",
                                plan.mover_at
                            ),
                        );
                    }
                });
            }
        }
        ui.add_space(4.0);

        ui.horizontal(|ui| {
            ui.label("Unpacking:");
            ui.selectable_value(&mut self.dir_choice, DirectionChoice::Auto, "Auto");
            ui.add_space(8.0);
            ui.selectable_value(&mut self.dir_choice, DirectionChoice::Forward, "Forward");
            ui.add_space(8.0);
            ui.selectable_value(&mut self.dir_choice, DirectionChoice::Backward, "Backward");
            if self.dir_choice != DirectionChoice::Auto {
                if let (Ok(p), false) = (self.placement(), self.image.is_empty()) {
                    if let Some((lo, hi)) = self.image.span() {
                        if let Err(e) = lazy_cruncher_workshop::sfx::resolve_direction((lo, hi), &p)
                        {
                            ui.colored_label(egui::Color32::from_rgb(0xE0, 0x80, 0x50), e);
                        }
                    }
                }
            }
        });
        // Per-crunch decoder tailoring — meaningful only for Exomizer (its
        // decoder has feature gates a stream may leave unused). Greyed out for
        // every other format so the Placement block keeps a stable layout.
        let is_exomizer = self.crunchers[self.format_idx].format == Format::Exomizer;
        ui.horizontal(|ui| {
            ui.add_enabled_ui(is_exomizer, |ui| {
                ui.label("Decoder:").on_hover_text(
                    "Per-crunch tailored Exomizer decoder: drops decoder sections this \
                     stream never uses (e.g. the literal-sequence handler). The compressed \
                     stream is unchanged — still standard `exomizer raw`.",
                );
                ui.selectable_value(&mut self.tailoring_choice, TailoringChoice::Auto, "Auto")
                    .on_hover_text("Build both; keep the smaller (never larger than standard).");
                ui.add_space(8.0);
                ui.selectable_value(
                    &mut self.tailoring_choice,
                    TailoringChoice::Standard,
                    "Standard",
                );
                ui.add_space(8.0);
                ui.selectable_value(
                    &mut self.tailoring_choice,
                    TailoringChoice::Tailored,
                    "Tailored",
                );
            });
            if !is_exomizer {
                ui.weak("Exomizer only");
            }
        });
        ui.horizontal(|ui| {
            ui.label("Decrunch routine:");
            ui.checkbox(&mut self.decr_auto, "auto");
            ui.add_enabled(
                !self.decr_auto,
                egui::TextEdit::singleline(&mut self.decr_text)
                    .desired_width(70.0)
                    .hint_text("$0100"),
            );
            if !self.decr_auto {
                if let Err(e) = parse_addr(&self.decr_text) {
                    ui.colored_label(egui::Color32::from_rgb(0xE0, 0x80, 0x50), e);
                }
            }
            // Live size estimate (includes the option epilogue) + a
            // non-blocking overflow warning. Auto placement never lands an
            // oversized blob at $0100, so this only fires on a manual override.
            if let Some(plan) = &plan {
                if plan.unavailable.is_none() {
                    if let Some(d) = plan.decruncher_at {
                        ui.weak(format!("~{} B staged", plan.staged_size));
                        if d == 0x0100 && plan.staged_size > STACK_PAGE_SLOT {
                            ui.colored_label(
                                egui::Color32::from_rgb(0xE0, 0x80, 0x50),
                                format!(
                                    "⚠ {} B spills past the $0100-$01DF slot into the stack",
                                    plan.staged_size
                                ),
                            );
                        }
                    }
                }
            }
        });
        ui.horizontal(|ui| {
            ui.label("Mover:");
            ui.checkbox(&mut self.mover_auto, "auto");
            ui.add_enabled(
                !self.mover_auto,
                egui::TextEdit::singleline(&mut self.mover_text)
                    .desired_width(70.0)
                    .hint_text("$02A7"),
            );
            if !self.mover_auto {
                if let Err(e) = parse_addr(&self.mover_text) {
                    ui.colored_label(egui::Color32::from_rgb(0xE0, 0x80, 0x50), e);
                }
            }
        });
        ui.horizontal(|ui| {
            // The whole row is greyed out for crunchers that use no scratch
            // buffer (zx02, tscrunch, …) — the choice is meaningless there.
            ui.add_enabled_ui(needs_buffer, |ui| {
                ui.label("Buffer:");
                ui.checkbox(&mut self.scratch_auto, "auto");
                ui.add_enabled(
                    needs_buffer && !self.scratch_auto,
                    egui::TextEdit::singleline(&mut self.scratch_text)
                        .desired_width(70.0)
                        .hint_text("$0334"),
                );
            });
            if needs_buffer && !self.scratch_auto {
                if let Err(e) = parse_addr(&self.scratch_text) {
                    ui.colored_label(egui::Color32::from_rgb(0xE0, 0x80, 0x50), e);
                }
            }
            if !needs_buffer {
                ui.weak("not needed for this cruncher");
            }
        });
        // While RUN drives $01/interrupts, its controls are locked to the
        // BASIC values.
        let run_locked = self.run_basic;
        ui.horizontal(|ui| {
            ui.label("Margin:");
            ui.add(egui::TextEdit::singleline(&mut self.margin_text).desired_width(40.0));
            ui.separator();
            ui.label("Clearance:")
                .on_hover_text(
                    "No-move safety gap (bytes): when the whole program image sits at least this \
                     far clear of the output, the payload is read in place — no move at all. The \
                     gap keeps the decompressed output from ever reaching the payload or decoder.",
                );
            ui.add(egui::TextEdit::singleline(&mut self.clearance_text).desired_width(40.0));
            if let Err(e) = parse_addr(&self.clearance_text) {
                ui.colored_label(egui::Color32::from_rgb(0xE0, 0x80, 0x50), e);
            }
            ui.separator();
            ui.label("$01 at JMP:");
            ui.add_enabled(
                !run_locked,
                egui::TextEdit::singleline(&mut self.bank_text)
                    .desired_width(50.0)
                    .hint_text("$30"),
            );
            if let Err(e) = parse_addr(&self.bank_text) {
                ui.colored_label(egui::Color32::from_rgb(0xE0, 0x80, 0x50), e);
            }
            ui.separator();
            ui.label("Interrupts:");
            ui.add_enabled_ui(!run_locked, |ui| {
                egui::ComboBox::from_id_salt("cli_before_jmp")
                    .selected_text(if self.cli_before_jmp { "CLI" } else { "SEI" })
                    .show_ui(ui, |ui| {
                        ui.selectable_value(&mut self.cli_before_jmp, false, "SEI (stay disabled)");
                        ui.selectable_value(&mut self.cli_before_jmp, true, "CLI (re-enable)");
                    });
            });
            if self.cli_before_jmp {
                if let Some(span) = self.image.span() {
                    if span_covers_irq_vector(span) {
                        ui.colored_label(
                            egui::Color32::from_rgb(0xE0, 0x80, 0x50),
                            "⚠ the IRQ vector ($0314/15) is inside the decompressed span — CLI may crash",
                        );
                    }
                }
            }
            if run_locked {
                ui.weak("(driven by RUN)");
            }
        });
        ui.horizontal(|ui| {
            ui.checkbox(&mut self.restore_2d2e, "Restore $2D/$2E");
            if self.restore_2d2e {
                let resp = ui.add(
                    egui::TextEdit::singleline(&mut self.restore_2d2e_text)
                        .desired_width(70.0)
                        .hint_text("$XXXX"),
                );
                if resp.changed() {
                    self.restore_2d2e_auto = false;
                }
                if let Err(e) = parse_addr(&self.restore_2d2e_text) {
                    ui.colored_label(egui::Color32::from_rgb(0xE0, 0x80, 0x50), e);
                }
                ui.weak("(end of the $0801 BASIC segment / VARTAB)");
            }
        });
        // The RUN convenience requires the $2D/$2E restore; drop it (and undo
        // its field overrides) if that is turned off.
        if !self.restore_2d2e && self.run_basic {
            self.run_basic = false;
            self.reset_after_run();
        }
    }

    fn start_addr_ui(&mut self, ui: &mut egui::Ui) {
        ui.strong("Start");
        ui.add_space(2.0);
        let run_locked = self.run_basic;
        ui.horizontal(|ui| {
            ui.label("Start after unpacking:");
            if run_locked {
                ui.monospace(format!("${RUN_BASIC_LOOP:04X}"));
                ui.weak("(BASIC interpreter loop)");
            } else {
                let auto_label = match self.default_start() {
                    Some(s) => format!("auto (${s:04X})"),
                    None => "auto".to_string(),
                };
                if ui.checkbox(&mut self.start_auto, auto_label).changed() {
                    self.refresh_start_default();
                }
                if !self.start_auto {
                    ui.add(
                        egui::TextEdit::singleline(&mut self.start_text)
                            .desired_width(70.0)
                            .hint_text("$080D"),
                    );
                    if let Err(e) = parse_addr(&self.start_text) {
                        ui.colored_label(egui::Color32::from_rgb(0xE0, 0x80, 0x50), e);
                    }
                }
            }
        });
        // "Force basic run": treat the decrunched program as BASIC. Standalone
        // — it turns on the $2D/$2E restore itself, so it needs no prerequisite.
        // Auto-enabled when a BASIC program is detected on load.
        ui.horizontal_wrapped(|ui| {
            let mut force = self.run_basic;
            if ui
                .checkbox(&mut force, "Force basic run")
                .on_hover_text(
                    "Run the decrunched program as BASIC: bank ROM in ($01=#$37), enable \
                     interrupts (CLI), restore $2D/$2E to the program end, JSR $A659 (CLR), \
                     then JMP to the BASIC run loop in ROM.",
                )
                .changed()
            {
                self.set_basic_run(force);
            }
            if self.run_basic {
                ui.weak(format!(
                    "→ $01=#${RUN_BASIC_BANK:02X}, CLI, restore $2D/$2E, JSR $A659 (CLR), JMP ${RUN_BASIC_LOOP:04X} (BASIC ROM run)"
                ));
            }
        });
    }

    fn output_ui(&mut self, ui: &mut egui::Ui) {
        ui.strong("Output file");
        ui.add_space(2.0);
        let mut do_export = false;
        ui.horizontal(|ui| {
            if ui.button("💾  Save as…").clicked() {
                let mut dialog = rfd::FileDialog::new().add_filter("C64 program", &["prg"]);
                if let Some(p) = &self.output_path {
                    if let Some(dir) = p.parent() {
                        dialog = dialog.set_directory(dir);
                    }
                    dialog = dialog.set_file_name(file_name(p));
                }
                if let Some(path) = dialog.save_file() {
                    self.output_path = Some(path);
                }
            }
            match &self.output_path {
                Some(p) => ui.monospace(file_name(p)),
                None => ui.weak("no output file selected"),
            };
            // Right corner: export ONLY the compressed data (no SFX wrapper).
            ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                let enabled = !self.busy() && !self.image.is_empty();
                if ui
                    .add_enabled(enabled, egui::Button::new("⬇  Export packed stream…"))
                    .on_hover_text(
                        "Save ONLY the compressed data (the packed stream, no self-extracting \
                         wrapper) as a .bin file — default name “…-crunched.bin”.",
                    )
                    .clicked()
                {
                    do_export = true;
                }
            });
        });
        if do_export {
            let ctx = ui.ctx().clone();
            self.start_export_stream(&ctx);
        }
    }

    /// Rich "current result" card for the last crunch (the running history
    /// lives in the log panel).
    fn result_ui(&mut self, ui: &mut egui::Ui) {
        match &self.status {
            Status::Idle | Status::Working => {}
            Status::Failed(msg) => {
                ui.colored_label(
                    egui::Color32::from_rgb(0xE0, 0x50, 0x50),
                    format!("✖  {msg}"),
                );
            }
            Status::Done(r) => {
                let ratio = if r.input_len > 0 {
                    r.output_len as f64 / r.input_len as f64 * 100.0
                } else {
                    0.0
                };
                egui::Frame::group(ui.style()).show(ui, |ui| {
                    ui.colored_label(
                        egui::Color32::from_rgb(0x50, 0xC0, 0x60),
                        "✔  Done — self-extracting .prg written",
                    );
                    egui::Grid::new("report")
                        .num_columns(2)
                        .spacing([16.0, 4.0])
                        .show(ui, |ui| {
                            ui.label("In (sum of .prg):");
                            ui.monospace(format!("{} bytes", r.input_len));
                            ui.end_row();
                            ui.label("Out:");
                            ui.monospace(format!(
                                "{} bytes  ({ratio:.1} %, {} saved)",
                                r.output_len,
                                r.input_len as i64 - r.output_len as i64
                            ));
                            ui.end_row();
                            ui.label("Span:");
                            ui.monospace(format!("${:04X}-${:04X}", r.span.0, r.span.1 - 1));
                            ui.end_row();
                            ui.label("Unpacking:");
                            ui.monospace(format!(
                                "{}, packed data @ ${:04X} ({} B)",
                                r.direction, r.packed_at, r.stream_len
                            ));
                            ui.end_row();
                            ui.label("Decrunch routine:");
                            if r.decoder_tailored {
                                ui.monospace(format!(
                                    "${:04X}  ({} B code, tailored −{} B)",
                                    r.decruncher_at, r.decoder_bytes, r.decoder_saved
                                ))
                                .on_hover_text(
                                    "Per-crunch tailored decoder: sections this stream never uses \
                                 were removed. The compressed stream is unchanged.",
                                );
                            } else {
                                ui.monospace(format!(
                                    "${:04X}  ({} B code)",
                                    r.decruncher_at, r.decoder_bytes
                                ));
                            }
                            ui.end_row();
                            ui.label("Buffer:");
                            ui.monospace(match r.scratch {
                                Some((a, l)) => format!("${a:04X} ({l} B)"),
                                None => "not needed".to_string(),
                            });
                            ui.end_row();
                            ui.label("Mover:");
                            ui.monospace(if !r.payload_moved {
                                "not needed — payload read in place".to_string()
                            } else {
                                match (r.mover_at, r.mover_folded) {
                                    (Some(a), _) => format!("${a:04X}"),
                                    (None, true) => "folded into the decoder blob".to_string(),
                                    (None, false) => "inline (from the main program)".to_string(),
                                }
                            });
                            ui.end_row();
                            ui.label("File:");
                            ui.monospace(file_name(&r.output_path));
                            ui.end_row();
                        });
                    for w in &r.warnings {
                        ui.colored_label(
                            egui::Color32::from_rgb(0xD0, 0xB0, 0x50),
                            format!("⚠  {w}"),
                        );
                    }
                });
            }
        }
    }

    /// The bottom action bar: Crunch! and Compare Crunchers plus live progress.
    fn actions_ui(&mut self, ui: &mut egui::Ui) {
        let working = matches!(self.status, Status::Working);
        let comparing = self.compare.is_some();
        let can_run = !self.image.is_empty()
            && self.output_path.is_some()
            && !working
            && self.import_rx.is_none()
            && !comparing;
        let can_compare = !self.image.is_empty() && !self.busy();
        ui.add_space(4.0);
        ui.horizontal(|ui| {
            if ui
                .add_enabled(
                    can_run,
                    egui::Button::new("⚙  Crunch!").min_size(egui::vec2(120.0, 28.0)),
                )
                .clicked()
            {
                self.start_crunch(ui.ctx());
            }
            if ui
                .add_enabled(
                    can_compare,
                    egui::Button::new("📊  Compare Crunchers").min_size(egui::vec2(160.0, 28.0)),
                )
                .on_hover_text(
                    "Pack with every fitting cruncher in parallel and rank them in the log \
                     (no files are written).",
                )
                .clicked()
            {
                self.start_compare(ui.ctx());
            }
            if working {
                ui.spinner();
                let secs = self.started.map(|t| t.elapsed().as_secs()).unwrap_or(0);
                ui.label(format!("Compressing… {secs} s"));
                ui.weak("(large files can take over a minute on slow formats)");
            }
            if let Some(run) = &self.compare {
                ui.spinner();
                ui.label(format!("Comparing… {}/{}", run.done, run.total));
                ui.weak(format!("{:.0} s", run.started.elapsed().as_secs_f64()));
            }
        });
        ui.add_space(4.0);
    }

    /// The scrollable status/history log at the bottom of the window.
    fn log_ui(&mut self, ui: &mut egui::Ui) {
        ui.add_space(4.0);
        ui.horizontal(|ui| {
            ui.strong("Status");
            ui.weak("— what happened, newest at the bottom");
            ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                if ui
                    .add_enabled(!self.log.is_empty(), egui::Button::new("Clear log").small())
                    .clicked()
                {
                    self.log.clear();
                }
            });
        });
        ui.separator();
        egui::ScrollArea::vertical()
            .auto_shrink([false, false])
            .stick_to_bottom(true)
            .show(ui, |ui| {
                ui.spacing_mut().item_spacing.y = 1.0;
                if self.log.is_empty() {
                    ui.weak(
                        "No activity yet. Add a .prg and press Crunch!, or Compare Crunchers to \
                         rank every format.",
                    );
                    return;
                }
                let dim = ui.visuals().weak_text_color();
                for line in &self.log {
                    let color = match line.kind {
                        LogKind::Head => ui.visuals().strong_text_color(),
                        LogKind::Info => ui.visuals().text_color(),
                        LogKind::Good => egui::Color32::from_rgb(0x50, 0xC0, 0x60),
                        LogKind::Warn => egui::Color32::from_rgb(0xD8, 0xB0, 0x50),
                        LogKind::Error => egui::Color32::from_rgb(0xE0, 0x60, 0x60),
                        LogKind::Muted => dim,
                    };
                    let mut rich = egui::RichText::new(&line.text)
                        .monospace()
                        .color(color)
                        .size(12.5);
                    if line.kind == LogKind::Head {
                        rich = rich.strong();
                    }
                    ui.add(egui::Label::new(rich).wrap());
                }
            });
    }
}

/// One aligned "Label: value" row inside the plan block.
fn plan_line(ui: &mut egui::Ui, label: &str, value: &str) {
    ui.horizontal_wrapped(|ui| {
        ui.weak(format!("{label}:"));
        ui.add(egui::Label::new(egui::RichText::new(value).weak()).wrap());
    });
}

fn file_name(p: &std::path::Path) -> String {
    p.file_name()
        .map(|s| s.to_string_lossy().into_owned())
        .unwrap_or_else(|| p.to_string_lossy().into_owned())
}

/// Parse a region's start + inclusive-end address fields into
/// `(start, end_exclusive)`. The end field is the last byte (matching the
/// `$start-$end` display), so the exclusive end is one past it (up to `$10000`).
fn parse_region_range(start_s: &str, end_s: &str) -> Result<(u16, u32), String> {
    let start = parse_addr(start_s).map_err(|e| format!("start: {e}"))?;
    let end_incl = parse_addr(end_s).map_err(|e| format!("end: {e}"))?;
    if end_incl < start {
        return Err("end is before start".into());
    }
    Ok((start, end_incl as u32 + 1))
}
