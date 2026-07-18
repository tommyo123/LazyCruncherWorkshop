//! Import of already-crunched `.prg` files. `unidecrunch` detects the
//! cruncher by signature, runs the depacker in its 6510 emulator, and the
//! unpacked program is handed back so it can join the region list as a chunk.

use unidecrunch::UniDecrunch;

/// One unpacked program, ready to add to the memory image.
pub struct ImportedPart {
    /// Cruncher name reported by the detector (a chain like
    /// "Exomizer \u{2192} TSCrunch" for double-crunched files).
    pub cruncher: String,
    /// First address of the unpacked program.
    pub start: u16,
    /// Last address (inclusive).
    pub end: u16,
    /// Where the original depacker jumped to launch the program — the real
    /// entry point, useful as a start-address default when the payload has
    /// no BASIC `SYS` stub.
    pub jump_start: u16,
    /// The unpacked program as a PRG image (load address + data).
    pub prg: Vec<u8>,
}

/// Detect and unpack a crunched PRG with a fresh detector instance.
pub fn import_crunched(bytes: &[u8]) -> Result<ImportedPart, String> {
    import_crunched_with(&UniDecrunch::new(), bytes)
}

/// Probe whether an already-unpacked program `prg` is ITSELF another crunched
/// layer, unpacking exactly one more level (signature-based only, so an unpacked
/// program's incidental cruncher-like code does not false-match). `entry` is the
/// previous layer's run address (the entry fallback when a rebuilt BASIC line
/// cannot be parsed). Returns the deeper part only when a real inner layer
/// unpacks to a *different* program — the building block for cascade unpacking.
pub fn probe_deeper_layer(
    ud: &UniDecrunch,
    prg: &[u8],
    entry: Option<u16>,
) -> Option<ImportedPart> {
    match ud.decrunch_layer(prg, entry) {
        Ok(Some(d)) if d.prg != prg => Some(ImportedPart {
            cruncher: d.cruncher,
            start: d.start,
            end: d.end,
            jump_start: d.jump_start,
            prg: d.prg,
        }),
        _ => None,
    }
}

/// Detect and unpack a crunched PRG. Errors distinguish "recognized but the
/// depacker did not finish" from "not recognized as a crunched file". The
/// signature check ignores the catch-all definition (which matches any program
/// with a `SYS` line), so recognition means a real byte signature matched
/// regardless of config order.
pub fn import_crunched_with(ud: &UniDecrunch, bytes: &[u8]) -> Result<ImportedPart, String> {
    match ud.decrunch_bytes(bytes)? {
        Some(d) => Ok(ImportedPart {
            cruncher: d.cruncher,
            start: d.start,
            end: d.end,
            jump_start: d.jump_start,
            prg: d.prg,
        }),
        None => {
            let signature = ud.detect_signature_bytes(bytes).ok().flatten();
            Err(match signature {
                Some(det) => format!(
                    "recognized as \"{}\" but the depacker did not finish.",
                    det.name()
                ),
                // A signature can also match while the BASIC entry (SYS) line
                // cannot be parsed, which the detector reports the same as no
                // match — hence the hedge on the entry line.
                None => "not recognized as a crunched file \
                         (or its BASIC entry line could not be parsed)."
                    .to_string(),
            })
        }
    }
}
