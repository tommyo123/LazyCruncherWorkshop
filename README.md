# LazyCruncher Workshop

LazyCruncher Workshop is a desktop application for creating self-extracting
Commodore 64 `.prg` files. Load one or more PRG files, select a compression
format, adjust the placement when required, and write an autostarting output
file.

## Features

- Creates self-extracting C64 PRG files from one or more input regions.
- Supports the compression formats provided by lzan-library.
- Imports supported crunched PRG files through UniDecrunch.
- Exports the current memory image as an uncompressed PRG file.
- Shows the generated memory layout and provides a memory viewer.
- Offers legal-only decoder variants where available.
- Can select the fastest decruncher for a format instead of the balanced one.

## Requirements

- Rust, installed through [rustup](https://rustup.rs/).

The companion libraries are fetched directly from their GitHub repositories by
Cargo; no adjacent local checkouts are required.

## Build and run

```powershell
cargo run --release --locked
```

The application opens a file picker for adding PRG files. The generated output
is a self-extracting PRG that starts at `$0801`.

## Test

```powershell
cargo test --locked
```

## Project layout

- `src/` contains the application and shared packing logic.
- `icons/` contains the application icons.
- `build.rs` embeds the Windows executable icon.

## License

Licensed under the [MIT License](LICENSE).
