//! Central configuration values the rest of the app keys off. The version is
//! sourced from `Cargo.toml` so the window title, the About dialog, and the
//! release/installer build all stay in sync with the package version. Bump the
//! version in `Cargo.toml`; everything else reads it from here.

/// Application version, taken from `Cargo.toml`.
pub const VERSION: &str = env!("CARGO_PKG_VERSION");

/// Application name, shown in the window title and the About dialog.
pub const APP_NAME: &str = "LazyCruncher Workshop";
