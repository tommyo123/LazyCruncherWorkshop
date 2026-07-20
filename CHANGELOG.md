# Changelog

Notable changes, newest first.

## [0.9.1] - 2026-07-20

### Added

- Option to select the fastest decruncher for a format instead of the balanced
  one. The compressed data is the same either way; only the decruncher differs.
- BoltLZ, the byte oriented format added in lzan-c64 1.0.1.

### Changed

- Unpacking speed scores are measured separately for forward and backward
  layouts, so a backward decode is scored by its own decruncher instead of the
  forward number. All scores share one reference and use one decimal.
- The compare list and the crunch report show the unpacking score next to the
  decruncher size.
- The default window is wider so a compare row fits on one line.

### Fixed

- The decruncher size shown in the compare list and the report was taken from
  the balanced decruncher even when the fastest one was selected. It now comes
  from the build result, so it always matches the generated file.

## [0.9.0] - 2026-07-19

First public release.
