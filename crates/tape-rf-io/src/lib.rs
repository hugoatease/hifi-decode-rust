//! RF/audio sample input, FLAC decoding, frequency-argument parsing, and
//! stdio duplication shared between
//! `tape-decode-cli` (video) and `hifi-decode-cli` (audio). Nothing here is
//! video- or audio-specific: it streams fixed-width or FLAC-encoded samples
//! to `f32` and hides the file-vs-pipe seek distinction behind one trait.

mod flac;
mod frequency;
mod os;
mod reader;
mod tracing_init;

pub use frequency::parse_frequency_mhz;
pub use os::{stderr_file, stdin_file, stdout_file};
pub use reader::{open_source, DecodeReader, SampleFormat, SampleSource};
pub use tracing_init::init_tracing;
