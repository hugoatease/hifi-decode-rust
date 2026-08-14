//! Audio output. `.wav` gets 16-bit PCM, anything else gets 24-bit FLAC —
//! the same two shapes `vhsdecode.hifi` writes (`main.py:998-1024`), and
//! written through the same library it uses, so the resulting files match
//! byte for byte rather than only sample for sample. See
//! `crate::sndfile` for what that binding covers and why it replaced a
//! pure-Rust encoder.
//!
//! Chunks stream straight to disk as they arrive, matching the input side
//! (`crate::stream`) — an hour-long tape's decoded audio is a few hundred
//! MB, small next to the RF, but there is no reason to hold it.
//!
//! Nothing here converts samples to integers: libsndfile does that
//! internally from the `f32` buffers it is handed, exactly as Python's
//! `buffer_write(..., dtype="float32")` does. That is deliberate — the
//! scale factor (a power of two, not the largest representable
//! magnitude), the half-to-even rounding and the clipping bounds are all
//! easy to get subtly wrong, and each costs an LSB on a large share of
//! samples.

use std::path::Path;

use anyhow::Result;

use crate::sndfile::SndFileSink;

/// One output file, fed one chunk at a time.
pub struct AudioSink {
    sink: SndFileSink,
    channels: usize,
    /// Reused across chunks so a long decode doesn't reallocate an
    /// interleave buffer per block.
    interleaved: Vec<f32>,
}

impl AudioSink {
    pub fn create(path: &Path, sample_rate: u32, channels: u16, is_wav: bool) -> Result<Self> {
        let channels = channels as usize;
        Ok(AudioSink {
            sink: SndFileSink::create(path, sample_rate, channels, is_wav)?,
            channels,
            interleaved: Vec::new(),
        })
    }

    pub fn write_chunk(&mut self, left: &[f32], right: Option<&[f32]>) -> Result<()> {
        self.interleaved.clear();
        match right {
            Some(right) => {
                assert_eq!(left.len(), right.len(), "channel length mismatch");
                assert_eq!(self.channels, 2, "stereo chunk written to a mono sink");
                self.interleaved.reserve(left.len() * 2);
                for (&l, &r) in left.iter().zip(right) {
                    self.interleaved.push(l);
                    self.interleaved.push(r);
                }
            }
            None => {
                assert_eq!(self.channels, 1, "mono chunk written to a stereo sink");
                self.interleaved.extend_from_slice(left);
            }
        }
        self.sink.write_frames(&self.interleaved)
    }

    pub fn finish(self) -> Result<()> {
        self.sink.finish()
    }
}
