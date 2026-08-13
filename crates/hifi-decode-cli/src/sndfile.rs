//! Minimal FFI binding to libsndfile, sized to exactly the calls Python's
//! `soundfile` makes when `hifi-decode` opens its output
//! (`main.py:998-1024`'s `as_outputfile`).
//!
//! Writing through the same library the reference does is what makes the
//! output file byte-identical rather than merely sample-identical: the
//! float-to-integer conversion, the clipping rule, libFLAC's encoder
//! settings, the metadata blocks and the stream MD5 all come from
//! libsndfile itself, so none of them has to be re-derived and kept in
//! sync by hand.
//!
//! It also sidesteps a real bug. The previous pure-Rust encoder
//! (`flacenc`) estimates Rice coding cost in a `u32` that it only
//! saturates once per 16 residuals; on loud 24-bit audio those sixteen
//! additions overflow, the estimate wraps to a small value, and the
//! verbatim fallback that should bound a frame at raw-PCM size never
//! fires. Output then grows *with amplitude* — measured here at 31x the
//! reference size at unity gain doubled, and 1521x at 8x gain, against
//! input whose raw PCM is 3.4MB. The overflow is a documented `TODO`
//! upstream and is still present in the latest release (0.5.1).

use std::ffi::{c_char, c_int, c_void, CString};
use std::path::Path;

use anyhow::{bail, Context as _, Result};

#[repr(C)]
#[derive(Default)]
struct SfInfo {
    frames: i64,
    samplerate: c_int,
    channels: c_int,
    format: c_int,
    sections: c_int,
    seekable: c_int,
}

extern "C" {
    fn sf_open(path: *const c_char, mode: c_int, info: *mut SfInfo) -> *mut c_void;
    fn sf_command(file: *mut c_void, command: c_int, data: *mut c_void, datasize: c_int) -> c_int;
    fn sf_writef_float(file: *mut c_void, ptr: *const f32, frames: i64) -> i64;
    fn sf_close(file: *mut c_void) -> c_int;
    fn sf_strerror(file: *mut c_void) -> *const c_char;
}

const SFM_WRITE: c_int = 0x20;
const SF_TRUE: c_int = 1;
const SF_FORMAT_WAV: c_int = 0x010000;
const SF_FORMAT_FLAC: c_int = 0x170000;
const SF_FORMAT_PCM_16: c_int = 0x0002;
const SF_FORMAT_PCM_24: c_int = 0x0003;
const SFC_SET_CLIPPING: c_int = 0x10C0;
const SFC_SET_COMPRESSION_LEVEL: c_int = 0x1301;

/// `compression_level=1.0`, the value `as_outputfile` passes for FLAC
/// output. libsndfile inverts it on the way to libFLAC
/// (`8 * (1.0 - level)`), so this is not "maximum effort" — but matching
/// the reference matters more than the setting's own merits.
const FLAC_COMPRESSION_LEVEL: f64 = 1.0;

/// A libsndfile output file, opened the way `soundfile.SoundFile` opens
/// one for writing.
pub struct SndFileSink {
    file: *mut c_void,
    channels: usize,
}

impl SndFileSink {
    /// `is_wav` selects Python's two output shapes: `.wav` gets
    /// `WAV`/`PCM_16`, everything else `FLAC`/`PCM_24` with the
    /// compression level set (`as_outputfile`).
    pub fn create(path: &Path, sample_rate: u32, channels: usize, is_wav: bool) -> Result<Self> {
        let format = if is_wav {
            SF_FORMAT_WAV | SF_FORMAT_PCM_16
        } else {
            SF_FORMAT_FLAC | SF_FORMAT_PCM_24
        };
        let mut info = SfInfo {
            samplerate: sample_rate as c_int,
            channels: channels as c_int,
            format,
            ..SfInfo::default()
        };

        let c_path = CString::new(path.as_os_str().as_encoded_bytes())
            .with_context(|| format!("output path is not representable as a C string: {}", path.display()))?;

        let file = unsafe { sf_open(c_path.as_ptr(), SFM_WRITE, &mut info) };
        if file.is_null() {
            bail!(
                "failed to open {} for writing: {}",
                path.display(),
                last_error(std::ptr::null_mut())
            );
        }

        let sink = SndFileSink { file, channels };

        // `soundfile` turns clipping on for every file it opens
        // (`SoundFile.__init__`), so out-of-range samples saturate instead
        // of wrapping. hifi-decode routinely produces those — a decode of
        // a real tape reports peaks well above 100% — which makes this
        // load-bearing rather than defensive.
        unsafe { sf_command(sink.file, SFC_SET_CLIPPING, std::ptr::null_mut(), SF_TRUE) };

        if !is_wav {
            let mut level = FLAC_COMPRESSION_LEVEL;
            let ok = unsafe {
                sf_command(
                    sink.file,
                    SFC_SET_COMPRESSION_LEVEL,
                    &mut level as *mut f64 as *mut c_void,
                    std::mem::size_of::<f64>() as c_int,
                )
            };
            if ok != SF_TRUE {
                bail!("failed to set FLAC compression level: {}", last_error(sink.file));
            }
        }

        Ok(sink)
    }

    /// Writes one span of interleaved frames. `frames` is a frame count,
    /// not a sample count — `sf_writef_float` interleaves by channel
    /// itself.
    pub fn write_frames(&mut self, interleaved: &[f32]) -> Result<()> {
        debug_assert_eq!(interleaved.len() % self.channels, 0);
        let frames = (interleaved.len() / self.channels) as i64;
        if frames == 0 {
            return Ok(());
        }
        let written = unsafe { sf_writef_float(self.file, interleaved.as_ptr(), frames) };
        if written != frames {
            bail!(
                "short write to output: {written} of {frames} frames ({})",
                last_error(self.file)
            );
        }
        Ok(())
    }

    /// Closes the file, flushing libFLAC's remaining frames and patching
    /// the header. Distinct from `Drop` so a failure here is reported
    /// rather than swallowed — a truncated output is worth an error.
    pub fn finish(mut self) -> Result<()> {
        let file = std::mem::replace(&mut self.file, std::ptr::null_mut());
        let err = unsafe { sf_close(file) };
        if err != 0 {
            bail!("failed to close output file (libsndfile error {err})");
        }
        Ok(())
    }
}

impl Drop for SndFileSink {
    fn drop(&mut self) {
        if !self.file.is_null() {
            unsafe { sf_close(self.file) };
        }
    }
}

// The handle is only ever touched through `&mut self`, and libsndfile
// serializes nothing internally — moving one between threads is fine as
// long as it isn't shared, which the type system already enforces.
unsafe impl Send for SndFileSink {}

fn last_error(file: *mut c_void) -> String {
    let message = unsafe { sf_strerror(file) };
    if message.is_null() {
        return "(no error message)".to_string();
    }
    unsafe { std::ffi::CStr::from_ptr(message) }
        .to_string_lossy()
        .into_owned()
}
