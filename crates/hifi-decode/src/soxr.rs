//! Minimal FFI binding to libsoxr, sized to exactly the calls Python's
//! `soxr.ResampleStream` makes (`HiFiDecode.py:1139-1143`, `:2195-2196`).
//!
//! This exists instead of a pure-Rust resampler because bit-exact parity
//! with `vhsdecode.hifi` is the goal, and no reimplementation reproduces
//! libsoxr's output sample-for-sample — the resampled signal is the single
//! largest source of divergence in the whole chain. Linking the same C
//! library Python's `soxr` wheel links makes the two byte-identical, which
//! is validated end-to-end by the fixture parity tests.
//!
//! Only the stream API is bound (`soxr_create`/`soxr_process`/
//! `soxr_delete`), matching how `ResampleStream` drives it: one
//! `resample_chunk(x, last=True)` per block, i.e. push the whole chunk then
//! drain the filter's tail. `python-soxr` builds its quality spec as
//! `soxr_quality_spec(recipe, 0)` and its I/O spec as float32-interleaved
//! in and out, so those are the only shapes bound here.

use std::os::raw::{c_char, c_uint, c_ulong, c_void};

#[repr(C)]
#[derive(Clone, Copy)]
struct IoSpec {
    itype: u32,
    otype: u32,
    scale: f64,
    e: *mut c_void,
    flags: c_ulong,
}

#[repr(C)]
#[derive(Clone, Copy)]
struct QualitySpec {
    precision: f64,
    phase_response: f64,
    passband_end: f64,
    stopband_begin: f64,
    e: *mut c_void,
    flags: c_ulong,
}

#[repr(C)]
#[derive(Clone, Copy)]
struct RuntimeSpec {
    log2_min_dft_size: c_uint,
    log2_large_dft_size: c_uint,
    coef_size_kbytes: c_uint,
    num_threads: c_uint,
    e: *mut c_void,
    flags: c_ulong,
}

extern "C" {
    fn soxr_io_spec(itype: u32, otype: u32) -> IoSpec;
    fn soxr_quality_spec(recipe: c_ulong, flags: c_ulong) -> QualitySpec;
    fn soxr_runtime_spec(num_threads: c_uint) -> RuntimeSpec;
    fn soxr_create(
        input_rate: f64,
        output_rate: f64,
        num_channels: c_uint,
        error: *mut *const c_char,
        io: *const IoSpec,
        quality: *const QualitySpec,
        runtime: *const RuntimeSpec,
    ) -> *mut c_void;
    fn soxr_process(
        resampler: *mut c_void,
        input: *const c_void,
        ilen: usize,
        idone: *mut usize,
        output: *mut c_void,
        olen: usize,
        odone: *mut usize,
    ) -> *const c_char;
    fn soxr_delete(resampler: *mut c_void);
}

/// `SOXR_FLOAT32_I` — 32-bit float, interleaved. `python-soxr` maps
/// `np.float32` to this, and `REAL_DTYPE` is float32 throughout
/// `HiFiDecode`.
const SOXR_FLOAT32_I: u32 = 0;

/// libsoxr quality recipes (`soxr.h:284-294`). `python-soxr`'s string
/// quality names map onto these one-for-one.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SoxrRecipe {
    Lq,
    Mq,
    Hq,
    Vhq,
}

impl SoxrRecipe {
    fn value(self) -> c_ulong {
        match self {
            SoxrRecipe::Lq => 1,
            SoxrRecipe::Mq => 2,
            SoxrRecipe::Hq => 4,  // SOXR_20_BITQ
            SoxrRecipe::Vhq => 6, // SOXR_28_BITQ
        }
    }
}

/// Resamples `input` in one shot at `out_rate / in_rate`, flushing the
/// filter tail — the exact equivalent of `ResampleStream.resample_chunk(x,
/// last=True)` followed by `.clear()`.
///
/// A fresh resampler is created per call rather than one being kept alive
/// and reset: Python clears its stream after every single chunk, so every
/// chunk it processes starts from clean state, which is what a fresh
/// `soxr_create` gives. Nothing carries across blocks either way.
///
/// `in_rate`/`out_rate` are passed through to libsoxr as-is; only their
/// ratio matters, and callers are expected to have already reproduced
/// Python's exact `Fraction`-derived pair (see `exact_ratio_pair`).
pub fn resample_chunk(input: &[f32], in_rate: f64, out_rate: f64, recipe: SoxrRecipe) -> Vec<f32> {
    if input.is_empty() {
        return Vec::new();
    }

    unsafe {
        let io = soxr_io_spec(SOXR_FLOAT32_I, SOXR_FLOAT32_I);
        let quality = soxr_quality_spec(recipe.value(), 0);
        // `python-soxr` leaves threading at libsoxr's default of one
        // thread; block-level parallelism is this port's own concern and
        // multi-threading inside a resample would change nothing about the
        // result but is not what the reference does.
        let runtime = soxr_runtime_spec(1);

        let mut error: *const c_char = std::ptr::null();
        let resampler = soxr_create(in_rate, out_rate, 1, &mut error, &io, &quality, &runtime);
        assert!(
            !resampler.is_null(),
            "soxr_create failed: {}",
            error_string(error)
        );

        // Generous headroom over the nominal output length: libsoxr never
        // emits more than the ratio implies plus the filter tail, and a
        // too-small buffer would silently truncate rather than error.
        let nominal = (input.len() as f64 * (out_rate / in_rate)).ceil() as usize;
        let mut output = vec![0.0f32; nominal + 8192];

        let mut idone = 0usize;
        let mut odone = 0usize;
        let err = soxr_process(
            resampler,
            input.as_ptr() as *const c_void,
            input.len(),
            &mut idone,
            output.as_mut_ptr() as *mut c_void,
            output.len(),
            &mut odone,
        );
        assert!(err.is_null(), "soxr_process failed: {}", error_string(err));
        debug_assert_eq!(idone, input.len(), "soxr did not consume the whole chunk");

        // Drain the filter tail (`last=True`): feed end-of-input until it
        // stops producing.
        let mut total = odone;
        loop {
            let mut drained = 0usize;
            let err = soxr_process(
                resampler,
                std::ptr::null(),
                0,
                std::ptr::null_mut(),
                output.as_mut_ptr().add(total) as *mut c_void,
                output.len() - total,
                &mut drained,
            );
            assert!(err.is_null(), "soxr_process flush failed: {}", error_string(err));
            if drained == 0 {
                break;
            }
            total += drained;
        }

        soxr_delete(resampler);
        output.truncate(total);
        output
    }
}

fn error_string(err: *const c_char) -> String {
    if err.is_null() {
        return "(no error message)".to_string();
    }
    unsafe { std::ffi::CStr::from_ptr(err) }
        .to_string_lossy()
        .into_owned()
}

/// Reproduces Python's `Fraction(rate_a / rate_b)` numerator/denominator
/// pair (`HiFiDecode.getResamplingRatios`), which is what gets handed to
/// `ResampleStream` as its (in_rate, out_rate).
///
/// `Fraction(float)` is the *exact* rational value of the double, i.e.
/// `mantissa / 2^k`, so this just scales by two until the value is
/// integral. Both halves come back exactly representable as `f64` — the
/// numerator is a double's mantissa (< 2^53) and the denominator a power of
/// two — so libsoxr receives the identical pair Python's does, rather than
/// a ratio that merely rounds to the same value.
pub fn exact_ratio_pair(ratio: f64) -> (f64, f64) {
    assert!(ratio.is_finite() && ratio > 0.0, "ratio must be finite and positive");
    let mut numerator = ratio;
    let mut denominator = 1.0f64;
    while numerator.fract() != 0.0 {
        numerator *= 2.0;
        denominator *= 2.0;
    }
    (numerator, denominator)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The pairs Python actually derives for this decoder's two resampling
    /// stages, checked against values computed with `fractions.Fraction`.
    #[test]
    fn exact_ratio_pair_matches_python_fraction() {
        // Fraction(192000 / 28636363)
        let (num, den) = exact_ratio_pair(192000.0 / 28636363.0);
        assert_eq!(num, 1932516088761993.0);
        assert_eq!(den, 288230376151711744.0);
        // The pair must reproduce the original double exactly, not merely
        // approximate it.
        assert_eq!(num / den, 192000.0 / 28636363.0);

        // Fraction(48000 / 192000) == 1/4
        assert_eq!(exact_ratio_pair(48000.0 / 192000.0), (1.0, 4.0));
    }

    #[test]
    fn resampling_a_quarter_rate_halves_length_as_expected() {
        let input: Vec<f32> = (0..4000)
            .map(|i| (i as f32 * 0.01).sin())
            .collect();
        let (num, den) = exact_ratio_pair(0.25);
        let out = resample_chunk(&input, den, num, SoxrRecipe::Vhq);
        assert!(
            out.len().abs_diff(1000) <= 2,
            "expected ~1000 output samples, got {}",
            out.len()
        );
    }

    #[test]
    fn empty_input_produces_empty_output() {
        assert!(resample_chunk(&[], 4.0, 1.0, SoxrRecipe::Vhq).is_empty());
    }
}
