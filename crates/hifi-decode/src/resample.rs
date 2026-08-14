//! Sample-rate conversion. Ports Python's `soxr.ResampleStream`
//! (`HiFiDecode.py:1139-1143`) by binding the same C library it does — see
//! `crate::soxr` for why a pure-Rust resampler can't stand in here if the
//! goal is byte-identical output.
//!
//! Matches Python's per-block usage exactly in one respect that matters a
//! lot for correctness: `resample_chunk(audio, True)` immediately followed
//! by `.clear()` (`HiFiDecode.py:2195-2196`, `:2229-2230`) means **no
//! resampler state survives across blocks** — each call is independently
//! flushed, and it's block overlap (not resampler continuity) that hides
//! the resulting edge transients.

use crate::soxr::{exact_ratio_pair, resample_chunk, SoxrRecipe};

/// Which link of the decode chain a resampler sits in. Python picks a
/// *different* soxr recipe per stage once the quality preset drops below
/// `high` (`HiFiDecode.py:1126-1137`), so the stage has to travel with the
/// quality setting rather than being folded into it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ResamplerStage {
    /// RF -> IF. Unused by the quadrature path (which never resamples RF),
    /// kept so the mapping table stays a faithful copy of Python's.
    If,
    /// Demodulated audio -> 192kHz intermediate rate.
    Audio,
    /// 192kHz intermediate -> the user's final output rate.
    AudioFinal,
}

/// Mirrors `--resampler_quality` (`HiFiDecode.py:1126-1137`) and its
/// per-stage expansion into soxr recipes.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ResamplerQuality {
    High,
    Medium,
    Low,
}

impl ResamplerQuality {
    fn recipe(self, stage: ResamplerStage) -> SoxrRecipe {
        match (self, stage) {
            (ResamplerQuality::High, _) => SoxrRecipe::Vhq,
            (ResamplerQuality::Medium, ResamplerStage::If) => SoxrRecipe::Lq,
            (ResamplerQuality::Medium, ResamplerStage::Audio) => SoxrRecipe::Mq,
            (ResamplerQuality::Medium, ResamplerStage::AudioFinal) => SoxrRecipe::Hq,
            (ResamplerQuality::Low, _) => SoxrRecipe::Lq,
        }
    }
}

/// One-shot block resampler, mirroring the resample-then-clear pattern
/// described above. `input_rate`/`output_rate` need only be
/// proportionally correct — only their ratio is used, and it is converted
/// to the same exact numerator/denominator pair Python derives with
/// `Fraction` before reaching libsoxr.
pub struct BlockResampler {
    /// libsoxr's input rate: the *denominator* of Python's `Fraction`.
    soxr_in_rate: f64,
    /// libsoxr's output rate: the *numerator* of Python's `Fraction`.
    soxr_out_rate: f64,
    recipe: SoxrRecipe,
}

impl BlockResampler {
    pub fn new(input_rate: f64, output_rate: f64, quality: ResamplerQuality, stage: ResamplerStage) -> Self {
        assert!(input_rate > 0.0 && output_rate > 0.0);
        // Python builds the stream as
        // `ResampleStream(denominator, numerator, ...)` from
        // `Fraction(output_rate / input_rate)`, so the ratio is reduced to
        // that exact pair *before* libsoxr sees it rather than being handed
        // over as two rates.
        let (numerator, denominator) = exact_ratio_pair(output_rate / input_rate);
        BlockResampler {
            soxr_in_rate: denominator,
            soxr_out_rate: numerator,
            recipe: quality.recipe(stage),
        }
    }

    /// Resamples the whole of `input` in one call, flushing the filter tail.
    pub fn resample(&self, input: &[f32]) -> Vec<f32> {
        resample_chunk(input, self.soxr_in_rate, self.soxr_out_rate, self.recipe)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f64::consts::PI;

    #[test]
    fn quality_presets_match_pythons_per_stage_recipes() {
        // `high` uses VHQ everywhere; `medium` fans out across three
        // different recipes; `low` uses LQ everywhere.
        for stage in [ResamplerStage::If, ResamplerStage::Audio, ResamplerStage::AudioFinal] {
            assert_eq!(ResamplerQuality::High.recipe(stage), SoxrRecipe::Vhq);
            assert_eq!(ResamplerQuality::Low.recipe(stage), SoxrRecipe::Lq);
        }
        assert_eq!(ResamplerQuality::Medium.recipe(ResamplerStage::If), SoxrRecipe::Lq);
        assert_eq!(ResamplerQuality::Medium.recipe(ResamplerStage::Audio), SoxrRecipe::Mq);
        assert_eq!(ResamplerQuality::Medium.recipe(ResamplerStage::AudioFinal), SoxrRecipe::Hq);
    }

    #[test]
    fn downsamples_a_tone_preserving_its_frequency() {
        let fs_in = 192_000.0;
        let fs_out = 48_000.0;
        let tone_hz = 1_000.0;
        let n = 9600usize; // 50ms at fs_in

        let input: Vec<f32> = (0..n)
            .map(|i| (2.0 * PI * tone_hz * i as f64 / fs_in).sin() as f32)
            .collect();

        let resampler = BlockResampler::new(fs_in, fs_out, ResamplerQuality::High, ResamplerStage::AudioFinal);
        let output = resampler.resample(&input);

        // Ratio 1:4, so output should be close to n/4 samples.
        let expected_len = (n as f64 * fs_out / fs_in) as usize;
        assert!(
            output.len().abs_diff(expected_len) < 50,
            "output len {} vs expected ~{expected_len}",
            output.len()
        );

        // Goertzel at 1kHz should dominate over a nearby off-tone bin,
        // skipping the filter's startup transient at the very start.
        let usable = &output[200..output.len() - 1];
        let goertzel = |freq: f64| -> f64 {
            let mut re = 0.0f64;
            let mut im = 0.0f64;
            for (i, &sample) in usable.iter().enumerate() {
                let angle = -2.0 * PI * freq * (i as f64) / fs_out;
                re += sample as f64 * angle.cos();
                im += sample as f64 * angle.sin();
            }
            (re * re + im * im).sqrt()
        };
        let at_tone = goertzel(tone_hz);
        let off_tone = goertzel(tone_hz * 1.7);
        assert!(
            at_tone > off_tone * 5.0,
            "tone bin {at_tone} not dominant over off-tone bin {off_tone}"
        );
    }
}
