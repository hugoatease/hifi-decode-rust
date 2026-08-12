//! Player/recorder calibration diagnostics. Ports the math half of
//! `HiFiDecode.log_bias` (`HiFiDecode.py:1506-1525`): converts the per-block
//! DC offset `cancel_dc_trim` measures into a carrier bias in kHz, and
//! classifies it against the same good/marginal/uncalibrated thresholds
//! Python uses. The message formatting and logging live in
//! `hifi-decode-cli`, alongside the rest of that crate's per-chunk
//! reporting (progress, etc.) — this module is deliberately just the pure
//! calculation, independent of `--bias_guess`/`--auto_fine_tune` (carrier
//! auto-tracking, not ported).

/// Carrier bias in kHz implied by a channel's post-demod DC offset.
/// `HiFiDecode.log_bias`'s `devL`/`devR` (`HiFiDecode.py:1507-1508`):
/// `dc * carrier_deviation / 1e3`.
pub fn carrier_bias_khz(dc: f32, carrier_deviation: f64) -> f64 {
    dc as f64 * carrier_deviation / 1e3
}

/// `HiFiDecode.log_bias`'s three-way classification
/// (`HiFiDecode.py:1517-1525`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CalibrationQuality {
    Good,
    Marginal,
    Uncalibrated,
}

/// Classifies a block's L/R carrier bias (in kHz) the way Python's
/// `log_bias` does. Pass `0.0` for whichever channel the current
/// `DecodeMode` doesn't use (mirrors Python forcing the unused channel's
/// `dc` to `0` in `block_decode`), so only the active channel(s) can trip
/// the thresholds.
pub fn classify_calibration(dev_l_khz: f64, dev_r_khz: f64) -> CalibrationQuality {
    let (l, r) = (dev_l_khz.abs(), dev_r_khz.abs());
    if l < 9.0 && r < 9.0 {
        CalibrationQuality::Good
    } else if (9.0..10.0).contains(&l) || (9.0..10.0).contains(&r) {
        CalibrationQuality::Marginal
    } else {
        CalibrationQuality::Uncalibrated
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn carrier_bias_khz_scales_dc_by_deviation_into_khz() {
        // dc=1.0 at a 150kHz deviation is a full 150kHz swing, i.e. 150kHz bias.
        assert_eq!(carrier_bias_khz(1.0, 150_000.0), 150.0);
        assert_eq!(carrier_bias_khz(-0.5, 150_000.0), -75.0);
        assert_eq!(carrier_bias_khz(0.0, 150_000.0), 0.0);
    }

    #[test]
    fn both_channels_under_nine_khz_is_good() {
        assert_eq!(classify_calibration(0.0, 0.0), CalibrationQuality::Good);
        assert_eq!(classify_calibration(8.99, 8.99), CalibrationQuality::Good);
        // abs() applies: a negative bias under threshold is still good.
        assert_eq!(classify_calibration(-8.99, 0.0), CalibrationQuality::Good);
    }

    #[test]
    fn either_channel_in_nine_to_ten_is_marginal() {
        assert_eq!(classify_calibration(9.0, 0.0), CalibrationQuality::Marginal);
        assert_eq!(classify_calibration(0.0, 9.5), CalibrationQuality::Marginal);
        assert_eq!(classify_calibration(-9.5, 0.0), CalibrationQuality::Marginal);
        // Just under the marginal band's upper edge.
        assert_eq!(classify_calibration(9.999, 0.0), CalibrationQuality::Marginal);
    }

    #[test]
    fn ten_khz_or_more_on_either_channel_is_uncalibrated() {
        assert_eq!(classify_calibration(10.0, 0.0), CalibrationQuality::Uncalibrated);
        assert_eq!(classify_calibration(0.0, 12.0), CalibrationQuality::Uncalibrated);
        assert_eq!(classify_calibration(-10.0, 0.0), CalibrationQuality::Uncalibrated);
    }

    #[test]
    fn good_requires_both_channels_under_threshold() {
        // L is good but R is marginal -> overall marginal, not good (AND vs OR).
        assert_eq!(classify_calibration(1.0, 9.2), CalibrationQuality::Marginal);
        // L is good but R is uncalibrated -> overall uncalibrated.
        assert_eq!(classify_calibration(1.0, 15.0), CalibrationQuality::Uncalibrated);
    }
}
