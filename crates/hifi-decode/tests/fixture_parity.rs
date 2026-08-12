//! Validates the AFE filter and quadrature FM demodulator against real
//! output from Python's `vhsdecode.hifi.HiFiDecode`, captured in
//! `fixtures/hifi/synthetic_pal_vhs/` (see `fixtures/hifi/README.md` for
//! how they were generated and their known limitations).

mod support;

use hifi_decode::{
    cancel_dc_trim, carrier_bias_khz, dropout_compensate, mix_for_mode_stereo, AfeFilter,
    AfeOverrides, AfeParams, DecodeMode, DropoutParams, FmDiscriminator, System, TapeFormat,
};

const FS: f64 = 8_000_000.0;
const AUDIO_RATE: f64 = 192_000.0;
const PRE_TRIM: usize = 1000;

fn fixture_path(name: &str) -> std::path::PathBuf {
    std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../fixtures/hifi/synthetic_pal_vhs")
        .join(name)
}

fn max_abs_diff(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len(), "length mismatch: {} vs {}", a.len(), b.len());
    a.iter()
        .zip(b)
        .map(|(&x, &y)| (x - y).abs())
        .fold(0.0f32, f32::max)
}

#[test]
fn afe_bandpass_matches_python() {
    let rf = support::load_f32(fixture_path("rf_input.npy"));
    let expected_l = support::load_f32(fixture_path("01_afe_filter_l.npy"));
    let expected_r = support::load_f32(fixture_path("01_afe_filter_r.npy"));

    let params = AfeParams::for_format(TapeFormat::Vhs, System::Pal, AfeOverrides::default());
    let afe_l = AfeFilter::design(params.l_carrier_ref, params.l_notch_width, FS);
    let afe_r = AfeFilter::design(params.r_carrier_ref, params.r_notch_width, FS);

    let got_l = afe_l.work(&rf);
    let got_r = afe_r.work(&rf);

    // Zero-phase SOS filtering, no resampling involved: expect near-bit
    // parity, limited only by f32-vs-f64 rounding differences between this
    // port's biquad recurrence and scipy/numpy's.
    let diff_l = max_abs_diff(&got_l, &expected_l);
    let diff_r = max_abs_diff(&got_r, &expected_r);
    assert!(diff_l < 1e-3, "L channel max abs diff {diff_l}");
    assert!(diff_r < 1e-3, "R channel max abs diff {diff_r}");
}

#[test]
fn quadrature_demod_matches_python() {
    let rf = support::load_f32(fixture_path("rf_input.npy"));
    let expected_l = support::load_f32(fixture_path("02_demod_l.npy"));
    let expected_r = support::load_f32(fixture_path("02_demod_r.npy"));

    let params = AfeParams::for_format(TapeFormat::Vhs, System::Pal, AfeOverrides::default());
    let afe_l = AfeFilter::design(params.l_carrier_ref, params.l_notch_width, FS);
    let afe_r = AfeFilter::design(params.r_carrier_ref, params.r_notch_width, FS);
    let filtered_l = afe_l.work(&rf);
    let filtered_r = afe_r.work(&rf);

    let disc_l = FmDiscriminator::new_quadrature(
        FS,
        params.l_carrier_ref,
        params.l_carrier_deviation,
        filtered_l.len(),
    );
    let disc_r = FmDiscriminator::new_quadrature(
        FS,
        params.r_carrier_ref,
        params.r_carrier_deviation,
        filtered_r.len(),
    );

    let got_l = disc_l.work(&filtered_l);
    let got_r = disc_r.work(&filtered_r);

    // Exclude the last sample: the Python reference never writes it (see
    // `FmDiscriminator::work_into`'s doc comment), so it's not meaningful
    // to compare.
    let n = expected_l.len() - 1;
    let diff_l = max_abs_diff(&got_l[..n], &expected_l[..n]);
    let diff_r = max_abs_diff(&got_r[..n], &expected_r[..n]);
    assert!(diff_l < 1e-2, "L channel max abs diff {diff_l}");
    assert!(diff_r < 1e-2, "R channel max abs diff {diff_r}");
}

#[test]
fn cancel_dc_trim_matches_python() {
    let mut got_l = support::load_f32(fixture_path("03_resampled_l.npy"));
    let mut got_r = support::load_f32(fixture_path("03_resampled_r.npy"));
    let expected_l = support::load_f32(fixture_path("04_dc_trimmed_l.npy"));
    let expected_r = support::load_f32(fixture_path("04_dc_trimmed_r.npy"));

    let dc_l = cancel_dc_trim(&mut got_l, PRE_TRIM);
    let dc_r = cancel_dc_trim(&mut got_r, PRE_TRIM);

    let diff_l = max_abs_diff(&got_l, &expected_l);
    let diff_r = max_abs_diff(&got_r, &expected_r);
    assert!(diff_l < 1e-4, "L channel max abs diff {diff_l}");
    assert!(diff_r < 1e-4, "R channel max abs diff {diff_r}");

    // Check against Python's own recorded DC values (04_dc_values.json),
    // not just internal self-consistency.
    let dc_json: serde_json::Value =
        serde_json::from_slice(&std::fs::read(fixture_path("04_dc_values.json")).unwrap()).unwrap();
    let expected_dc_l = dc_json["dc_l"].as_f64().unwrap() as f32;
    let expected_dc_r = dc_json["dc_r"].as_f64().unwrap() as f32;
    assert!((dc_l - expected_dc_l).abs() < 1e-6, "dc_l {dc_l} vs python {expected_dc_l}");
    assert!((dc_r - expected_dc_r).abs() < 1e-6, "dc_r {dc_r} vs python {expected_dc_r}");
}

/// `carrier_bias_khz` (the math half of `HiFiDecode.log_bias`,
/// `HiFiDecode.py:1507-1508`) against the real DC values Python's own
/// decoder measured on this fixture, converted with the same fixture's
/// recorded carrier deviations (`metadata.json`'s `standard.LCarrierDeviation`
/// / `RCarrierDeviation`) — the one place a `carrier_bias_khz` result is
/// checked against something derived from real Python output rather than
/// only synthetic unit-test inputs (see `hifi_decode::bias`'s own tests for
/// those).
#[test]
fn carrier_bias_khz_matches_hand_computed_value_from_fixture() {
    let dc_json: serde_json::Value =
        serde_json::from_slice(&std::fs::read(fixture_path("04_dc_values.json")).unwrap()).unwrap();
    let dc_l = dc_json["dc_l"].as_f64().unwrap() as f32;
    let dc_r = dc_json["dc_r"].as_f64().unwrap() as f32;

    let metadata: serde_json::Value =
        serde_json::from_slice(&std::fs::read(fixture_path("metadata.json")).unwrap()).unwrap();
    let l_deviation = metadata["standard"]["LCarrierDeviation"].as_f64().unwrap();
    let r_deviation = metadata["standard"]["RCarrierDeviation"].as_f64().unwrap();

    // Hand-computed from the same formula (`dc * carrier_deviation / 1e3`)
    // against the full f64-precision JSON values, independent of the f32
    // round-trip `carrier_bias_khz` itself does on `dc`.
    let expected_dev_l = dc_json["dc_l"].as_f64().unwrap() * l_deviation / 1e3;
    let expected_dev_r = dc_json["dc_r"].as_f64().unwrap() * r_deviation / 1e3;

    let dev_l = carrier_bias_khz(dc_l, l_deviation);
    let dev_r = carrier_bias_khz(dc_r, r_deviation);

    assert!((dev_l - expected_dev_l).abs() < 1e-6, "dev_l {dev_l} vs expected {expected_dev_l}");
    assert!((dev_r - expected_dev_r).abs() < 1e-6, "dev_r {dev_r} vs expected {expected_dev_r}");
}

#[test]
fn dropout_compensate_matches_python_on_clean_signal() {
    let mut got_l = support::load_f32(fixture_path("04_dc_trimmed_l.npy"));
    let mut got_r = support::load_f32(fixture_path("04_dc_trimmed_r.npy"));
    let expected_l = support::load_f32(fixture_path("05_doc_l.npy"));
    let expected_r = support::load_f32(fixture_path("05_doc_r.npy"));

    let params = DropoutParams::new(AUDIO_RATE);
    dropout_compensate(&mut got_l, &mut got_r, &params, DecodeMode::Stereo, false);

    // On this clean synthetic signal Python's dropout_compensate is a
    // documented no-op (fixtures/hifi/README.md); the port should agree
    // exactly, not just approximately, since no fill/mute path is taken.
    assert_eq!(got_l, expected_l);
    assert_eq!(got_r, expected_r);
}

#[test]
fn mix_for_mode_stereo_is_identity_in_stereo_mode() {
    let l = support::load_f32(fixture_path("06_headswitch_final_l.npy"));
    let r = support::load_f32(fixture_path("06_headswitch_final_r.npy"));
    let expected_l = support::load_f32(fixture_path("07_stereo_mixed_l.npy"));
    let expected_r = support::load_f32(fixture_path("07_stereo_mixed_r.npy"));

    let (got_l, got_r) = mix_for_mode_stereo(&l, &r, DecodeMode::Stereo);
    assert_eq!(got_l, expected_l);
    assert_eq!(got_r, expected_r);
}
