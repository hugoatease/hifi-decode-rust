use anyhow::{bail, Context as _, Result};

/// Parse a sample-rate/frequency argument and return it in MHz.
///
/// Mirrors `vhsdecode/hifi/main.py`'s `parse_frequency` (and ld-decode's
/// equivalent): a bare number is MHz, and `ghz`/`mhz`/`khz`/`hz` scale from
/// that. `fsc` and `fscpal` are multiples of the NTSC and PAL color
/// subcarriers, so the common cxadc rates spell as `8fsc` (28.636 MHz) or
/// `4fsc`.
pub fn parse_frequency_mhz(value: &str) -> Result<f64> {
    let value = value.trim();
    let suffix_start = value
        .find(|ch: char| !matches!(ch, '0'..='9' | '.' | '-' | '+' | 'e' | 'E'))
        .unwrap_or(value.len());
    let (number, suffix) = value.split_at(suffix_start);
    let base = number.parse::<f64>().context("invalid frequency value")?;
    let multiplier = match suffix.to_ascii_lowercase().as_str() {
        "" | "m" | "mhz" => 1.0,
        "g" | "ghz" => 1.0e3,
        "k" | "khz" => 1.0e-3,
        "hz" => 1.0e-6,
        // 315/88 MHz, and 283.75 * 15625 + 25 Hz.
        "fsc" => 315.0 / 88.0,
        "fscpal" => (283.75 * 15625.0 + 25.0) / 1.0e6,
        _ => bail!("unknown frequency suffix: {suffix}"),
    };
    Ok(base * multiplier)
}

#[cfg(test)]
mod tests {
    use super::parse_frequency_mhz;

    fn parse(value: &str) -> f64 {
        parse_frequency_mhz(value).unwrap()
    }

    #[test]
    fn bare_number_is_mhz() {
        assert_eq!(parse("40"), 40.0);
        assert_eq!(parse(" 28.636 "), 28.636);
    }

    #[test]
    fn decimal_suffixes_scale_to_mhz() {
        assert_eq!(parse("1ghz"), 1000.0);
        assert_eq!(parse("40MHz"), 40.0);
        assert_eq!(parse("40M"), 40.0);
        assert_eq!(parse("40000khz"), 40.0);
        assert_eq!(parse("40000000hz"), 40.0);
    }

    #[test]
    fn subcarrier_suffixes_match_upstream() {
        assert!((parse("1fsc") - 3.579_545_454_545).abs() < 1e-9);
        assert!((parse("8fsc") - 28.636_363_636_363).abs() < 1e-9);
        assert_eq!(parse("1fscpal"), 4.433_618_75);
        assert_eq!(parse("4fscpal"), 4.433_618_75 * 4.0);
    }

    #[test]
    fn unknown_suffix_and_bad_number_are_errors() {
        assert!(parse_frequency_mhz("8fps").is_err());
        assert!(parse_frequency_mhz("fsc").is_err());
    }
}
