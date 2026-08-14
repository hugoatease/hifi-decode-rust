# hifi-decode-rust

Extracts audio from raw HiFi FM RF captures — VHS HiFi and Video8/Hi8 AFM.

A Rust port of [`vhs-decode`](https://github.com/oyvindln/vhs-decode)'s `hifi-decode`
(`vhsdecode/hifi/HiFiDecode.py`). It aims to be a drop-in replacement: same command-line
surface, same decoding chain, and output that lines up with the reference sample for
sample.

## Status

Validated end-to-end against `vhsdecode.hifi` on a full 63-minute 8fsc capture:

| Check | Result |
|---|---|
| Sample count | identical |
| Timing offset at 60 / 600 / 1800 / 3000 s | 0 samples, no drift |
| Bit-identical samples in sampled windows | 99.98% |
| Residual difference | −160 dBFS RMS, −132 dBFS peak |

The residual is 1–2 LSB at 24 bits on a minority of samples — inaudible, and no sample
anywhere differs by more than −60 dBFS. Reduced to 16 bits, 17 samples in 1,126,538 differ,
by one LSB each.

Individual DSP stages are checked against captured Python output by fixture tests: the AFE
bandpass, quadrature demodulator, resampler, DC trim, dropout compensation and stereo mix
are bit-identical.

## Requirements

Two native libraries, both of which the Python reference also uses (via the `soxr` and
`soundfile` packages). They are found with `pkg-config`:

```sh
# macOS
brew install libsoxr libsndfile

# Debian / Ubuntu
apt install libsoxr-dev libsndfile1-dev

# Fedora
dnf install soxr-devel libsndfile-devel

# Arch
pacman -S libsoxr libsndfile
```

Resampling goes through libsoxr and output is written through libsndfile so that both match
the reference exactly, rather than being approximated and kept in sync by hand. The build
fails with an install hint if either is missing.

## Build

```sh
cargo build --release
```

The binary lands at `target/release/hifi-decode`.

## Usage

```sh
hifi-decode [OPTIONS] <INFILE> <OUTFILE>
```

A PAL Video8 tape, captured at 28.636 MHz — `8fsc`, a standard cxadc clock rate, named
after the NTSC colour subcarrier regardless of the tape's own standard — decoded to 48 kHz
FLAC:

```sh
hifi-decode -p -f 8fsc --8mm --audio_rate 48000 capture.flac out.flac
```

A 40 MHz VHS HiFi capture of raw 8-bit samples, read from stdin:

```sh
cat capture.raw | hifi-decode -f 40 --input-format u8 - out.flac
```

`OUTFILE` ending in `.wav` writes 16-bit PCM; anything else writes 24-bit FLAC. In the `d`
and `dms` audio modes two mono files are written instead, named `<stem>_channel_1<ext>` and
`<stem>_channel_2<ext>`.

### Common options

| Option | Meaning |
|---|---|
| `-f, --frequency` | RF sample rate. Bare number is MHz; `hz`/`khz`/`mhz`/`ghz`/`fsc`/`fscpal` suffixes accepted (e.g. `8fsc`, `28.636mhz`). Default `40` |
| `-p` / `-n` | PAL / NTSC source |
| `--8mm` | Video8/Hi8 AFM settings instead of VHS HiFi |
| `--input-format` | Input sample encoding — required for stdin |
| `--audio_rate` | Output sample rate in Hz. Default `48000` |
| `--audio_mode` | `s`, `ms`, `d`, `dms`, `l`, `r`, `sum`. Defaults to `s` for VHS, `ms` for `--8mm` |
| `--gain` | Manual output gain multiplier |
| `--resampler-quality` | `high` (default), `medium`, `low` |
| `--overwrite` | Allow replacing an existing output file |

Run `hifi-decode --help` for the full list, including the expander, de-emphasis,
head-switching and dropout-compensation parameters.

## Not ported

Passing these is either rejected or has no effect:

- **Hilbert demodulation** (`--demod hilbert`) — only the quadrature path is implemented
- **Carrier auto-tracking** (`--bias_guess`, `--auto_fine_tune`)
- **Spectral noise reduction** (`--NR_spectral_amount`) — only `0` is accepted
- The GUI, and the real-time preview/playback options

## Known limitations

**The final ~0.27 s of a decode cannot match the reference.** `vhsdecode.hifi` assembles its
last block in a recycled shared-memory buffer and leaves part of it uninitialised, so its
own output there changes with thread count. This port zero-fills instead, which is at least
deterministic.

**A 42.5 ppm rate error is inherited from the reference.** Each block emits 23,475 samples
for a stride whose true duration is 23,474.0017 — roughly 165 ms of accumulated offset over
an hour. This comes from the upstream block-sizing math, which `vhs-decode` itself warns
about (`WARNING: The input sample rate is not evenly divisible by the output sample rate`).
Aligning the decoded audio against the video TBC — for instance with
[VhsDecodeAutoAudioAlign](https://github.com/oyvindln/vhs-decode/wiki) — remains necessary.

## Repository layout

| Crate | Contents |
|---|---|
| `crates/hifi-decode` | The decoding chain: AFE filtering, FM demodulation, resampling, dropout compensation, head-switch noise removal, post-processing |
| `crates/hifi-decode-cli` | The `hifi-decode` binary: argument parsing, block-parallel orchestration, output writing |
| `crates/tape-dsp` | Shared DSP primitives: Chebyshev II design, zero-phase SOS filtering, FFT, angle unwrapping |
| `crates/tape-rf-io` | RF input: sample-format decoding, FLAC reading, frequency parsing |

`fixtures/hifi/` holds stage-by-stage output captured from the real Python decoder, run
against a deterministic synthetic RF signal; `scripts/hifi-fixtures/` regenerates it.

```sh
cargo test --release
```

## Licence

GPL-3.0, matching `vhs-decode`. See [COPYING](COPYING).
