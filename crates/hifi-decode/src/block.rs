//! Block sizing and overlap. Ports `HiFiDecode.calculate_block_sizes` /
//! `_set_block_overlap` (`HiFiDecode.py:1279-1387`), `DecoderState` /
//! `DecoderSharedMemory` (`utils.py:257-309`, `:433-490`), and the
//! block-assembly half of `read_and_send_to_decoder`
//! (`main.py:1497-1567`) — together, the math that sizes ~0.5s decode
//! blocks, the RF-domain overlap padding that hides resampler edge
//! transients at block boundaries, and the rule for which slice of each
//! decoded block is that block's real contribution.
//!
//! The shape to keep in mind, because it is not the obvious one: each
//! block's *buffer* is `block_size` RF samples, but consecutive blocks
//! advance by only `stride = block_size - 2 * block_overlap`. Every block
//! therefore re-decodes `2 * block_overlap` samples its predecessor
//! already saw, and throws away the corresponding audio at both ends. The
//! per-block audio sizes are **not** simply `audio_final_rate / 2`: Python
//! re-derives them from the integer buffer length
//! (`ceil(audio_final_rate * buffer_len / input_rate)`), and because
//! `block_size` is itself a `ceil`, that ratio sits a hair above 0.5 and
//! the outer `ceil` rounds a whole sample up. On an 8fsc capture that
//! makes `block_audio_final_size` 24001 rather than 24000 — a one-sample
//! per-block difference that accumulates into seconds of drift over a
//! tape if it is not reproduced exactly.
//!
//! Unlike Python, which carries the overlap through a hand-rolled
//! shared-memory ring buffer (a multiprocessing IPC concern that doesn't
//! exist for Rust threads sharing one address space), this crate exposes
//! it as plain `[start, end)` ranges into the RF stream plus a small
//! recipe for how to assemble the buffer — see `Block`.

const BLOCKS_PER_SECOND: f64 = 2.0;

fn gcd(mut a: u64, mut b: u64) -> u64 {
    while b != 0 {
        (a, b) = (b, a % b);
    }
    a
}

/// Python's `round()`: half-to-even, not half-away-from-zero. The
/// distinction is load-bearing in at least two places here — the
/// `(decoded_len - emit_len) / 2` overlap trim lands on an exact `.5` on
/// every ordinary block of an 8fsc capture (525/2), and so does the
/// output's leading silence pad (263/2) — so using Rust's `f64::round`
/// would shift every block by one sample.
fn py_round(x: f64) -> f64 {
    x.round_ties_even()
}

/// How one decode block is assembled from the RF stream and which slice of
/// its decoded audio survives.
///
/// The buffer is `read` (the RF samples in `[read_start, read_end)`) with
/// `prepend_duplicate` samples copied from its own head glued onto the
/// front and `append_pad` samples of filler glued onto the back — both
/// zero except on the first and last block respectively.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Block {
    pub index: usize,
    /// RF span actually read from the input for this block.
    pub read_start: usize,
    pub read_end: usize,
    /// Samples duplicated from the head of `read` onto the front of the
    /// buffer. Non-zero only for block 0, which has no predecessor to
    /// borrow a left overlap from and so fakes one from its own start
    /// (`main.py:1525-1528`).
    pub prepend_duplicate: usize,
    /// Filler samples appended after `read`. Non-zero only for the last
    /// block — see `BlockLayout::last_block`.
    pub append_pad: usize,
    /// Total buffer length: `prepend_duplicate + (read_end - read_start) + append_pad`.
    pub buffer_len: usize,
    /// How many decoded samples this block contributes
    /// (`DecoderState.block_audio_final_len`).
    pub emit_len: usize,
    /// `block_audio_final_overlap`, carried here because the last block's
    /// trim is expressed in terms of it.
    pub audio_final_overlap: usize,
    pub is_last: bool,
}

impl Block {
    /// Where this block's contribution starts within its decoded output,
    /// given how many samples the decode actually produced
    /// (`hifi_decode_worker`, `HiFiDecode.py:2448-2460`).
    ///
    /// Note this is derived from the *measured* decoded length rather than
    /// a predicted one: the resampler's output length is its own business,
    /// and Python centres the kept span inside whatever it got.
    pub fn output_skip(&self, decoded_len: usize) -> usize {
        if self.is_last {
            // Take the tail instead of the middle: everything after this
            // block's real content is the `append_pad` filler, so the kept
            // span is pinned to end `audio_final_overlap` samples before
            // the end of the decode.
            decoded_len.saturating_sub(self.emit_len + self.audio_final_overlap)
        } else {
            let excess = decoded_len as f64 - self.emit_len as f64;
            if excess <= 0.0 {
                0
            } else {
                py_round(excess / 2.0) as usize
            }
        }
    }

    /// The `[skip, skip + take)` span of a decoded block to keep, already
    /// clamped to what the decode actually produced.
    pub fn output_span(&self, decoded_len: usize) -> (usize, usize) {
        let skip = self.output_skip(decoded_len).min(decoded_len);
        let take = self.emit_len.min(decoded_len - skip);
        (skip, take)
    }
}

/// Precomputed block/overlap sizing for one decode run. All fields mirror
/// their Python namesakes 1:1 for cross-reference.
#[derive(Clone, Copy, Debug)]
pub struct BlockLayout {
    input_rate: f64,
    audio_final_rate: f64,
    /// `HiFiDecode._initial_block_size`: the RF length of one block's
    /// buffer, ~0.5s.
    pub block_size: usize,
    /// `HiFiDecode._block_overlap`: RF-domain overlap on each side.
    pub block_overlap: usize,
    /// `HiFiDecode._block_read_overlap`, i.e. `2 * block_overlap`.
    pub block_read_overlap: usize,
    /// `HiFiDecode._block_audio_final_overlap`: the same overlap measured
    /// in final-rate audio samples.
    pub block_audio_final_overlap: usize,
    /// Fresh RF samples consumed per block — the length of Python's
    /// `block_in` view (`utils.py:532-539`), and hence how far the read
    /// position advances each iteration.
    pub stride: usize,
    pub pre_trim: usize,
}

impl BlockLayout {
    pub fn new(input_rate: f64, audio_rate: f64, audio_final_rate: f64) -> Self {
        let blocks_per_second_ratio = 1.0 / BLOCKS_PER_SECOND;
        let block_size = (input_rate * blocks_per_second_ratio).ceil() as u64;
        let block_audio_size = (audio_rate * blocks_per_second_ratio).ceil() as u64;
        let block_audio_final_size = (audio_final_rate * blocks_per_second_ratio).ceil() as u64;

        let block_size_gcd = gcd(block_size, block_audio_final_size);
        let block_audio_overlap_divisor = if block_size_gcd > 5 {
            block_audio_size / block_size_gcd
        } else {
            tracing::warn!(
                "input sample rate is not evenly divisible by the output sample rate; \
                 audio sync issues may occur (input {input_rate} Hz, output {audio_final_rate} Hz)"
            );
            1
        };

        let pre_trim: u64 = 1000;
        let min_resampler_overlap = pre_trim + 50;
        let min_overlap = ((min_resampler_overlap as f64) / audio_rate * audio_final_rate).ceil() as u64;
        let block_audio_final_overlap_seed =
            (min_overlap as f64 / block_audio_overlap_divisor as f64).ceil() as u64 * block_audio_overlap_divisor;

        let overlap_seconds = block_audio_final_overlap_seed as f64 / audio_final_rate;
        let block_overlap = py_round(input_rate * overlap_seconds) as u64;
        let block_read_overlap = block_overlap * 2;
        let block_audio_final_overlap = py_round(audio_final_rate * overlap_seconds) as u64;

        BlockLayout {
            input_rate,
            audio_final_rate,
            block_size: block_size as usize,
            block_overlap: block_overlap as usize,
            block_read_overlap: block_read_overlap as usize,
            block_audio_final_overlap: block_audio_final_overlap as usize,
            stride: (block_size - block_read_overlap) as usize,
            pre_trim: pre_trim as usize,
        }
    }

    /// `DecoderState`'s re-derivation of the audio block size from the
    /// buffer's own integer length (`utils.py:259`, via
    /// `calculate_block_sizes`'s `block_size` argument). See this module's
    /// doc comment for why this is not just `audio_final_rate / 2`.
    pub fn block_audio_final_size(&self, buffer_len: usize) -> usize {
        let ratio = buffer_len as f64 / self.input_rate;
        (self.audio_final_rate * ratio).ceil() as usize
    }

    /// How many decoded samples an ordinary (non-last) block contributes:
    /// its audio size less the overlap discarded at each end
    /// (`DecoderState.block_audio_final_len`, `utils.py:299-308`).
    fn emit_len(&self, buffer_len: usize) -> usize {
        let bafs = self.block_audio_final_size(buffer_len);
        bafs.saturating_sub(self.block_audio_final_overlap * 2).max(50)
    }

    /// The last block keeps only as much audio as the RF it actually read
    /// is worth, plus the overlap a following block would have consumed
    /// (`utils.py:301-305`).
    fn last_emit_len(&self, buffer_len: usize, frames_read: usize) -> usize {
        let bafs = self.block_audio_final_size(buffer_len);
        let audio_size = py_round(frames_read as f64 * bafs as f64 / buffer_len as f64) as usize;
        (audio_size + self.block_audio_final_overlap).max(50)
    }

    /// Fresh RF samples block `index` reads: `[index * stride, ... + stride)`.
    /// Python reads exactly this much per iteration and lets the block
    /// assembly pull the rest from the previous buffer.
    pub fn fresh_read_start(&self, index: usize) -> usize {
        index * self.stride
    }

    /// Block 0 (`main.py:1497-1528`): there is no predecessor, so the left
    /// overlap is faked by duplicating the first `block_overlap` samples of
    /// the block's own data. The buffer is correspondingly shorter than
    /// every other block's, which is why its audio size gets re-derived
    /// from that shorter length rather than shared with the rest.
    pub fn first_block(&self, frames_read: usize, is_last: bool) -> Block {
        let prepend_duplicate = self.block_overlap.min(frames_read);
        let buffer_len = frames_read + prepend_duplicate;
        let emit_len = if is_last {
            self.last_emit_len(buffer_len, frames_read)
        } else {
            self.emit_len(buffer_len)
        };
        Block {
            index: 0,
            read_start: 0,
            read_end: frames_read,
            prepend_duplicate,
            append_pad: 0,
            buffer_len,
            emit_len,
            audio_final_overlap: self.block_audio_final_overlap,
            is_last,
        }
    }

    /// An ordinary block: a contiguous `block_size`-sample RF window whose
    /// first `block_read_overlap` samples are the tail of the previous
    /// block's buffer. Because reads are contiguous, that window simply
    /// starts `block_read_overlap` samples before this block's own fresh
    /// read.
    pub fn middle_block(&self, index: usize) -> Block {
        debug_assert!(index > 0, "block 0 is built by first_block");
        let read_start = self.fresh_read_start(index) - self.block_read_overlap;
        let read_end = read_start + self.block_size;
        Block {
            index,
            read_start,
            read_end,
            prepend_duplicate: 0,
            append_pad: 0,
            buffer_len: self.block_size,
            emit_len: self.emit_len(self.block_size),
            audio_final_overlap: self.block_audio_final_overlap,
            is_last: false,
        }
    }

    /// The final block (`main.py:1529-1545`), which ran out of input part
    /// way through its fresh read.
    ///
    /// Python keeps the buffer a full `block_size` long by sliding the
    /// short read to the right so it ends `block_overlap` samples before
    /// the buffer's end, backfilling the front from the previous block and
    /// leaving that trailing `block_overlap` window untouched. Those
    /// trailing samples are whatever the recycled shared-memory buffer
    /// happened to hold — genuinely stale data on Python's side — but they
    /// sit entirely inside the span the last block's trim discards, so
    /// this port fills them with zeros rather than reproducing an
    /// uninitialised read. (Filtering is not perfectly local, so the
    /// filler can bleed a little way back into the kept audio; that is
    /// true of Python's stale data too, and is the one place where the two
    /// implementations cannot be made identical by construction.)
    pub fn last_block(&self, index: usize, input_end: usize, frames_read: usize) -> Block {
        let append_pad = self.block_overlap;
        let read_len = self.block_size.saturating_sub(append_pad).min(input_end);
        let read_start = input_end - read_len;
        let buffer_len = read_len + append_pad;
        Block {
            index,
            read_start,
            read_end: input_end,
            prepend_duplicate: 0,
            append_pad,
            buffer_len,
            emit_len: self.last_emit_len(buffer_len, frames_read),
            audio_final_overlap: self.block_audio_final_overlap,
            is_last: true,
        }
    }

    /// Assembles a block's decode buffer from the RF samples of its
    /// `[read_start, read_end)` span.
    pub fn assemble(&self, block: &Block, read: &[f32]) -> Vec<f32> {
        debug_assert_eq!(read.len(), block.read_end - block.read_start);
        let mut buffer = Vec::with_capacity(block.buffer_len);
        buffer.extend_from_slice(&read[..block.prepend_duplicate]);
        buffer.extend_from_slice(read);
        buffer.resize(block.buffer_len, 0.0);
        buffer
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The real 8fsc / 48kHz configuration, whose numbers were read out of
    /// an instrumented `vhsdecode.hifi` run.
    fn pal_8fsc() -> BlockLayout {
        BlockLayout::new(28_636_363.0, 192_000.0, 48_000.0)
    }

    #[test]
    fn sizing_matches_instrumented_python_run() {
        let layout = pal_8fsc();
        assert_eq!(layout.block_size, 14_318_182);
        assert_eq!(layout.block_overlap, 156_903);
        assert_eq!(layout.block_read_overlap, 313_806);
        assert_eq!(layout.block_audio_final_overlap, 263);
        assert_eq!(layout.stride, 14_004_376);
    }

    /// The `ceil` that bumps an ordinary block to 24001 rather than 24000
    /// audio samples — see the module doc comment.
    #[test]
    fn audio_size_is_re_derived_from_the_integer_buffer_length() {
        let layout = pal_8fsc();
        assert_eq!(layout.block_audio_final_size(layout.block_size), 24_001);
        assert_eq!(layout.emit_len(layout.block_size), 23_475);
    }

    #[test]
    fn first_block_fakes_its_left_overlap_and_gets_its_own_audio_size() {
        let layout = pal_8fsc();
        let block = layout.first_block(layout.stride, false);
        assert_eq!(block.prepend_duplicate, 156_903);
        assert_eq!(block.buffer_len, 14_161_279);
        assert_eq!(layout.block_audio_final_size(block.buffer_len), 23_738);
        assert_eq!(block.emit_len, 23_212);
    }

    /// Ordinary blocks are contiguous RF windows that step by `stride`
    /// while spanning `block_size`, so each re-reads its predecessor's
    /// last `block_read_overlap` samples and nothing is ever skipped.
    #[test]
    fn middle_blocks_overlap_by_the_read_overlap_without_gaps() {
        let layout = pal_8fsc();
        let a = layout.middle_block(1);
        let b = layout.middle_block(2);
        assert_eq!(a.read_end - a.read_start, layout.block_size);
        assert_eq!(b.read_start - a.read_start, layout.stride);
        assert_eq!(a.read_end - b.read_start, layout.block_read_overlap);
        // Block 1 picks up exactly where block 0's fresh read left off,
        // minus the overlap it needs from it.
        assert_eq!(a.read_start, layout.stride - layout.block_read_overlap);
    }

    /// Python centres the kept span in the decode, rounding halves to
    /// even: an ordinary 8fsc block decodes to 24000 samples and keeps
    /// 23475 of them, so the excess is 525 and the trim lands on 262.5.
    #[test]
    fn output_span_rounds_halves_to_even_like_python() {
        let layout = pal_8fsc();
        let block = layout.middle_block(1);
        let (skip, take) = block.output_span(24_000);
        assert_eq!((skip, take), (262, 23_475));
    }

    #[test]
    fn last_block_keeps_the_tail_ending_one_overlap_early() {
        let layout = pal_8fsc();
        // 7531339 frames read, as in the instrumented 12s run.
        let block = layout.last_block(24, 24 * layout.stride + 7_531_339, 7_531_339);
        assert_eq!(block.buffer_len, layout.block_size);
        assert_eq!(block.append_pad, layout.block_overlap);
        assert_eq!(block.emit_len, 12_887);
        let (skip, take) = block.output_span(24_000);
        assert_eq!((skip, take), (10_850, 12_887));
        assert_eq!(skip + take, 24_000 - layout.block_audio_final_overlap);
    }

    #[test]
    fn assemble_duplicates_the_head_and_pads_the_tail() {
        let layout = BlockLayout::new(8_000_000.0, 192_000.0, 48_000.0);
        let block = Block {
            index: 0,
            read_start: 0,
            read_end: 4,
            prepend_duplicate: 2,
            append_pad: 3,
            buffer_len: 9,
            emit_len: 50,
            audio_final_overlap: layout.block_audio_final_overlap,
            is_last: true,
        };
        let buffer = layout.assemble(&block, &[1.0, 2.0, 3.0, 4.0]);
        assert_eq!(buffer, vec![1.0, 2.0, 1.0, 2.0, 3.0, 4.0, 0.0, 0.0, 0.0]);
    }
}
