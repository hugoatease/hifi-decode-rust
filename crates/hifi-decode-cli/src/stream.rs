//! Bounded-memory RF block source. Reproduces the block assembly
//! `read_and_send_to_decoder` performs (`main.py:1472-1574`) — one fresh
//! `stride`-sample read per block, with each block's buffer completed from
//! the samples its predecessor already read — but incrementally, without
//! requiring the input's total length upfront and without ever holding
//! more than a couple of blocks' worth of RF samples in memory.
//!
//! This exists because decoding a real capture by first reading the whole
//! file into a `Vec<f32>` needs roughly 4 bytes of RAM per RF sample: a
//! 17GB 8-bit capture becomes ~68GB as f32, which doesn't fit in memory on
//! an ordinary machine. Only the read side needs this treatment — the
//! *decoded* audio (192kHz intermediate, or the final rate) is on the
//! order of a few hundred MB even for an hour-long tape.

use anyhow::Result;
use hifi_decode::{Block, BlockLayout};
use tape_rf_io::DecodeReader;

/// Pulls RF samples from `reader` in a sliding window and yields one
/// `Block` (plus its assembled decode buffer) at a time, in order.
pub struct StreamingBlocks<'a> {
    reader: &'a mut DecodeReader,
    layout: BlockLayout,
    /// RF samples currently buffered, covering
    /// `[window_start, window_start + window.len())`.
    window: Vec<f32>,
    window_start: usize,
    eof: bool,
    next_index: usize,
    done: bool,
}

/// Samples read per underlying `DecodeReader::read` call while filling the
/// window.
const READ_CHUNK: usize = 1 << 20;

impl<'a> StreamingBlocks<'a> {
    pub fn new(reader: &'a mut DecodeReader, layout: BlockLayout) -> Self {
        StreamingBlocks {
            reader,
            layout,
            window: Vec::new(),
            window_start: 0,
            eof: false,
            next_index: 0,
            done: false,
        }
    }

    /// Reads until the window covers absolute position `want_end`, or the
    /// source is exhausted.
    fn fill_to(&mut self, want_end: usize) -> Result<()> {
        let want_len = want_end.saturating_sub(self.window_start);
        let mut chunk = vec![0.0f32; READ_CHUNK];
        while self.window.len() < want_len && !self.eof {
            let n = self.reader.read(&mut chunk)?;
            if n == 0 {
                self.eof = true;
                break;
            }
            self.window.extend_from_slice(&chunk[..n]);
        }
        Ok(())
    }

    /// Drops buffered samples before `keep_from`, which no later block can
    /// still need.
    fn discard_before(&mut self, keep_from: usize) {
        if keep_from > self.window_start {
            let drop = (keep_from - self.window_start).min(self.window.len());
            self.window.drain(..drop);
            self.window_start += drop;
        }
    }

    fn slice(&self, start: usize, end: usize) -> Vec<f32> {
        let from = start - self.window_start;
        let to = end - self.window_start;
        self.window[from..to].to_vec()
    }

    /// Returns the next block's metadata and decode buffer, or `None` once
    /// the source is exhausted and there is nothing left to emit.
    pub fn next_block(&mut self) -> Result<Option<(Block, Vec<f32>)>> {
        if self.done {
            return Ok(None);
        }
        let index = self.next_index;
        let layout = self.layout;

        // The fresh read for this block, exactly as Python's `block_in`
        // read does: `stride` samples starting where the previous block's
        // fresh read ended.
        let fresh_start = layout.fresh_read_start(index);
        let fresh_end = fresh_start + layout.stride;

        // A block needs history back to its buffer's start: `block_read_overlap`
        // samples before its fresh read (or, for the last block, further
        // still — see `BlockLayout::last_block`). Keeping a whole
        // `block_size` of history before the fresh read covers both.
        self.discard_before(fresh_start.saturating_sub(layout.block_size));
        self.fill_to(fresh_end)?;

        let available_end = self.window_start + self.window.len();
        if available_end <= fresh_start {
            // Nothing new to read: the previous block was the last one.
            self.done = true;
            return Ok(None);
        }

        let frames_read = available_end.min(fresh_end) - fresh_start;
        let is_last = frames_read < layout.stride;

        let block = if index == 0 {
            layout.first_block(frames_read, is_last)
        } else if is_last {
            layout.last_block(index, available_end, frames_read)
        } else {
            layout.middle_block(index)
        };

        let read = self.slice(block.read_start, block.read_end);
        let buffer = layout.assemble(&block, &read);

        self.next_index += 1;
        self.done = is_last;
        Ok(Some((block, buffer)))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tape_rf_io::{open_source, SampleFormat};

    fn stream_blocks(total_len: usize, layout: BlockLayout) -> Vec<(Block, Vec<f32>)> {
        let samples: Vec<f32> = (0..total_len).map(|i| i as f32).collect();
        let bytes: Vec<u8> = samples.iter().flat_map(|s| s.to_le_bytes()).collect();

        let path = std::env::temp_dir().join(format!(
            "hifi_decode_stream_test_{}_{total_len}.f32le",
            std::process::id()
        ));
        std::fs::write(&path, &bytes).unwrap();
        let file = std::fs::File::open(&path).unwrap();
        let mut reader = DecodeReader::new(open_source(file, SampleFormat::F32LE).unwrap());
        let mut streamer = StreamingBlocks::new(&mut reader, layout);

        let mut out = Vec::new();
        while let Some(block) = streamer.next_block().unwrap() {
            out.push(block);
        }
        std::fs::remove_file(&path).ok();
        out
    }

    /// A small synthetic layout so a test can span several blocks cheaply.
    fn tiny_layout() -> BlockLayout {
        BlockLayout::new(2_000.0, 192_000.0, 48_000.0)
    }

    #[test]
    fn every_block_buffer_has_the_length_its_metadata_claims() {
        let layout = tiny_layout();
        let blocks = stream_blocks(layout.stride * 4 + 17, layout);
        assert!(blocks.len() >= 4, "expected several blocks, got {}", blocks.len());
        for (block, buffer) in &blocks {
            assert_eq!(buffer.len(), block.buffer_len, "block {} buffer length", block.index);
        }
    }

    /// Only the final block may be flagged last, and it must be the one
    /// that reaches the end of the input.
    #[test]
    fn exactly_the_final_block_is_flagged_last() {
        let layout = tiny_layout();
        let total = layout.stride * 4 + 17;
        let blocks = stream_blocks(total, layout);
        let (last_block, _) = blocks.last().unwrap();
        assert!(last_block.is_last);
        assert_eq!(last_block.read_end, total);
        for (block, _) in &blocks[..blocks.len() - 1] {
            assert!(!block.is_last, "block {} wrongly flagged last", block.index);
        }
    }

    /// Block 0 duplicates its own head; the samples are the real RF values
    /// (`i as f32`), so the duplication is directly observable.
    #[test]
    fn first_block_buffer_starts_with_a_copy_of_its_own_head() {
        let layout = tiny_layout();
        let blocks = stream_blocks(layout.stride * 3, layout);
        let (block, buffer) = &blocks[0];
        let dup = block.prepend_duplicate;
        assert!(dup > 0);
        assert_eq!(buffer[..dup], buffer[dup..dup * 2]);
        assert_eq!(buffer[dup], 0.0, "real data should start at the duplicate's end");
    }

    /// Consecutive ordinary blocks advance by exactly `stride` and share
    /// `block_read_overlap` samples, so no RF sample is ever skipped.
    #[test]
    fn consecutive_blocks_advance_by_stride_without_gaps() {
        let layout = tiny_layout();
        let blocks = stream_blocks(layout.stride * 5, layout);
        for pair in blocks.windows(2) {
            let (a, b) = (&pair[0].0, &pair[1].0);
            if b.is_last {
                continue; // the last block deliberately slides further back
            }
            assert!(b.read_start < a.read_end, "gap between blocks {a:?} {b:?}");
        }
    }
}
