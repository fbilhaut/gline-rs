//! Span decoding steps

pub mod span;
pub mod token;
pub mod token_flat;
pub mod greedy;

use crate::text::span::Span;

/// Represents the final output of the post-processing steps, as a list of spans for each input sequence
#[derive(Debug)]
pub struct SpanOutput {
    pub spans: Vec<Vec<Span>>,
}


impl SpanOutput {
    pub fn new(spans: Vec<Vec<Span>>) -> Self {
        Self {
            spans
        }
    }
}