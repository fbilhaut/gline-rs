//! Greedy-search is the second step of span decoding

use composable::Composable;
use crate::util::result::Result;
use crate::text::span::Span;
use super::SpanOutput;

/// Greedy decoding implementation.
/// 
/// See section 4.2 of <https://aclanthology.org/2022.umios-1.1.pdf>
pub struct GreedySearch {
    flat_ner: bool,
    multi_label: bool,
}

impl GreedySearch {
    /// Creates a new greedy-search performer
    /// 
    /// Arguments:
    /// * `flat_ner`: if `true`, a span is not allowed not embed another one
    /// * `multi_label`: if `true`, the same span can belong to multiple classes
    pub fn new(flat_ner: bool, multi_label: bool) -> Self {
        Self { flat_ner, multi_label }
    }

    /// Perform greedy search
    /// 
    /// Note: spans are supposed to be sorted by start, and then end, offsets.
    pub fn search(spans: &[Span], flat_ner: bool, multi_label: bool) -> Vec<Span> {
        if spans.is_empty() {
            return Vec::new();
        }

        let mut result = Vec::with_capacity(spans.len());
        let mut prev = 0usize;
        let mut next = 1usize;

        while next < spans.len() {
            let prev_span = spans.get(prev).unwrap();
            let next_span = spans.get(next).unwrap();
            if !Self::overlapping(prev_span, next_span, flat_ner, multi_label) {
                result.push(prev_span.clone());
                prev = next;
            }
            else if prev_span.probability() < next_span.probability() {
                prev = next
            }
            next += 1;
        }

        let prev_span = spans.get(prev).unwrap();
        result.push(prev_span.clone());

        result
    }

    /// Returns `true` iif:
    /// * one span overlaps the other one
    /// * and the two spans don't share the same offsets, unless `multi_label` is set to `true`
    /// * and neither span is nested in into another, unless `flat_ner` is set to `false`
    fn overlapping(s1: &Span, s2: &Span, flat_ner: bool, multi_label: bool) -> bool {
        if s1.same_offsets(s2) { !multi_label }
        else if flat_ner && (s1.is_nested_in(s2) || s2.is_nested_in(s1)) { true }
        else { s1.overlaps(s2) }
    }
}


/// Composable: SpanOutput => SpanOutput
impl Composable<SpanOutput, SpanOutput> for GreedySearch {
    fn apply(&self, input: SpanOutput) -> Result<SpanOutput> {        
        let spans = input.spans
            .iter()
            .map(|s| Self::search(s, self.flat_ner, self.multi_label))
            .collect();
        Ok(SpanOutput::new(input.texts, input.entities, spans))
    }
}