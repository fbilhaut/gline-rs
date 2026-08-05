//! Greedy-search is the second step of span decoding

use composable::Composable;
use crate::util::result::Result;
use crate::text::span::Span;
use super::{SpanOutput, RelexOutput};

/// Greedy decoding implementation.
/// 
/// See section 4.2 of <https://aclanthology.org/2022.umios-1.1.pdf>
pub struct GreedySearch {
    flat_ner: bool,
    dup_label: bool,
    multi_label: bool,
}

impl GreedySearch {
    /// Creates a new greedy-search performer
    /// 
    /// Arguments:
    /// * `flat_ner`: if `true`, a span is not allowed not embed another one
    /// * `multi_label`: if `true`, the same span can belong to multiple classes
    pub fn new(flat_ner: bool, dup_label: bool, multi_label: bool) -> Self {
        Self { flat_ner, dup_label, multi_label }
    }

    /// Perform greedy search
    /// 
    /// Note: spans are supposed to be sorted by start, and then end, offsets.
    pub fn search(&self, spans: &[Span]) -> Vec<Span> {
        if spans.is_empty() {
            return Vec::new();
        }

        let mut result = Vec::with_capacity(spans.len());
        let mut prev = 0usize;
        let mut next = 1usize;

        while next < spans.len() {
            let prev_span = spans.get(prev).unwrap();
            let next_span = spans.get(next).unwrap();
            if self.accept(prev_span, next_span) {
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

    /// Returns `true` iif the span is valid wrt. the provided flags.
    /// 
    /// Namely:
    /// * All overlapping spans are forbiden if `flat_ner` is `true`.
    /// * Otherwise, `dup_label=true` allows for overlapping spans with the *same* label.
    /// * And `multi_label=true` allows for overlapping spans with *different* labels.
    /// 
    /// The checks are only relative to the previous span, as we expect them to be sorted by offset.    
    fn accept(&self, s1: &Span, s2: &Span) -> bool {                
        // if there is no overlap, we accept immediately
        if s1.is_disjoint(s2) { true }
        // otherwise, if `flat_ner=true` we reject immediately
        else if self.flat_ner { false }
        // otherwise, if the labels are the same, we accept overlaps only if `dup_label=true`
        else if !self.dup_label && s1.class().eq(s2.class()) { false }
        // otherwise the labels are different, so we accept overlaps only if `multi_label=true`
        else { self.multi_label }
    }
}


/// Composable: SpanOutput => SpanOutput
impl Composable<SpanOutput, SpanOutput> for GreedySearch {
    fn apply(&self, input: SpanOutput) -> Result<SpanOutput> {        
        let spans = input.spans
            .iter()
            .map(|s| self.search(s))
            .collect();
        Ok(SpanOutput::new(input.texts, input.entities, spans))
    }
}


/// Composable: RelexOutput => RelexOutput
impl Composable<RelexOutput, RelexOutput> for GreedySearch {
    fn apply(&self, input: RelexOutput) -> Result<RelexOutput> {      
        let spans = GreedySearch::apply(&self, input.spans)?;        
        Ok(RelexOutput::new(spans))
    }
}