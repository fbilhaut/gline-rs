//! Data to be transmitted, beside the tensors themselves, from pre-processing to post-processing.

use std::collections::{HashMap, HashSet};
use crate::util::{error::IndexError, result::Result};
use crate::text::span::Span;
use crate::text::token::Token;


// Context for NER pipelines
pub struct EntityContext {
    pub texts: Vec<String>,
    pub tokens: Vec<Vec<Token>>,
    pub entities: Vec<String>,
    pub relations: Vec<String>,
    pub num_words: usize,
}

impl EntityContext {
    /// Creates a span given the necessary indexes and the tensor meta data.
    pub fn create_span(&self, sequence_id: usize, start_token: usize, end_token: usize, class: usize, probability: f32) -> Result<Span> {
        let sequence = self.tokens.get(sequence_id).ok_or(IndexError::new("meta.tokens", sequence_id))?;
        let start_token = sequence.get(start_token).ok_or(IndexError::new("meta.tokens[]", start_token))?;
        let start_offset = start_token.start();
        let end_token = sequence.get(end_token).ok_or(IndexError::new("meta.tokens[]", end_token))?;
        let end_offset = end_token.end();
        let text = &self.texts.get(sequence_id).ok_or(IndexError::new("meta.texts", sequence_id))?;
        let text = text[start_offset..end_offset].to_string();
        let class = self.entities.get(class).ok_or(IndexError::new("meta.entities", class))?.to_string();
        Ok(Span::new(
            sequence_id, 
            start_offset, 
            end_offset,
            text, 
            class, 
            probability)
        )
    }
}


// Context for RE pipeline
pub struct RelationContext {
    pub entity_labels: HashMap<String, HashSet<String>>
}
