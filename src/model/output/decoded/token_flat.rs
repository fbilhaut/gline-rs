//! Experimental alternative for the first step of span decoding (in token mode)

use std::iter;
use composable::Composable;
use crate::util::result::Result;
use crate::util::error::IndexError;
use crate::model::output::tensors::TensorOutput;
use crate::{model::pipeline::context::EntityContext, text::span::Span};
use crate::util::math::sigmoid;
use super::SpanOutput;


/// *Experimental* token decoding with a one-dimensional approach, working directly on a flat representation of 
/// the model output, with one padding by dimension to access appropriate value. Not very readable, but might 
/// be interresting from a performance standpoint. To be benchmarked, and checked for accurracy according to
/// the original implementation. In the meantime, prefer the `token.rs` which performs the same operation in 
/// a much more readable way, basing on the four-dimensional output tensor.
pub struct FlatTokenDecoder {
    threshold: f32,
}


impl FlatTokenDecoder {

    fn new(threshold: f32) -> Self {
        Self {
            threshold,
        }
    }

    fn decode(&self, model_output: &[f32], input: &EntityContext) -> Result<Vec<Vec<Span>>> {        
        let tokens = &input.tokens;
        let batch_size = tokens.len();
        let num_entities = input.entities.len();

        // compute paddings to navigate the flattened tensor
        let sequence_padding = input.num_words * num_entities;
        let position_padding = batch_size * sequence_padding;
        let token_padding = num_entities;

        // prepare the set of spans
        let mut spans: Vec<Vec<Span>> = iter::repeat_with(Vec::new).take(batch_size).collect();

        // iterate over the whole vector
        for start_idx in 0..position_padding {
            // check the start token score is above threshold, otherwise continue
            if sigmoid(Self::get(model_output, start_idx)?) < self.threshold {
                continue
            }

            // retrieve the appropriate indices
            let sequence_id = (start_idx / sequence_padding) % batch_size;
            let start_token = (start_idx / token_padding) % input.num_words;
            let class = start_idx % num_entities;

            // accumulators to compute the mean score of inside tokens
            let mut sum = 0f32;
            let mut count = 0usize;

            // iterate over end tokens
            let mut end_token = start_token;
            let mut end_idx = start_idx + position_padding;

            while (((end_idx / sequence_padding) % batch_size) == sequence_id) && (end_idx < 2 * position_padding) {
                // check the end token score is above threshold, otherwise continue
                if sigmoid(Self::get(model_output, end_idx)?) >= self.threshold {
                    // we won't consider a span at all if it contains a score below the threshold
                    let score = sigmoid(Self::get(model_output, end_idx + position_padding)?);
                    if score < self.threshold {
                        break
                    }
                    // consume next inside token and update the results
                    else {
                        // compute the actual probability (score) for the current span
                        sum += score;
                        count += 1;
                        let probability = sum / (count as f32);

                        // actually create the span
                        let span = input.create_span(sequence_id, start_token, end_token, class, probability)?;
                        spans.get_mut(sequence_id).ok_or(IndexError::new("flat token decode spans", sequence_id))?.push(span);
                    }
                }

                // proceed
                end_token += 1;
                end_idx += token_padding;
            }
        }

        Ok(spans)
    }

    #[inline] fn get(model_output: &[f32], index: usize) -> Result<f32> {
        Ok(*model_output.get(index).ok_or(IndexError::new("flat token decode model output", index))?)
    }
}


pub struct TensorsToDecoded {
    decoder: FlatTokenDecoder,
}

impl TensorsToDecoded {
    pub fn new(threshold: f32) -> Self {
        Self { 
            decoder: FlatTokenDecoder::new(threshold)
        }
    }
}

impl Composable<TensorOutput<'_>, SpanOutput> for TensorsToDecoded {
    fn apply(&self, input: TensorOutput) -> Result<SpanOutput> {        
        let logits = input.tensors.get("logits").ok_or("logits not found in model output")?;
        let (_shape, logits) = logits.try_extract_raw_tensor::<f32>()?;
        let spans = self.decoder.decode(logits, &input.context)?;        
        Ok(SpanOutput::new(input.context.texts, input.context.entities, spans))      
    }
}


#[cfg(test)]
mod tests {
    use super::*;

    /// An out-of-bounds index into the flat model-output slice must return Err
    /// from `get` instead of panicking. A valid index still returns Ok, so
    /// behavior is preserved on well formed input.
    #[test]
    fn get_out_of_bounds_returns_err() {
        let model_output = [0.1f32, 0.2, 0.3];
        assert!(FlatTokenDecoder::get(&model_output, 99).is_err());
        assert!(FlatTokenDecoder::get(&model_output, 1).is_ok());
    }
}
