//! First step of span decoding (in token mode)

use composable::Composable;
use crate::util::result::Result;
use crate::util::error::IndexError;
use crate::util::math::sigmoid;
use crate::text::span::Span;
use crate::model::pipeline::context::EntityContext;
use crate::model::output::tensors::TensorOutput;
use super::SpanOutput;


const TENSOR_LOGITS: &str = "logits";


/// Decoding method for token mode.
/// 
/// From the related (GLiNER multi-task) [paper](https://arxiv.org/pdf/2406.12925v1): 
/// 
/// > The function `ϕi(k,c)` is the inside score for the token at position `k` within the span, 
/// > indicating the likelihood of the token being part of the entity class `c`. By averaging
/// > these scores over the span, we obtain a measure of how well the span fits the token class `c`.
/// 
/// Rq: greedy search must be applied the same way as in span mode (shall be called as a subsequent 
/// step in the pipeline).
pub struct TensorsToDecoded {
    threshold: f32,
}

impl TensorsToDecoded {
    pub fn new(threshold: f32) -> Self {
        Self { 
            threshold,
        }
    }

    pub fn outputs() -> [&'static str; 1] {
        [TENSOR_LOGITS]
    }

    fn decode(&self, input: &TensorOutput) -> Result<Vec<Vec<Span>>> {
        // prepare output vector
        let batch_size = input.context.texts.len();
        let mut result: Vec<Vec<Span>> = std::iter::repeat_with(Vec::new).take(batch_size).collect();

        // look for logits and check its shape
        let logits = input.tensors.get(TENSOR_LOGITS).ok_or("logits not found in model output")?;
        self.check_shape(logits.shape(), &input.context)?;
    
        // extract the actual array
        let array  = logits.try_extract_array::<f32>()?;
        //println!("{:?}", array.map(|x| crate::util::math::sigmoid(*x)));

        // iterate over sequences
        for sequence_id in 0..batch_size {
            // get a slice for the current sequence (2nd dimension)
            let scores = array.slice(ndarray::s![.., sequence_id, .., ..]);

            // get slices for start, end, and inside scores (1st dimension)
            let scores_start = scores.slice(ndarray::s![0, .., ..]);
            let scores_end = scores.slice(ndarray::s![1, .., ..]);
            let scores_inside = scores.slice(ndarray::s![2, .., ..]);            

            // generate all possible spans and iterate over them
            for span in self.generate_spans(&scores_start, &scores_end)? {
                // compute score
                let score = self.compute_span_score(span, &scores_inside)?;
                // reject span if score is below threshold
                if score < self.threshold {
                    continue
                }                
                // create actual span
                let (start_token, end_token, class) = span;
                let span = input.context.create_span(sequence_id, start_token, end_token, class, score)?;
                result.get_mut(sequence_id).ok_or(IndexError::new("token decode result", sequence_id))?.push(span);
            }
        }
        Ok(result)
    }


    /// Generates all possible `(i,j,c)` spans where:
    /// * `i <= j`
    /// * `score(i) >= threshold`
    /// * `score(j) >= threshold`.
    /// * `c` == `class(i) == class(j)`
    fn generate_spans(&self, scores_start: &ndarray::ArrayView2::<f32>, scores_end: &ndarray::ArrayView2::<f32>) -> Result<Vec<(usize, usize, usize)>> {
        // Return Err on a malformed model output instead of asserting, so the
        // caller gets a clean error rather than a panic.
        if scores_start.dim() != scores_end.dim() {
            return Err(IndexError::with("token decode: start/end score dimensions differ").into());
        }
        let (num_tokens, num_classes) = scores_start.dim();
        let mut result = Vec::new();
        for class in 0..num_classes {
            for start in 0..num_tokens {            
                let score_start = sigmoid(*scores_start.get((start, class)).ok_or(IndexError::with("token decode: start score index out of bounds"))?);
                if score_start < self.threshold {
                    continue
                }
                for end in start..num_tokens {
                    let score_end = sigmoid(*scores_end.get((end, class)).ok_or(IndexError::with("token decode: end score index out of bounds"))?);
                    if score_end < self.threshold {
                        continue
                    }
                    result.push((start, end, class));
                }
            }
        }
        Ok(result)
    }


    /// Computes the score of a span, defined as the mean of the inside scores (see above).
    /// Spans with one or more inside scores below the threshold will return a zero score, 
    /// since they should be discarded.
    fn compute_span_score(&self, span: (usize, usize, usize), scores_inside: &ndarray::ArrayView2<f32>) -> Result<f32> {
        let (start, end, class) = span;
        // Return Err on a malformed model output instead of asserting, so the
        // caller gets a clean error rather than a panic.
        if end < start {
            return Err(IndexError::with("token decode: span end precedes start").into());
        }
        let mut sum = 0f32;
        for i in start..end+1 {
            let score_inside = sigmoid(*scores_inside.get((i, class)).ok_or(IndexError::with("token decode: inside score index out of bounds"))?);
            if score_inside < self.threshold {
                return Ok(0.);
            }
            sum += score_inside;
        }
        Ok(sum / ((end - start + 1) as f32))
    }
    

    /// Checks coherence of the output shape.
    /// Expected shape is (3, batch_size, num_words, num_classes).
    /// The first dimension is related to `start`, `end` and `inside` positions in that order.
    fn check_shape(&self, actual_shape: &[i64], context: &EntityContext) -> Result<()> {
        let expected_shape = vec![3, context.texts.len() as i64, context.num_words as i64, context.entities.len() as i64];
        if actual_shape != expected_shape.as_slice() {
            Err("unexpected logits shape".into())
        }
        else {
            Ok(())
        }
    }

}

impl Composable<TensorOutput<'_>, SpanOutput> for TensorsToDecoded {
    fn apply(&self, input: TensorOutput) -> Result<SpanOutput> {        
        let decoded = self.decode(&input)?;
        Ok(SpanOutput::new(input.context.texts, input.context.entities, decoded))
    }
}


#[cfg(test)]
mod tests {
    use super::*;

    /// A start/end score tensor shape mismatch (a model output-shape invariant
    /// break) must return Err from `generate_spans` instead of panicking on a
    /// dimension assertion.
    #[test]
    fn generate_spans_rejects_mismatched_dims() {
        let decoder = TensorsToDecoded::new(0.5);
        let scores_start = ndarray::Array2::<f32>::zeros((2, 3));
        let scores_end = ndarray::Array2::<f32>::zeros((3, 2));
        let result = decoder.generate_spans(&scores_start.view(), &scores_end.view());
        assert!(result.is_err(), "mismatched start/end dims must return Err, not panic");
    }

    /// A span whose end precedes its start must return Err from
    /// `compute_span_score` instead of panicking.
    #[test]
    fn compute_span_score_rejects_end_before_start() {
        let decoder = TensorsToDecoded::new(0.5);
        let scores_inside = ndarray::Array2::<f32>::zeros((4, 2));
        let result = decoder.compute_span_score((3, 1, 0), &scores_inside.view());
        assert!(result.is_err(), "span end before start must return Err, not panic");
    }

    /// Positive path: well formed start/end/inside score tensors whose sigmoids
    /// sit above the threshold decode end to end - `generate_spans` returns
    /// exactly the expected `(start, end, class)` tuples, and
    /// `compute_span_score` returns `Ok(mean-of-inside)`, proving behavior is
    /// preserved on valid input.
    ///
    /// Construction (threshold 0.4, sigmoid(x) = 1 / (1 + e^-x)):
    ///   * start logits `[10.0, -10.0]` -> sigmoids ~0.99995 / ~0.00005, so only
    ///     token 0 clears the threshold as a span start;
    ///   * end logits `[-10.0, 10.0]` -> only token 1 clears it as a span end;
    ///     the sole admissible span is therefore `(start = 0, end = 1, class = 0)`;
    ///   * inside logits `[ln 3, 0.0]` -> sigmoids exactly 0.75 and 0.5 (both above
    ///     0.4, so the below-threshold zero short-circuit does not fire), whose
    ///     mean is `(0.75 + 0.5) / 2 = 0.625`.
    #[test]
    fn generate_spans_and_score_positive_path() {
        let decoder = TensorsToDecoded::new(0.4);

        // (num_tokens, num_classes) = (2, 1), row-major: element (token, class).
        // Only start-token 0 and end-token 1 clear the threshold.
        let scores_start =
            ndarray::Array2::<f32>::from_shape_vec((2, 1), vec![10.0, -10.0]).unwrap();
        let scores_end =
            ndarray::Array2::<f32>::from_shape_vec((2, 1), vec![-10.0, 10.0]).unwrap();

        let spans = decoder
            .generate_spans(&scores_start.view(), &scores_end.view())
            .expect("valid start/end tensors must decode without error");

        assert_eq!(
            spans,
            vec![(0, 1, 0)],
            "the sole threshold-clearing (start, end, class) span must be (0, 1, 0)"
        );

        // sigmoid(ln 3) = 3 / (1 + 3) = 0.75; sigmoid(0) = 0.5; mean = 0.625.
        let scores_inside =
            ndarray::Array2::<f32>::from_shape_vec((2, 1), vec![3.0_f32.ln(), 0.0]).unwrap();

        let score = decoder
            .compute_span_score(spans[0], &scores_inside.view())
            .expect("an all-above-threshold span must score without error");

        assert!(
            (score - 0.625).abs() < 1e-5,
            "span score must be the mean of the inside sigmoids (0.75 + 0.5) / 2 = 0.625, got {}",
            score
        );
    }
}