use ort::session::SessionInputs;
use ort::value::Tensor;
use composable::Composable;
use crate::util::result::Result;
use super::super::encoded::EncodedInput;
use super::super::super::pipeline::context::EntityContext;


const TENSOR_INPUT_IDS: &str = "input_ids";
const TENSOR_ATTENTION_MASK: &str = "attention_mask";
const TENSOR_WORD_MASK: &str = "words_mask";
const TENSOR_TEXT_LENGTHS: &str = "text_lengths";
const TENSOR_SPAN_IDX: &str = "span_idx";
const TENSOR_SPAN_MASK: &str = "span_mask";


/// Ready-for-inference tensors (span mode)
pub struct SpanTensors<'a> {
    pub tensors: SessionInputs<'a, 'a>,
    pub context: EntityContext,    
}

impl SpanTensors<'_> {

    pub fn from(encoded: EncodedInput, max_width: usize) -> Result<Self> {
        let (span_idx, span_mask) = Self::make_spans_tensors(&encoded, max_width);
        let inputs = ort::inputs!{
            TENSOR_INPUT_IDS => Tensor::from_array(encoded.input_ids)?,
            TENSOR_ATTENTION_MASK => Tensor::from_array(encoded.attention_masks)?,
            TENSOR_WORD_MASK => Tensor::from_array(encoded.word_masks)?,
            TENSOR_TEXT_LENGTHS => Tensor::from_array(encoded.text_lengths)?,
            TENSOR_SPAN_IDX => Tensor::from_array(span_idx)?,
            TENSOR_SPAN_MASK => Tensor::from_array(span_mask)?,
        };
        Ok(Self {
            tensors: inputs.into(),
            context: EntityContext { 
                texts: encoded.texts, 
                tokens: encoded.tokens, 
                entities: encoded.entities, 
                num_words: encoded.num_words 
            },            
        })
    }

    pub fn inputs() -> [&'static str; 6] {
        [TENSOR_INPUT_IDS, TENSOR_ATTENTION_MASK, TENSOR_WORD_MASK, TENSOR_TEXT_LENGTHS, TENSOR_SPAN_IDX, TENSOR_SPAN_MASK]
    }

    /// Expected tensor for num_words=4 and max_width=12:
    /// ```text
    /// start, end, mask
    /// 0, 0, true
    /// 0, 1, true
    /// 0, 2, true
    /// 0, 3, true
    /// 0, 0, false
    /// [...until we have all the 12 spans for this token]
    /// 1, 1, true
    /// 1, 2, true
    /// 1, 3, true
    /// 0, 0, false
    /// [...]
    /// 2, 2, true
    /// 2, 3, true
    /// 0, 0, false
    /// [...]
    /// 3, 3, true
    /// 0, 0, false
    /// [...]
    /// ```    
    fn make_spans_tensors(encoded: &EncodedInput, max_width: usize) -> (ndarray::Array3<i64>, ndarray::Array2<bool>) {
        // total number of spans for each sequence: at most num_words * max_width
        let num_spans = encoded.num_words * max_width;
        
        // prepare output tensors (zero-filled, values will be set in place)
        let mut span_idx = ndarray::Array::zeros((encoded.texts.len(), num_spans, 2));
        let mut span_mask = ndarray::Array::from_elem((encoded.texts.len(), num_spans), false);

        // iterate over segments
        for s in 0..encoded.texts.len() {
            // get the actual width of the current segment
            let text_width = *encoded.text_lengths.get((s, 0)).unwrap() as usize;

            // repeat for each start offset in [0;text_width]
            for start in 0..text_width {          
                // remaining width from start offset
                let remaining_width = text_width - start;
                // the maximum span width is no more than remaining width in the sequence, or maximum span width
                let actual_max_width = std::cmp::min(max_width, remaining_width);
                // repeat for each possible width in between
                for width in 0..actual_max_width {
                    // retrieve the appropriate dimension on the second axis
                    let dim = start * max_width + width;
                    // fill the tensors in place
                    span_idx[[s, dim, 0]] = start as i64; // start offset
                    span_idx[[s, dim, 1]] = (start + width) as i64; // end offset
                    span_mask[[s, dim]] = true; // mask
                }
            }
        }

        // return both tensors
        (span_idx, span_mask)
    }

}


/// Composable: Encoded => SpanTensors
pub struct EncodedToTensors { 
    max_width: usize,
}

impl EncodedToTensors {
    pub fn new(max_width: usize) -> Self { 
        Self { max_width } 
    }
}

impl<'a> Composable<EncodedInput, SpanTensors<'a>> for EncodedToTensors {
    fn apply(&self, input: EncodedInput) -> Result<SpanTensors<'a>> {
        SpanTensors::from(input, self.max_width)
    }
}


/// Composable: SpanTensors => (SessionInput, EntityContext) 
#[derive(Default)]
pub struct TensorsToSessionInput { 
}


impl<'a> Composable<SpanTensors<'a>, (SessionInputs<'a, 'a>, EntityContext)> for TensorsToSessionInput {
    fn apply(&self, input: SpanTensors<'a>) -> Result<(SessionInputs<'a, 'a>, EntityContext)> {
        Ok((input.tensors, input.context))
    }
}


/// Unit tests
#[cfg(test)]
mod tests {
    use ort::session::SessionInputValue;
    use super::*;

    #[test]
    fn test() -> Result<()> {        
        // Silent some clippy warnings for unit tests
        #![allow(clippy::get_first)]
        #![allow(clippy::unwrap_used)]
        // Processing
        let splitter = crate::text::splitter::RegexSplitter::default();        
        let tokenizer = crate::text::tokenizer::HFTokenizer::from_file(std::path::Path::new("models/gliner_small-v2.1/tokenizer.json"))?;
        let batch = [ "My name is James Bond", "I like to drive my Aston Martin"];
        let entities = [ "movie character", "vehicle" ];
        let input = super::super::super::text::TextInput::from_str(&batch, &entities)?;
        let tokenized = super::super::super::tokenized::TokenizedInput::from(input, &splitter, None)?;
        let prepared = super::super::super::prompt::PromptInput::from(tokenized);
        let encoded = EncodedInput::from(prepared, &tokenizer)?;
        let spans = SpanTensors::from(encoded, 12)?;
        let span_idx = get_tensor("span_idx", &spans.tensors)?;
        let span_idx = span_idx.try_extract_array::<i64>()?;
        let span_masks = get_tensor("span_mask", &spans.tensors)?;
        let span_masks = span_masks.try_extract_array::<bool>()?;
        // Some prints
        if false {
            println!("Spans: {:?}", &span_idx);
            println!("Spans Masks: {:?}", &span_masks);
        }
        // Assertions (TODO: add more)
        assert_eq!(span_idx.shape(), &[2, 84, 2]);
        assert_eq!(span_masks.shape(), &[2, 84]);
        // Everything rules
        Ok(())
    }

    fn get_tensor<'a>(key: &str, si: &'a SessionInputs<'a, 'a>) -> Result<&'a SessionInputValue<'a>> {
        if let SessionInputs::ValueMap(map) = si {
            for (k, v) in map {
                if k.eq(key) {
                    return Ok(v);
                }
            }
        }
        Err("cannot extract expected tensor".into())
    }

}