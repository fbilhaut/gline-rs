//! Relex decoding (span mode)

use composable::Composable;
use crate::model::output::tensors::TensorOutput;
use crate::model::output::decoded::{RelexOutput, SpanOutput};
use crate::model::pipeline::context::EntityContext;
use crate::text::span::Span;
use crate::util::result::Result;


const TENSOR_REL_LOGITS: &str = "rel_logits";
const TENSOR_REL_IDX: &str = "rel_idx";
const TENSOR_REL_MASK: &str = "rel_mask";
const MODEL_THRESHOLD_ON_ENTITIES: f32 = 0.5;


/// Decoding method for relex span mode.
pub struct TensorsToDecoded {
    span_decoder: super::span::TensorsToDecoded,
    relation_threshold: f32,
}

impl TensorsToDecoded {
    pub fn new(relation_threshold: f32, max_width: usize) -> Self {
        Self {            
            // we must not filter on probability yet, as we could loose spans that are referenced by relations. span filtering will occur in the end
            span_decoder: super::span::TensorsToDecoded::new(0.0, max_width),
            relation_threshold,
        }
    }

    pub fn outputs() -> [&'static str; 4] {
        [super::span::TensorsToDecoded::outputs()[0], TENSOR_REL_LOGITS, TENSOR_REL_IDX, TENSOR_REL_MASK]
    }

    // The model's select_span_target_embedding selects spans where
    // sigmoid(scores).max(class_axis) > threshold, then packs them to the front
    // using argsort(mask, descending=True).  rel_idx values reference positions
    // in this packed list.
    fn decode(&self, input: &TensorOutput) -> Result<Vec<Vec<Span>>> {
        // call regular span decoder (without threshold, see above)
        let span_result = self.span_decoder.decode(input)?;

        // compute max_entities (max number of entities for a sequence), num_pairs, and num_rels
        let max_entities = span_result.iter().map(|v| v.iter().filter(|s| s.probability() >= MODEL_THRESHOLD_ON_ENTITIES).count()).max().unwrap_or(0);
        let num_pairs = if max_entities > 0 { max_entities * (max_entities - 1) } else { 0 };
        let num_rels = input.context.relations.len();
        dbg!(num_pairs);
        
        if max_entities > 0 && num_rels > 0 && num_pairs > 0 {
            // get and check output tensors
            let logits = input.tensors.get(TENSOR_REL_LOGITS).ok_or("rel_logits not found in model output")?;
            self.check_logits_shape(logits.shape()?, &input.context, num_pairs)?;
            let logits = logits.try_extract_tensor::<f32>()?;

            let idx = input.tensors.get(TENSOR_REL_IDX).ok_or("rel_idx not found in model output")?;
            self.check_idx_shape(idx.shape()?, &input.context, num_pairs)?;
            let idx = idx.try_extract_tensor::<i64>()?;
            
            let mask = input.tensors.get(TENSOR_REL_MASK).ok_or("rel_mask not found in model output")?;
            self.check_mask_shape(mask.shape()?, &input.context, num_pairs)?;
            let mask = mask.try_extract_tensor::<bool>()?;
                        
            println!("LOGITS: {logits}");
            println!("IDX: {idx}");
            println!("MASK: {mask}");
            let entity_spans = input.tensors.get("entity_spans").ok_or("rel_idx not found in model output")?;
            let entity_spans = entity_spans.try_extract_tensor::<i64>()?;
            println!("ENTITY_SPANS: {entity_spans}");
            for span in &span_result {
                println!("SPANS: {span:?}");
            }

            let batch_size = input.context.texts.len();
            for sequence_id in 0..batch_size {
                println!("*** Batch {sequence_id}");
                // get a slice for the current sequence (1st dimension)
                let logits = logits.slice(ndarray::s![sequence_id, .., ..]);
                let idx = idx.slice(ndarray::s![sequence_id, .., ..]);
                let mask = mask.slice(ndarray::s![sequence_id, ..]);
                
                /*println!("{:?}", idx);
                println!("{:?}", mask);
                println!("{:?}", logits.map(|x| crate::util::math::sigmoid(*x)));*/

                // iterate on pairs
                for pair in 0..num_pairs {                    
                    if mask[pair] {
                        for rel in 0..num_rels {
                            let probability = logits.get((pair, rel)).unwrap(); // safe unwrap as shapes are checked
                            let probability = crate::util::math::sigmoid(*probability);
                            if probability >= self.relation_threshold {
                                let subject = idx.get((pair, 0)).unwrap(); // safe unwrap as shapes are checked
                                let object = idx.get((pair, 1)).unwrap(); // safe unwrap as shapes are checked
                                let rel = input.context.relations.get(rel).unwrap();
                                println!("{rel}: {subject}->{object} = {probability}");
                            }
                        }
                    }
                }
                //let num_tokens = input.context.tokens[sequence_id].len();
                //println!("{num_tokens}");
            }
        }
        
        
        
        // job's done
        Ok(span_result)
    }
    
    /// Checks coherence of the rel_logits shape, expecting (batch_size, num_pairs, num_rels)
    fn check_logits_shape(&self, actual_shape: Vec<i64>, context: &EntityContext, num_pairs: usize) -> Result<()> {
        let expected_shape = vec![context.texts.len() as i64, num_pairs as i64, context.relations.len() as i64];
        if actual_shape != expected_shape {
            Err(format!("unexpected rel_idx shape: {:?} (expected {:?})", actual_shape, expected_shape).into())
        }
        else {
            Ok(())
        }
    }

    /// Checks coherence of the rel_idx shape, expecting (batch_size, num_pairs, 2)
    fn check_idx_shape(&self, actual_shape: Vec<i64>, context: &EntityContext, num_pairs: usize) -> Result<()> {
        let expected_shape = vec![context.texts.len() as i64, num_pairs as i64, 2];
        if actual_shape != expected_shape {
            Err(format!("unexpected rel_idx shape: {:?} (expected {:?})", actual_shape, expected_shape).into())
        }
        else {
            Ok(())
        }
    }

    /// Checks coherence of the rel_mask shape, expecting (batch_size, num_pairs)
    fn check_mask_shape(&self, actual_shape: Vec<i64>, context: &EntityContext, num_pairs: usize) -> Result<()> {
        let expected_shape = vec![context.texts.len() as i64, num_pairs as i64];
        if actual_shape != expected_shape {
            Err(format!("unexpected rel_mask shape: {:?} (expected {:?})", actual_shape, expected_shape).into())
        }
        else {
            Ok(())
        }
    }

}


impl Composable<TensorOutput<'_>, RelexOutput> for TensorsToDecoded {
    fn apply(&self, input: TensorOutput) -> Result<RelexOutput> {        
        let decoded = self.decode(&input)?;
        Ok(RelexOutput::new(SpanOutput::new(input.context.texts, input.context.entities, decoded)))
    }
}