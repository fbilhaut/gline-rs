//! Pre-defined pipeline for NER (span mode)

use std::collections::HashSet;
use std::path::Path;
use ::composable::*;
use orp::{pipeline::*, params::RuntimeParameters};
use crate::util::result::Result;
use super::super::super::text::{splitter::Splitter, tokenizer::Tokenizer};
use super::super::{input, output, params};
use super::context::EntityContext;


/// Generic span-level pipeline
pub struct SpanPipeline<S, T> {
    splitter: S,
    tokenizer: T,
    expected_inputs: HashSet<&'static str>,
    expected_outputs: HashSet<&'static str>,
}

impl<'a, S: Splitter, T:Tokenizer> Pipeline<'a> for SpanPipeline<S, T> {
    type Input = input::text::TextInput;
    type Output = output::decoded::SpanOutput;
    type Context = EntityContext;
    type Parameters = params::Parameters;

    fn pre_processor(&self, params: &Self::Parameters) -> impl PreProcessor<'a, Self::Input, Self::Context>{
        composed![
            input::tokenized::RawToTokenized::new(&self.splitter, params.max_length),
            input::prompt::TokenizedToPrompt::default(),
            input::encoded::PromptsToEncoded::new(&self.tokenizer),
            input::tensors::span::EncodedToTensors::new(params.max_width),
            input::tensors::span::TensorsToSessionInput::default()
        ]
    }

    fn post_processor(&self, params: &Self::Parameters) -> impl PostProcessor<'a, Self::Output, Self::Context> {
        composed![
            output::tensors::SessionOutputToTensors::default(),
            output::decoded::span::TensorsToDecoded::new(params.threshold, params.max_width),
            output::decoded::sort::SpanSort::default(),
            output::decoded::greedy::GreedySearch::new(params.flat_ner, params.dup_label, params.multi_label)
        ]
    }

    fn expected_inputs(&self, _params: &Self::Parameters) -> Option<impl Iterator<Item = &str>> {
        Some(self.expected_inputs.iter().copied())
    }

    fn expected_outputs(&self, _params: &Self::Parameters) -> Option<impl Iterator<Item = &str>> {
        Some(self.expected_outputs.iter().copied())
    }

}

/// Specific implementation using HF tokenizer and default splitter
impl SpanPipeline<crate::text::splitter::RegexSplitter, crate::text::tokenizer::HFTokenizer> {
    pub fn new<P: AsRef<Path>>(tokenizer_path: P) -> Result<Self> {
        Ok(Self {
            splitter: crate::text::splitter::RegexSplitter::default(),
            tokenizer: crate::text::tokenizer::HFTokenizer::from_file(tokenizer_path)?,
            expected_inputs: input::tensors::span::SpanTensors::inputs().into_iter().collect(),
            expected_outputs: output::decoded::span::TensorsToDecoded::outputs().into_iter().collect(),
        })
    }

    pub fn new_from_bytes(tokenizer_bytes: &[u8]) -> Result<Self> {
        Ok(Self {
            splitter: crate::text::splitter::RegexSplitter::default(),
            tokenizer: crate::text::tokenizer::HFTokenizer::from_bytes(tokenizer_bytes)?,
            expected_inputs: input::tensors::span::SpanTensors::inputs().into_iter().collect(),
            expected_outputs: output::decoded::span::TensorsToDecoded::outputs().into_iter().collect(),
        })
    }
}

/// Shorthand for the default span pipeline type (eases disambiguation when calling `GLiNER::new`)
pub type SpanMode = SpanPipeline<crate::text::splitter::RegexSplitter, crate::text::tokenizer::HFTokenizer>;


/// Specific GLiNER implementation using the default span-mode pipeline
impl super::super::GLiNER<SpanMode> {
    pub fn new<P: AsRef<Path>>(params: params::Parameters, runtime_params: RuntimeParameters, tokenizer_path: P, model_path: P) -> Result<Self> {
        Ok(Self {            
            model: super::super::Model::new(model_path, runtime_params)?,
            pipeline: SpanPipeline::new(tokenizer_path)?,
            params,
        })
    }

    pub fn new_from_bytes(params: params::Parameters, runtime_params: RuntimeParameters, tokenizer_bytes: &[u8], model_bytes: &[u8]) -> Result<Self> {
        Ok(Self {
            model: super::super::Model::new_from_bytes(model_bytes, runtime_params)?,
            pipeline: SpanPipeline::new_from_bytes(tokenizer_bytes)?,
            params,
        })
    }
}