//! Pre-defined pipeline for NER (token mode)

use std::collections::HashSet;
use std::path::Path;
use ::composable::*;
use orp::{pipeline::*, params::RuntimeParameters};
use crate::util::result::Result;
use super::super::super::text::{splitter::Splitter, tokenizer::Tokenizer};
use super::super::{input, output, params};
use super::context::EntityContext;


/// Generic token-level pipeline
pub struct TokenPipeline<S, T> {
    splitter: S,
    tokenizer: T,
    expected_inputs: HashSet<&'static str>,
    expected_outputs: HashSet<&'static str>,    
}

impl<'a, S: Splitter, T:Tokenizer> Pipeline<'a> for TokenPipeline<S, T> {
    type Input = input::text::TextInput;
    type Output = output::decoded::SpanOutput;
    type Context = EntityContext;
    type Parameters = params::Parameters;

    fn pre_processor(&self, params: &Self::Parameters) -> impl PreProcessor<'a, Self::Input, Self::Context> {
        composed![
            input::tokenized::RawToTokenized::new(&self.splitter, params.max_length),
            input::prompt::TokenizedToPrompt::default(),
            input::encoded::PromptsToEncoded::new(&self.tokenizer),
            input::tensors::token::EncodedToTensors::default(),
            input::tensors::token::TensorsToSessionInput::default()
        ]
    }

    fn post_processor(&self, params: &Self::Parameters) -> impl PostProcessor<'a, Self::Output, Self::Context> {
        composed![
            output::tensors::SessionOutputToTensors::default(),            
            output::decoded::token::TensorsToDecoded::new(params.threshold),
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
impl TokenPipeline<crate::text::splitter::RegexSplitter, crate::text::tokenizer::HFTokenizer> {
    pub fn new<P: AsRef<Path>>(tokenizer_path: P) -> Result<Self> {
        Ok(Self {
            splitter: crate::text::splitter::RegexSplitter::default(),
            tokenizer: crate::text::tokenizer::HFTokenizer::from_file(tokenizer_path)?,
            expected_inputs: input::tensors::token::TokenTensors::inputs().into_iter().collect(),
            expected_outputs: output::decoded::token::TensorsToDecoded::outputs().into_iter().collect(),
        })
    }
}

/// Shorthand for the default token pipeline type (eases disambiguation when calling `GLiNER::new`)
pub type TokenMode = TokenPipeline<crate::text::splitter::RegexSplitter, crate::text::tokenizer::HFTokenizer>;


/// Specific GLiNER implementation using the default token-mode pipeline
impl super::super::GLiNER<TokenMode> {
    pub fn new<P: AsRef<Path>>(params: params::Parameters, runtime_params: RuntimeParameters, tokenizer_path: P, model_path: P) -> Result<Self> {
        Ok(Self {
            params, 
            model: super::super::Model::new(model_path, runtime_params)?,
            pipeline: TokenPipeline::new(tokenizer_path)?,
        })
    }
}