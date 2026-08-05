use composable::Composable;
use crate::util::result::Result;
use crate::text::{prompt::Prompt, token::Token};
use super::tokenized::TokenizedInput;


/// Prepared prompts, appending entity and text tokens. 
/// 
/// Output form: 
/// ```text
/// [<<ENT>>, type1, <<ENT>>, type2, ..., <<ENT>>, typeK, <<SEP>>, token1, token2, ..., tokenN]
/// ```
pub struct PromptInput {
    /// Texts (moved from input)
    pub texts: Vec<String>,
    /// Tokens (moved from input)
    pub tokens: Vec<Vec<Token>>,
    /// Entities (moved from input)
    pub entities: Vec<String>,
    /// Relations (moved from input)
    pub relations: Vec<String>,
    /// Number of tokens of the text part for each prompt
    pub text_lengths: Vec<usize>,
    /// Maximum number of words in a prompt excluding entities (number of tokens in the largest sequence in the batch)
    pub num_words: usize,
    /// The actual prompts
    pub prompts: Vec<Prompt>,    
}


const ENTITY_TOKEN: &str = "<<ENT>>";
const RELATIONS_TOKEN: &str = "<<REL>>";
const SEP_TOKEN: &str = "<<SEP>>";


impl PromptInput {

    pub fn from(input: TokenizedInput) -> Self {
        // prepare the entities/relations part of the prompt (will be copied into each actual prompt)
        let additional_prompt = Self::additional_prompt(&input);
        // the text lengths for each sequence (number of actual tokens beside the entities part)
        let mut text_lengths = Vec::<usize>::new();
        // the maximum number of words in a prompt excluding entities (number of tokens in the largest sequence in the batch)
        let mut num_words = 0;
        // the actual prompts that will be created for each token sequence
        let mut prompts = Vec::new();    
        
        // iterate over each sequence of tokens
        for tokens in &input.tokens {
            // prepare the sequence of tokens for this prompt
            let mut prompt = Vec::with_capacity(additional_prompt.len() + tokens.len());
            // copy the entities part
            prompt.extend(additional_prompt.clone());
            // append each text token of the current sequence
            prompt.extend(tokens.iter().map(|token| token.text().to_string()));
            // update output data
            prompts.push(Prompt::new(prompt, tokens.len(), additional_prompt.len()));        
            text_lengths.push(tokens.len());
            num_words = std::cmp::max(num_words, tokens.len());
    
        }
    
        // job's done
        Self {
            texts: input.texts,
            tokens: input.tokens,
            entities: input.entities,
            relations: input.relations,
            text_lengths,
            num_words,
            prompts,            
        }
    
    }

    /// Create the entities/relations part of the prompt.
    fn additional_prompt(input: &TokenizedInput) -> Vec<String> {
        let mut result = Self::additional_tokens(&input.entities, ENTITY_TOKEN);
        result.append(&mut Self::additional_tokens(&input.relations, RELATIONS_TOKEN));
        result
    }

    fn additional_tokens(items: &Vec<String>, prefix_token: &str) -> Vec<String> {
        let mut result = Vec::with_capacity(items.len() * 2 + 1);
        for item in items {
            result.push(prefix_token.to_string());
            result.push(item.clone());
        }
        if !result.is_empty() {
            result.push(SEP_TOKEN.to_string());
        }
        result
    }

}


/// Composable: Tokenized => Prompt
#[derive(Default)]
pub struct TokenizedToPrompt { 
}


impl Composable<TokenizedInput, PromptInput> for TokenizedToPrompt {
    fn apply(&self, input: TokenizedInput) -> Result<PromptInput> {
        Ok(PromptInput::from(input))
    }
}

/// Unit tests
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test() -> Result<()> {
        // Silent some clippy warnings for unit tests
        #![allow(clippy::get_first)]
        #![allow(clippy::unwrap_used)]
        // Processing
        let splitter = crate::text::splitter::RegexSplitter::default();
        let batch = [ "This is a text !", "This is a longer one."];
        let entities = [ "Person", "Place" ];
        let input = super::super::text::TextInput::from_str(&batch, &entities)?;
        let tokenized = super::super::tokenized::TokenizedInput::from(input, &splitter, None)?;
        let prepared = PromptInput::from(tokenized);        
        // Assertions
        assert_eq!(prepared.prompts.len(), 2);
        let prompt1 = prepared.prompts.get(0).unwrap();
        let prompt2 = prepared.prompts.get(1).unwrap();
        assert_eq!(prompt1.tokens().len(), 10);
        assert_eq!(prompt2.tokens().len(), 11);        
        assert_eq!(prompt1.text_len(), 5);
        assert_eq!(prompt2.text_len(), 6);
        assert_eq!(prompt1.entities_len(), prompt2.entities_len());
        assert_eq!(prompt1.tokens().get(4).unwrap(), "<<SEP>>"); 
        assert_eq!(prompt2.tokens().get(5).unwrap(), "This"); 
        assert_eq!(prompt2.tokens().get(1).unwrap(), entities[0]);
        assert_eq!(prompt2.tokens().get(3).unwrap(), entities[1]);
        assert_eq!(prepared.num_words, prompt2.text_len()); // second prompt has the most tokens     
        assert_eq!(prepared.text_lengths.len(), 2);
        assert_eq!(*prepared.text_lengths.get(0).unwrap(), prompt1.text_len());
        assert_eq!(*prepared.text_lengths.get(1).unwrap(), prompt2.text_len());
        // Everything rules
        Ok(())
    }
}