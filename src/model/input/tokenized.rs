use composable::Composable;
use crate::util::result::Result;
use crate::text::token::Token;
use crate::text::splitter::Splitter;
use super::text::TextInput;


/// Represents the output of the word-level segmentation
pub struct TokenizedInput {
    /// Tokens
    pub tokens: Vec<Vec<Token>>,
    /// Original sequences
    pub texts: Vec<String>,    
    /// Original entities
    pub entities: Vec<String>, 
    /// Original relations
    pub relations: Vec<String>, 
}


impl TokenizedInput {

    pub fn from(input: TextInput, splitter: &impl Splitter, max_length: Option<usize>) -> Result<Self> {
        // leverage the given `Splitter` to tokenize each input sequence
        let mut tokens = Vec::with_capacity(input.texts.len());
        for s in &input.texts {
            tokens.push(splitter.split(s, max_length)?);
        }

        Ok(Self {
            tokens,
            texts: input.texts,            
            entities: input.entities,
            relations: input.relations,
        })
    }
}

/// Composable: Text => Tokenized
pub struct RawToTokenized<'a, S> { 
    splitter: &'a S,
    max_length: Option<usize>
}

impl<'a, S> RawToTokenized<'a, S> {
    pub fn new(splitter: &'a S, max_length: Option<usize>) -> Self {
        Self { 
            splitter, 
            max_length
        }
    }
}

impl<S: Splitter> Composable<TextInput, TokenizedInput> for RawToTokenized<'_, S> {
    fn apply(&self, input: TextInput) -> Result<TokenizedInput> {
        TokenizedInput::from(input, self.splitter, self.max_length)
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
        let batch = [ "This is a text", "This is another one"];
        let entities = [ "person", "place" ];
        let input = TextInput::from_str(&batch, &entities)?;
        let tokenized = TokenizedInput::from(input, &splitter, None)?;
        // Some prints
        if false {
            println!("{:?}", tokenized.tokens);
        }
        // Assertions
        assert_eq!(tokenized.tokens.len(), 2);
        assert_eq!(tokenized.tokens.get(0).unwrap().len(), 4);
        assert_eq!(tokenized.tokens.get(0).unwrap().get(0).unwrap().text(), "This");
        assert_eq!(tokenized.tokens.get(0).unwrap().get(0).unwrap().start(), 0);
        assert_eq!(tokenized.tokens.get(1).unwrap().len(), 4);
        assert_eq!(tokenized.tokens.get(1).unwrap().get(3).unwrap().text(), "one");
        assert_eq!(tokenized.tokens.get(1).unwrap().get(3).unwrap().end(), batch[1].len());
        // Everything rules
        Ok(())
    }
}