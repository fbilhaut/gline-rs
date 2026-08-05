use std::path::Path;
use crate::util::result::Result;

/// Represents the raw text input, as a list of text chunks and a list of entity classes
#[derive(Debug, Clone)]
pub struct TextInput {
    pub texts: Vec<String>,
    pub entities: Vec<String>,
    pub relations: Vec<String>,
}


impl TextInput {

    /// Default constructor that moves the input data given as a vector of the text 
    /// sequences to be analyzed, and a vector of entity classes.
    pub fn new(texts: Vec<String>, entities: Vec<String>) -> Result<Self> {
        Self::new_with_relations(texts, entities, vec![])
    }

    /// Default constructor that moves the input data given as a vector of the text 
    /// sequences to be analyzed, and a vector of entity classes.
    pub fn new_with_relations(texts: Vec<String>, entities: Vec<String>, relations: Vec<String>) -> Result<Self> {
        if texts.is_empty() || entities.is_empty() {
            Err("invalid input: empty texts and/or entities".into())
        }
        else {
            Ok(Self { texts, entities, relations })
        }
    }

    /// This constructor will mostly be used to test with plain arrays of static `str`s.
    pub fn from_str(texts: &[&str], entities: &[&str]) -> Result<Self> {
        Self::from_str_with_relations(texts, entities, &[])
    }

    /// This constructor will mostly be used to test with plain arrays of static `str`s.
    pub fn from_str_with_relations(texts: &[&str], entities: &[&str], relations: &[&str]) -> Result<Self> {
        Self::new_with_relations(
            texts.iter().map(|s| s.to_string()).collect(),
            entities.iter().map(|s| s.to_string()).collect(),
            relations.iter().map(|s| s.to_string()).collect(),
        )
    }

    /// For testing purposes. 
    /// Panics if the specified column does not exist
    pub fn new_from_csv<P: AsRef<Path>>(path: P, column: usize, limit: usize, entities: Vec<String>) -> Result<Self> {
        let mut csv = csv::Reader::from_path(path)?;
        let texts: Vec<String> = csv.records()
            .take(limit)
            .map(|r| r.unwrap().get(column).unwrap().to_string())
            .collect();
        Self::new(texts, entities)
    }

}