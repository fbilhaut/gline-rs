//! Sort spans by offsets (which is expected by greedy-search)

use composable::Composable;
use crate::util::result::Result;
use super::{SpanOutput, RelexOutput};

#[derive(Default)] 
pub struct SpanSort {}


/// Composable: SpanOutput => SpanOutput
impl Composable<SpanOutput, SpanOutput> for SpanSort {
    fn apply(&self, input: SpanOutput) -> Result<SpanOutput> {      
        let mut spans = input.spans;
        for sequence in &mut spans {
            // "Unstable" sort (which is perfectly safe despite the name ;) is more efficient, and sufficient 
            // in our case as we don't need to preserve the initial order of equal elements. Also note that
            // calling `cmp()` on a tuple does exactly what we ant here (sort by start, then end, offsets).
            sequence.sort_unstable_by(|s1, s2| s1.offsets().cmp(&s2.offsets()));
        }
        Ok(SpanOutput::new(input.texts, input.entities, spans))
    }
}

/// Composable: RelexOutput => RelexOutput
impl Composable<RelexOutput, RelexOutput> for SpanSort {
    fn apply(&self, input: RelexOutput) -> Result<RelexOutput> {      
        let spans = SpanSort::apply(&self, input.spans)?;        
        Ok(RelexOutput::new(spans))
    }
}