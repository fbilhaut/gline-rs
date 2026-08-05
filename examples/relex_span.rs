use orp::params::RuntimeParameters;
use gliner::util::result::Result;
use gliner::model::{GLiNER, input::text::TextInput, params::Parameters};
use gliner::model::pipeline::relex_span::RelexSpanMode;

/// Sample usage of the public API in span mode
fn main() -> Result<()> {    
    
    println!("Loading model...");
    let model = GLiNER::<RelexSpanMode>::new(
        Parameters::default(),
        RuntimeParameters::default(),
        "models/gliner-relex-large-1.0/tokenizer.json",
        "models/gliner-relex-large-1.0/onnx/model.onnx",
    )?;

    let input = TextInput::from_str_with_relations(
        &[ 
            //"padding test",
            "John loves Mary",
            "Mary hates Bob and Jessica",
            //"James is a friend of Bill.",
            //"Bob works at Porsche. James Bond owns a beautiful Aston Martin. John works at Apple Inc.",
            //"Bob works at Porsche. James Bond lives in New-York and owns a beautiful Aston Martin.",
            //"James is a friend of Bill.",
            //"Bill lives in New-York",
        ],
        &[
            "person", 
            //"location",
            //"vehicle",
            //"company",
        ],
        &[
            //"own",
            //"work for",
            "love",
            "hate",
        ],
    )?;

    println!("Inferencing...");
    let output = model.inference(input)?;

    println!("Results:");
    for spans in output.spans.spans {
        for span in spans {
            println!("{:3} | {:16} | {:10} | {:.1}%", span.sequence(), span.text(), span.class(), span.probability() * 100.0);
        }
    }

    Ok(())

}
