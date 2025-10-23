use orp::params::RuntimeParameters;
use gliner::model::input;
use gliner::model::pipeline::token::TokenMode;
use gliner::util::result::Result;
use gliner::model::{GLiNER, params::Parameters};

#[cfg(feature = "memprof")] 
use gliner::util::memprof::*;

const REPEAT: usize = 100;
const MAX_SAMPLES: usize = 100;
const THREADS: usize = 12;
const CSV_PATH: &str = "data/nuner-sample-1k.csv";

fn main() -> Result<()> {
    let entities = [
        "person", 
        "location",
        "vehicle",
    ];

    println!("Loading data...");
    let input = input::text::TextInput::new_from_csv(CSV_PATH, 0, MAX_SAMPLES, entities.map(|x| x.to_string()).to_vec())?;
    let nb_samples = input.texts.len();
    
    println!("Loading model...");
    let model = GLiNER::<TokenMode>::new(
        Parameters::default(),
        RuntimeParameters::default().with_threads(THREADS),
        std::path::Path::new("models/gliner-multitask-large-v0.5/tokenizer.json"),
        std::path::Path::new("models/gliner-multitask-large-v0.5/onnx/model.onnx")
    )?;
    
    let global_inference_start = std::time::Instant::now();

    for i in 0..REPEAT {
        println!("Inferencing ({})...", i + 1);
        let inference_start = std::time::Instant::now();
        let _output = model.inference(input.clone())?;
        
        let inference_time = inference_start.elapsed();
        println!("Took {} seconds on {} samples ({:.2} samples/sec)", inference_time.as_secs(), nb_samples, nb_samples as f32 / inference_time.as_secs() as f32);

        #[cfg(feature = "memprof")] 
        print_memory_usage();
    }

    let global_inference_time = global_inference_start.elapsed();
    let global_nb_samples = nb_samples * REPEAT;
    println!("All {} inferences took {} seconds on {} samples total ({:.2} samples/sec)", REPEAT, global_inference_time.as_secs(), global_nb_samples, global_nb_samples as f32 / global_inference_time.as_secs() as f32);

    Ok(())
}
