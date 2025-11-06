[![Crates.io](https://img.shields.io/crates/v/gline-rs?logo=rust&style=flat-square&logoColor=white "Crates.io")](https://crates.io/crates/gline-rs)

# 🌿 gline-rs: Inference Engine for GLiNER Models, in Rust

## 💬 Introduction

`gline-rs` is an inference engine for [GLiNER](https://github.com/urchade/GLiNER) models. These language models proved to be efficient at **zero-shot** [Named Entity Recognition](https://paperswithcode.com/task/cg) (NER) and other tasks such as [Relation Extraction](https://paperswithcode.com/task/relation-extraction), while consuming less resources than large generative models (LLMs).

This implementation has been written from the ground up in Rust, and supports both span- and token-oriented variants (for inference only). It is based on a flexible pipeline architecture with ready-to-use implementations for NER and Relation Extraction.

It aims to provide a production-grade and user-friendly API in a modern and safe programming language, including a clean and maintainable implementation of the mechanics surrounding these models.

For those interested, it can also help getting a deep understanding of GLiNER's operation.

## 📌 Latest News 

### 🎉 Version `1.0.0` is out !  
Following extensive testing across various environments, `gline-rs` is now considered ready for production!

### ⚠️ Broken builds with `0.9.x`

Due to missing pins in `ort` prior to the rc.10 release (see [this issue](https://github.com/pykeio/ort/issues/399)), some builds may have broken recently with `gline-rs` 0.9.4 and earlier. Version 1.0.0 prevents this by explicitly pinning the transitive dependency (until it adopts `ort` rc.10).

## 💡 Background and Motivation

Common drawbacks of machine learning systems include cryptic implementations and high resource consumption. `gline-rs` aims to take a step toward a more maintainable and sustainable approach. 🌱

**Why [GLiNER](https://github.com/urchade/GLiNER/tree/main)?** 

The term stands for "Generalist and Lightweight Model for Named Entity Recognition", after an original work from [Zaratiana at al.](https://arxiv.org/abs/2311.08526) It now refers to a family of lightweight models capable of performing various zero-shots extractions using a bidirectional transformer architecture (BERT-like). For this kind of tasks, this approach can be much more relevant than full-blown LLMs.

However, it is characterized by a number of operations that need to be performed both upstream and downstream of applying the pre-trained model. These operations are conceptually described in the academic papers, but the implementations details are not trivial to understand and reproduce. To address this issue, this implementation emphasizes code readability, modularity, and documentation.

**Why [Rust](https://www.rust-lang.org)?** 

The original implementation was written in Python, which is widely used for machine learning, but not particularly efficient and not always suitable in production environments.

Rust combines bare-metal performance with memory and thread safety. It helps to write fast, reliable, and resource-efficient code by ensuring sound concurrency and memory use at compile time. For example, the borrow checker enforces strict ownership rules, reducing costly operations like cloning to prevent data races. 

Although it is not yet as widespread as Python in the ML world, it makes an excellent candidate for enabling reliable and efficient ML systems.


## 🎓 Public API

Include `gline-rs` as a regular dependency in your `Cargo.toml`:

```toml
[dependencies]
"gline-rs" = "1"
```

The public API is self-explanatory:

```rust
let model = GLiNER::<TokenMode>::new(
    Parameters::default(),
    RuntimeParameters::default(),
    "tokenizer.json",
    "model.onnx",
)?;

let input = TextInput::from_str(
    &[
        "My name is James Bond.", 
        "I like to drive my Aston Martin.",
    ],
    &[
        "person", 
        "vehicle",
    ],
)?;

let output = model.inference(input)?;

// => "James Bond" : "person"
// => "Aston Martin" : "vehicle"
```

For complete working examples please refer to the `examples` directory.


## 🧬 Getting the Models

To leverage `gline-rs`, you need the appropriate models in [ONNX](https://onnx.ai) format.

Ready-to-use models can be downloaded from Hugging Face repositories. For example:

| Model                                                                                         | ONNX                                                                      | Mode  |
|-----------------------------------------------------------------------------------------------|---------------------------------------------------------------------------|-------|
| [gliner small 2.1](https://huggingface.co/urchade/gliner_small-v2.1)                          | [link](https://huggingface.co/onnx-community/gliner_small-v2.1)           | Span  |
| [gliner multitask large 0.5](https://huggingface.co/knowledgator/gliner-multitask-large-v0.5) | [link](https://huggingface.co/onnx-community/gliner-multitask-large-v0.5) | Token |
| [gliner-x-large](https://huggingface.co/knowledgator/gliner-x-large)                          | [link](https://huggingface.co/knowledgator/gliner-x-large/tree/main/onnx) | Span  |

The original GLiNER implementation also provides [some tools](https://github.com/urchade/GLiNER/blob/main/examples/convert_to_onnx.ipynb) to convert models by your own.


## 🚀 Running the Examples

They are located in the `examples` directory. For instance, in token mode:

```bash
$ cargo run --example token-mode
```

Expected output:

```
0 | James Bond      | person     | 99.7%
1 | James           | person     | 98.1%
1 | Chelsea         | location   | 96.4%
1 | London          | location   | 92.4%
2 | James Bond      | person     | 99.4%
3 | Aston Martin    | vehicle    | 99.9%
```

**You first need to get the models to run the examples** (see above). To run them without any modification, the following file structure is expected:

For token-mode:
```
models/gliner-multitask-large-v0.5/tokenizer.json
models/gliner-multitask-large-v0.5/onnx/model.onnx
```

For span-mode:
```
models/gliner_small-v2.1/tokenizer.json
models/gliner_small-v2.1/onnx/model.onnx
```

Or you can just adapt the paths in the example code.

## ⚡️ GPU/NPU Inferences

The `ort` execution providers can be leveraged to perform considerably faster inferences on GPU/NPU hardware. A working example is provided in `examples/benchmark-gpu.rs`.

The first step is to pass the appropriate execution providers in `RuntimeParameters` (which is then passed to `GLiNER` initialization). For example:

```rust
let rtp = RuntimeParameters::default().with_execution_providers([
    CUDAExecutionProvider::default().build()
]);
```

The second step is to activate the appropriate features (see related section below), otherwise the example will **silently fall-back** to CPU. For example:

```console
$ cargo run --example benchmark-gpu --features=cuda
```

Please refer to `doc/ORT.md` for details about execution providers.


## 📦 Crate Features

This create mirrors the following `orp`/`ort` features flags:

* Dynamic loading of ONNX-runtime libraries: `load-dynamic`
* Activation of execution providers: `cuda`, `tensorrt`, `directml`, `coreml`, `rocm`, `openvino`, `onednn`, `xnnpack`, `qnn`, `cann`, `nnapi`, `tvm`, `acl`, `armnn`, `migraphx`, `vitis`, and `rknpu`

Other feature flags:

* `memprof`: basic tooling for memory profiling/benchmarking (see `util::memprof`)


## ⏱️ Performances

### CPU

Comparing performances from one implementation to another is complicated, as they depend on many factors. But according to the first measures, it appears that `gline-rs` can run **4x faster** on CPU than the original implementation out of the box:

| Implementation | sequences/second |
|----------------|------------------|
| gline-rs       | 6.67             |
| GLiNER.py      | 1.61             |

Both implementations have been tested under the following configuration:

* Dataset: subset of the [NuNER](https://huggingface.co/datasets/numind/NuNER) dataset (first 100 entries)
* Mode: token, flat_ner: true
* Number of entity classes: 3
* Threshold: 0.5
* Model: [gliner-multitask-large-v0.5](https://huggingface.co/knowledgator/gliner-multitask-large-v0.5)
* CPU specs: Intel Core i9 @2.3Ghz with 8 cores (12 threads)
* `gline-rs` version: 0.9.0

### GPU

Unsurprisingly, leveraging a GPU dramatically increases the throughput:

| Implementation | sequences/second |
|----------------|------------------|
| gline-rs       | 248.75           |

The configuration of the test is similar to the above, except:

* Dataset: subset of the [NuNER](https://huggingface.co/datasets/numind/NuNER) dataset (first 1000 entries)
* Execution provider: CUDA
* GPU specs: NVIDIA RTX 4080 (16Gb VRAM)
* CPU specs: Intel Core i7 13700KF @3.4Ghz 
* `gline-rs` version: 0.9.1

(Comparison with the original implementation has yet to be done.)


## ⚙️ Design Principles

`gline-rs` is written in pure and safe Rust (beside the ONNX runtime itself), with the following dependencies:

* [`orp`](https://github.com/fbilhaut/orp), which relies on the [`ort`](https://ort.pyke.io) ONNX runtime wrapper,
* [`tokenizers`](https://github.com/huggingface/tokenizers) by Hugging-Face,
* [`ndarray`](https://docs.rs/ndarray/latest/ndarray/),
* [`regex`](https://crates.io/crates/regex).

The implementation aims to clearly distinguish and comment each processing step, make them easily configurable, and model the pipeline concept almost declaratively. 

Default configurations are provided, but it should be easy to adapt them:

* One can have a look at the `model::{pipeline, input, output}` modules to see how the pre- and post-processing steps are defined by implementing the `Pipeline` trait defined by [`orp`](https://github.com/fbilhaut/orp).
* Others traits like `Splitter` or `Tokenizer` can be easily leveraged to experiment with other implementations of the pre-processing steps.
* While there is always room for improvement, special care has been taken to craft idiomatic, generic, commented, and efficient code.

A matrix-level documentation of the processing pipelines is provided in `doc/Processing.[pdf,typ]`.


## 👉 Related

* 🏷️ [gliclass-rs](https://github.com/fbilhaut/gliclass-rs): inference engine for GLiClass models
* 🧲 [gte-rs](https://github.com/fbilhaut/gte-rs): general text embedding and re-ranking


## 📖 References and Acknowledgments

The following papers were used as references:

* [GLiNER: Generalist Model for Named Entity Recognition using Bidirectional Transformer](https://arxiv.org/abs/2311.08526) by Urchade Zaratiana, Nadi Tomeh, Pierre Holat and Thierry Charnois (2023).
* [GLiNER multi-task: Generalist Lightweight Model for Various Information Extraction Tasks](https://arxiv.org/abs/2406.12925) by Ihor Stepanov and Mykhailo Shtopko (2024).
* [Named Entity Recognition as Structured Span Prediction](https://aclanthology.org/2022.umios-1.1/) by Urchade Zaratiana, Nadi Tomeh, Pierre Holat, Thierry Charnois (2022).

The original implementation was also used to check for the details. 

Special thanks to the original authors of GLiNER for this great and original work. 🙏
