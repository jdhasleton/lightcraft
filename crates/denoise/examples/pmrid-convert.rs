//! Turn the PMRID authors' PyTorch weights into a LightCraft denoise model, once, outside the app.
//!
//! ```text
//! git clone https://github.com/MegEngine/PMRID
//! cargo run --release -p lightcraft-denoise --features candle --example pmrid-convert -- \
//!     PMRID/models/torch_pretrained.ckp <out-folder>
//! ```
//!
//! writes `<out-folder>/model.safetensors` (the weights as F32) and `<out-folder>/denoise-model.json` (the manifest,
//! with the file's SHA-256 and size), which Settings ▸ AI Denoise ▸ Install from file… takes. The checkpoint is read with
//! candle's pickle reader, which understands tensors only and runs no code; every tensor PMRID has must be there with
//! its shape and nothing else, or nothing is written. See docs/denoise.md → PMRID for the licence of the weights and
//! the data they were trained on.

use std::error::Error;
use std::path::PathBuf;

use lightcraft_denoise::candle::pmrid::Pmrid;
use lightcraft_denoise::candle::weights::Weights;
use lightcraft_denoise::licence::{Commercial, Licence};
use lightcraft_denoise::manifest::{self, Architecture, DenoiserManifest, Domain, Gain, Network, NoiseMap};

/// The noise PMRID was trained for (its k-sigma transform's anchor, ISO 1600 of the OPPO Reno 10x): `K(1600)` and
/// `Sigma(1600)` of `run_benchmark.py`'s noise model, in 10-bit numbers (`V = 959`), as variance = `a · x + b` for `x` in
/// 0..1.
fn anchor() -> (f32, f32) {
    let iso = 1600f64;
    let k = 0.0005995267 * iso + 0.00868861;
    let sigma = 7.11772e-7 * iso * iso + 6.514934e-4 * iso + 0.11492713;
    let v = 959.0f64;
    ((k / v) as f32, (sigma / (v * v)) as f32)
}

pub fn pmrid_manifest(sha256: String, size: u64) -> DenoiserManifest {
    let (a, b) = anchor();
    DenoiserManifest {
        id: "pmrid".into(),
        name: "PMRID".into(),
        version: "1".into(),
        licence: Licence {
            name: "Apache-2.0".into(),
            commercial: Commercial::Yes,
            url: Some("https://github.com/MegEngine/PMRID/blob/main/LICENSE".into()),
            notice: "Weights published by the PMRID authors with their code under the Apache License 2.0 (the repository's \
                     licence; no separate terms are stated for the weights). LightCraft does not include them: you converted \
                     them yourself."
                .into(),
        },
        source: Some("https://github.com/MegEngine/PMRID/tree/main/models".into()),
        sha256: Some(sha256),
        size_bytes: Some(size),
        provenance: "Trained by its authors (Wang et al., ECCV 2020) on 214 clean long exposures from the See-in-the-Dark \
                     dataset (Sony a7S II; the dataset's repository is MIT) with synthetic noise of an OPPO Reno 10x phone \
                     (ISO 100 to 6400), mapped by the k-sigma transform to that phone's ISO 1600."
            .into(),
        domain: Domain::BayerToRgb,
        tile: 512,
        overlap: 32,
        gain: Gain::None,
        network: Network::Safetensors { architecture: Architecture::Pmrid, noise: NoiseMap::KSigma { a, b }, input_scale: 256.0 },
    }
}

fn main() -> Result<(), Box<dyn Error>> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let [ckp, out] = args.as_slice() else {
        return Err("usage: pmrid-convert <torch_pretrained.ckp> <out-folder>".into());
    };
    let (ckp, out) = (PathBuf::from(ckp), PathBuf::from(out));
    let tensors = candle_core::pickle::read_all(&ckp)?;
    let mut w = Weights::default();
    for (name, t) in tensors {
        let shape = t.dims().to_vec();
        let values = t.to_dtype(candle_core::DType::F32)?.flatten_all()?.to_vec1::<f32>()?;
        // a checkpoint saved from a wrapped model names its tensors `module.…`
        w.insert(name.strip_prefix("module.").unwrap_or(&name), &shape, values)?;
    }
    let want = Pmrid::parameters()?;
    let count: usize = want.iter().map(|(_, s)| s.iter().product::<usize>()).sum();
    // check before writing anything: every parameter with its shape, nothing left over
    let mut check = Weights::parse(&w.to_bytes())?;
    for (name, shape) in &want {
        check.take(name, shape)?;
    }
    if !check.left().is_empty() {
        return Err(format!("the checkpoint holds tensors PMRID does not have: {:?}", check.left()).into());
    }
    let bytes = w.to_bytes();
    std::fs::create_dir_all(&out)?;
    let model = out.join("model.safetensors");
    std::fs::write(&model, &bytes)?;
    let sha = lightcraft_denoise::hash::sha256_file(&model)?;
    let m = pmrid_manifest(sha.clone(), bytes.len() as u64);
    manifest::validate(&m)?;
    std::fs::write(out.join("denoise-model.json"), serde_json::to_vec_pretty(&m)?)?;
    println!("{} tensors, {count} numbers → {} ({} bytes, sha256 {sha})", want.len(), model.display(), bytes.len());
    Ok(())
}
