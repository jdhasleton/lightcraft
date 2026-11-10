//! Fine-tune the PMRID denoise network on your own raw photos, outside the app (docs/denoise.md → Training on your photos).
//!
//! ```text
//! cargo run --release -p lightcraft-engine --features denoise --example denoise-train -- \
//!     prepare --from <folder of raws> --out <data folder> [--max-iso 400] [--crops 6] [--cells 256]
//! cargo run --release -p lightcraft-engine --features denoise --example denoise-train -- \
//!     train --data <data folder> --init <converted PMRID folder> --out <model folder> [--steps 3000] [--batch 8]
//! ```
//!
//! `prepare` reads raw files (never writes to them), keeps Bayer raws shot at `--max-iso` or below, bins each 2 × 2 cells of
//! one colour into one (half the size, half the noise: cleaner ground truth) and stores random well-exposed crops of the
//! four packed planes, about one photo in twenty held out for validation. `train` starts from PMRID's weights and teaches
//! the network to clean noise of a Canon full-frame body (Poisson-Gaussian, along the noise measured on EOS R bodies from
//! ISO 400 to 51200, `docs/denoise.md`) added to those crops, in the k-sigma domain the runner uses, with an L1 loss and
//! Adam; every `--every` steps it reports the PSNR on the held-out crops and writes a model folder that Settings ▸ AI
//! Denoise ▸ Install from file… takes.
//!
//! The crops and the trained weights are made from your photos: they stay in the folders you name, on your computer.
//! Never commit or publish them.

use std::path::{Path, PathBuf};
use std::time::Instant;

use lightcraft_denoise::candle::pmrid::{Device, GradStore, Pmrid, Tensor, Var};
use lightcraft_denoise::candle::weights::Weights;
use lightcraft_denoise::candle::{gpu_device, k_sigma};
use lightcraft_denoise::manifest::{self, DenoiserManifest, Network, NoiseMap};
use rayon::prelude::*;

type Res<T> = Result<T, String>;

fn err(e: impl std::fmt::Display) -> String {
    e.to_string()
}

/// The value after `--name`, or `default`.
fn opt<T: std::str::FromStr>(args: &[String], name: &str, default: T) -> Res<T> {
    match args.iter().position(|a| a == name) {
        Some(i) => args.get(i + 1).ok_or(format!("{name} needs a value"))?.parse().map_err(|_| format!("{name}: not a valid value")),
        None => Ok(default),
    }
}

fn path(args: &[String], name: &str) -> Res<PathBuf> {
    args.iter().position(|a| a == name).and_then(|i| args.get(i + 1)).map(PathBuf::from).ok_or(format!("{name} <folder> is needed"))
}

/// A small fast generator (xorshift64*), the same sequence for the same seed.
struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 >> 12;
        self.0 ^= self.0 << 25;
        self.0 ^= self.0 >> 27;
        self.0.wrapping_mul(0x2545_f491_4f6c_dd1d)
    }
    /// Uniform in 0..1.
    fn uniform(&mut self) -> f32 {
        ((self.next() >> 40) as f32 + 0.5) / (1u64 << 24) as f32
    }
    fn below(&mut self, n: usize) -> usize {
        (self.next() % n.max(1) as u64) as usize
    }
    /// Standard normal (Box-Muller).
    fn normal(&mut self) -> f32 {
        let (u, v) = (self.uniform(), self.uniform());
        (-2.0 * u.ln()).sqrt() * (std::f32::consts::TAU * v).cos()
    }
    /// Log-uniform between `lo` and `hi`.
    fn log_uniform(&mut self, lo: f32, hi: f32) -> f32 {
        (lo.ln() + self.uniform() * (hi.ln() - lo.ln())).exp()
    }
}

// ---------------------------------------------------------------------------------------------------------------------
// prepare

const RAW_EXTENSIONS: [&str; 9] = ["cr3", "cr2", "nef", "arw", "dng", "rw2", "orf", "pef", "srw"];

fn walk(dir: &Path, out: &mut Vec<PathBuf>, depth: usize) {
    let Ok(entries) = std::fs::read_dir(dir) else { return };
    for e in entries.flatten() {
        let p = e.path();
        if p.is_dir() && depth < 12 {
            walk(&p, out, depth + 1);
        } else if p.extension().and_then(|e| e.to_str()).is_some_and(|e| RAW_EXTENSIONS.iter().any(|r| e.eq_ignore_ascii_case(r))) {
            out.push(p);
        }
    }
}

/// What one photo gave.
enum Taken {
    Crops(Vec<Vec<u16>>, String),
    Skipped(&'static str),
}

/// The four planes of a normalised Bayer mosaic (RGGB from `(ox, oy)`), each 2 × 2 cells of one colour averaged.
fn binned(data: &[f32], w: usize, h: usize, ox: usize, oy: usize) -> (Vec<Vec<f32>>, usize, usize) {
    let (cw, ch) = ((w.saturating_sub(ox)) / 2, (h.saturating_sub(oy)) / 2);
    let (bw, bh) = (cw / 2, ch / 2);
    let at = |x: usize, y: usize| data.get(y * w + x).copied().unwrap_or(0.0);
    let planes = (0..4)
        .map(|p| {
            let (dx, dy) = (p % 2, p / 2);
            let mut v = Vec::with_capacity(bw * bh);
            for by in 0..bh {
                for bx in 0..bw {
                    let mut s = 0.0;
                    for (cx, cy) in [(2 * bx, 2 * by), (2 * bx + 1, 2 * by), (2 * bx, 2 * by + 1), (2 * bx + 1, 2 * by + 1)] {
                        s += at(ox + 2 * cx + dx, oy + 2 * cy + dy);
                    }
                    v.push(s / 4.0);
                }
            }
            v
        })
        .collect();
    (planes, bw, bh)
}

fn take(path: &Path, max_iso: u32, crops: usize, cells: usize, seed: u64) -> Taken {
    let Ok(bytes) = std::fs::read(path) else { return Taken::Skipped("unreadable") };
    let meta = lightcraft_meta::extract(&bytes);
    match meta.iso {
        Some(iso) if iso > 0 && iso <= max_iso => {}
        Some(_) => return Taken::Skipped("ISO above the limit"),
        None => return Taken::Skipped("no ISO"),
    }
    let Ok(n) = lightcraft_raw::decode(&bytes).and_then(|r| r.normalized()) else { return Taken::Skipped("not decoded") };
    drop(bytes);
    let Some(cfa) = n.cfa.as_ref().filter(|_| n.cpp == 1) else { return Taken::Skipped("not a Bayer raw") };
    let Some(layout) = lightcraft_denoise::bayer::Layout::from_cell([cfa.color_at(0, 0), cfa.color_at(1, 0), cfa.color_at(0, 1), cfa.color_at(1, 1)])
    else {
        return Taken::Skipped("not a Bayer raw");
    };
    let (planes, bw, bh) = binned(&n.data, n.width, n.height, layout.dx, layout.dy);
    drop(n);
    if bw < cells || bh < cells {
        return Taken::Skipped("too small");
    }
    let mut rng = Rng(seed | 1);
    let mut out = Vec::new();
    for _ in 0..crops * 20 {
        if out.len() == crops {
            break;
        }
        let (x0, y0) = (rng.below(bw - cells + 1), rng.below(bh - cells + 1));
        let crop: Vec<f32> = planes
            .iter()
            .flat_map(|p| (0..cells).flat_map(move |y| (0..cells).map(move |x| p.get((y0 + y) * bw + x0 + x).copied().unwrap_or(0.0))))
            .collect();
        let mean = crop.iter().sum::<f32>() / crop.len() as f32;
        let clipped = crop.iter().filter(|v| **v > 0.95).count() as f32 / crop.len() as f32;
        // well exposed, hardly clipped: dark crops say little and blown ones nothing
        if !(0.03..=0.6).contains(&mean) || clipped > 0.002 {
            continue;
        }
        out.push(crop.iter().map(|v| (v.clamp(0.0, 1.0) * 65535.0).round() as u16).collect());
    }
    Taken::Crops(out, format!("{} {}", meta.make.unwrap_or_default(), meta.model.unwrap_or_default()).trim().to_string())
}

fn prepare(args: &[String]) -> Res<()> {
    let (from, out) = (path(args, "--from")?, path(args, "--out")?);
    let (max_iso, crops, cells, threads) =
        (opt(args, "--max-iso", 400u32)?, opt(args, "--crops", 6usize)?, opt(args, "--cells", 256usize)?, opt(args, "--threads", 6usize)?);
    let mut files = Vec::new();
    walk(&from, &mut files, 0);
    files.sort();
    println!("{} raw files under {}", files.len(), from.display());
    let started = Instant::now();
    let pool = rayon::ThreadPoolBuilder::new().num_threads(threads.max(1)).build().map_err(err)?;
    let done = std::sync::atomic::AtomicUsize::new(0);
    let taken: Vec<(usize, Taken)> = pool.install(|| {
        files
            .par_iter()
            .enumerate()
            .map(|(i, f)| {
                let t = take(f, max_iso, crops, cells, 0x9e37_79b9 ^ i as u64);
                let n = done.fetch_add(1, std::sync::atomic::Ordering::Relaxed) + 1;
                if n.is_multiple_of(200) {
                    println!("  {n} of {} read ({:.0} s)", files.len(), started.elapsed().as_secs_f64());
                }
                (i, t)
            })
            .collect()
    });
    std::fs::create_dir_all(&out).map_err(err)?;
    let (mut train, mut holdout) = (Vec::<u8>::new(), Vec::<u8>::new());
    let (mut n_train, mut n_holdout, mut photos) = (0usize, 0usize, 0usize);
    let mut skipped = std::collections::BTreeMap::<&str, usize>::new();
    let mut cameras = std::collections::BTreeMap::<String, usize>::new();
    for (i, t) in taken {
        match t {
            Taken::Skipped(why) => *skipped.entry(why).or_default() += 1,
            Taken::Crops(c, camera) if !c.is_empty() => {
                photos += 1;
                *cameras.entry(camera).or_default() += 1;
                // one photo in twenty, all of its crops, is held out
                let (buf, n) = if (i as u64).wrapping_mul(0x9e37_79b9_7f4a_7c15) >> 59 == 0 {
                    (&mut holdout, &mut n_holdout)
                } else {
                    (&mut train, &mut n_train)
                };
                for crop in c {
                    buf.extend(crop.iter().flat_map(|v| v.to_le_bytes()));
                    *n += 1;
                }
            }
            Taken::Crops(..) => *skipped.entry("no well-exposed crop").or_default() += 1,
        }
    }
    std::fs::write(out.join("train.u16"), &train).map_err(err)?;
    std::fs::write(out.join("holdout.u16"), &holdout).map_err(err)?;
    let info = serde_json::json!({"cells": cells, "train": n_train, "holdout": n_holdout, "photos": photos, "maxIso": max_iso,
        "binned": 2, "skipped": skipped, "cameras": cameras});
    std::fs::write(out.join("data.json"), serde_json::to_vec_pretty(&info).map_err(err)?).map_err(err)?;
    println!("{info}\n{:.0} s", started.elapsed().as_secs_f64());
    Ok(())
}

// ---------------------------------------------------------------------------------------------------------------------
// train

/// The noise of a Canon EOS R body at `iso` (variance = a · signal + b, white 1): the levels measured on real raws
/// (docs/denoise.md → Fitted on real cameras), interpolated in logarithms.
fn canon_noise(iso: f32) -> (f32, f32) {
    const AT: [(f32, f32, f32); 5] =
        [(400.0, 4e-5, 1e-7), (1600.0, 2.1e-4, 3e-7), (6400.0, 4.9e-4, 1e-6), (25600.0, 1.19e-3, 1.7e-5), (51200.0, 6.9e-3, 2.08e-4)];
    let iso = iso.clamp(AT[0].0, AT[4].0);
    let i = AT.windows(2).position(|w| iso <= w[1].0).unwrap_or(3);
    let (lo, hi) = (AT[i], AT[i + 1]);
    let t = (iso.ln() - lo.0.ln()) / (hi.0.ln() - lo.0.ln());
    let mix = |a: f32, b: f32| (a.ln() + t * (b.ln() - a.ln())).exp();
    (mix(lo.1, hi.1), mix(lo.2, hi.2))
}

/// One training (or validation) sample: the network's input and its target, both mapped and scaled as the runner does.
fn sample(clean: &[f32], iso: f32, gain: f32, jitter: (f32, f32), anchor: (f32, f32), scale: f32, rng: &mut Rng) -> (Vec<f32>, Vec<f32>) {
    let (a, b) = canon_noise(iso);
    let (s, o) = k_sigma((a * jitter.0, b * jitter.1), anchor);
    let mut input = Vec::with_capacity(clean.len());
    let mut target = Vec::with_capacity(clean.len());
    for &c in clean {
        let x = (c * gain).min(1.0);
        let noisy = x + (a * x + b).max(0.0).sqrt() * rng.normal();
        input.push((s * noisy.clamp(0.0, 1.0) + o) * scale);
        target.push((s * x + o) * scale);
    }
    (input, target)
}

fn crops_of(path: &Path, per: usize) -> Res<Vec<f32>> {
    let bytes = std::fs::read(path).map_err(|e| format!("{}: {e}", path.display()))?;
    if bytes.len() % (2 * per) != 0 {
        return Err(format!("{}: not whole crops", path.display()));
    }
    Ok(bytes.as_chunks::<2>().0.iter().map(|b| f32::from(u16::from_le_bytes(*b)) / 65535.0).collect())
}

/// PSNR in dB of `a` against `b` (both in the linear 0..1 mosaic), after the exposure that puts their mean at 18 % and a
/// 2.2 gamma, as the benchmark measures.
fn psnr(a: &[f32], b: &[f32]) -> f64 {
    let mean = b.iter().map(|v| f64::from(*v)).sum::<f64>() / b.len().max(1) as f64;
    let k = (0.18 / mean.max(1e-6)) as f32;
    let g = |v: f32| (v * k).clamp(0.0, 1.0).powf(1.0 / 2.2);
    let mse = a.iter().zip(b).map(|(x, y)| f64::from(g(*x) - g(*y)).powi(2)).sum::<f64>() / a.len().max(1) as f64;
    -10.0 * mse.max(1e-12).log10()
}

struct Adam {
    m: Vec<Tensor>,
    v: Vec<Tensor>,
    t: i32,
}

impl Adam {
    fn new(vars: &[(String, Var)]) -> Res<Self> {
        let zeros = |v: &Var| v.as_tensor().zeros_like().map_err(err);
        Ok(Adam { m: vars.iter().map(|(_, v)| zeros(v)).collect::<Res<_>>()?, v: vars.iter().map(|(_, v)| zeros(v)).collect::<Res<_>>()?, t: 0 })
    }

    fn step(&mut self, vars: &[(String, Var)], grads: &GradStore, lr: f64) -> Res<()> {
        let (b1, b2, eps) = (0.9f64, 0.999f64, 1e-8f64);
        self.t += 1;
        let (c1, c2) = (1.0 - b1.powi(self.t), 1.0 - b2.powi(self.t));
        for (i, (_, var)) in vars.iter().enumerate() {
            // the gradient carries the step's graph (every activation of the forward pass): the moments must not keep it
            let Some(g) = grads.get(var).map(Tensor::detach) else { continue };
            let (Some(m), Some(v)) = (self.m.get_mut(i), self.v.get_mut(i)) else { continue };
            *m = ((&*m * b1).map_err(err)? + (&g * (1.0 - b1)).map_err(err)?).map_err(err)?.detach();
            *v = ((&*v * b2).map_err(err)? + (g.sqr().map_err(err)? * (1.0 - b2)).map_err(err)?).map_err(err)?.detach();
            let update = ((&*m / c1).map_err(err)? / ((&*v / c2).map_err(err)?.sqrt().map_err(err)? + eps).map_err(err)?).map_err(err)?;
            var.set(&(var.as_tensor().detach() - (update * lr).map_err(err)?).map_err(err)?).map_err(err)?;
        }
        Ok(())
    }
}

fn save(dir: &Path, vars: &[(String, Var)], init: &DenoiserManifest, note: &str) -> Res<()> {
    std::fs::create_dir_all(dir).map_err(err)?;
    let bytes = Pmrid::weights_of(vars).map_err(err)?.to_bytes();
    let file = dir.join("model.safetensors");
    std::fs::write(&file, &bytes).map_err(err)?;
    let mut m = init.clone();
    m.id = "pmrid-yours".into();
    m.name = "PMRID, fine-tuned on your photos".into();
    m.version = note.chars().take(40).collect();
    m.sha256 = Some(lightcraft_denoise::hash::sha256_file(&file).map_err(err)?);
    m.size_bytes = Some(bytes.len() as u64);
    m.licence.notice = "Fine-tuned by you from PMRID's weights (Apache License 2.0) on your own photos. Keep it on your computer: \
                        it was made from your pictures."
        .into();
    m.provenance = "PMRID (Wang et al., ECCV 2020) fine-tuned with LightCraft's denoise-train on clean low-ISO crops of the \
                    user's own raw photos, with synthetic noise of a Canon full-frame body from ISO 400 to 51200."
        .into();
    manifest::validate(&m).map_err(err)?;
    std::fs::write(dir.join("denoise-model.json"), serde_json::to_vec_pretty(&m).map_err(err)?).map_err(err)
}

fn train(args: &[String]) -> Res<()> {
    let (data, init, out) = (path(args, "--data")?, path(args, "--init")?, path(args, "--out")?);
    let (steps, batch, lr, every) =
        (opt(args, "--steps", 3000usize)?, opt(args, "--batch", 8usize)?, opt(args, "--lr", 1e-4f64)?, opt(args, "--every", 250usize)?);
    let info: serde_json::Value = serde_json::from_slice(&std::fs::read(data.join("data.json")).map_err(err)?).map_err(err)?;
    let cells = info["cells"].as_u64().ok_or("data.json has no cells")? as usize;
    let per = 4 * cells * cells;
    let (train, holdout) = (crops_of(&data.join("train.u16"), per)?, crops_of(&data.join("holdout.u16"), per)?);
    let (n_train, n_hold) = (train.len() / per, holdout.len() / per);
    if n_train == 0 || n_hold == 0 {
        return Err("no training or held-out crops: run prepare first".into());
    }
    let m: DenoiserManifest = manifest::parse(&std::fs::read(init.join("denoise-model.json")).map_err(err)?).map_err(err)?;
    let Network::Safetensors { noise: NoiseMap::KSigma { a, b }, input_scale, .. } = m.network else {
        return Err("--init must be a converted PMRID folder (a safetensors network with a k-sigma noise map)".into());
    };
    let anchor = (a, b);
    let device = if args.iter().any(|a| a == "--cpu") { Device::Cpu } else { gpu_device().unwrap_or(Device::Cpu) };
    let (net, vars) = Pmrid::trainable(&mut Weights::read(&init.join("model.safetensors")).map_err(err)?, &device).map_err(err)?;
    // The transposed convolutions (`*.upsample.*`) keep PMRID's weights unless `--train-upsample`: trained with the rest,
    // the held-out PSNR fell below the published weights within a few hundred steps at every learning rate tried.
    let trained: Vec<(String, Var)> =
        vars.iter().filter(|(name, _)| args.iter().any(|a| a == "--train-upsample") || !name.contains(".upsample.")).cloned().collect();
    let mut adam = Adam::new(&trained)?;
    println!("{n_train} training crops, {n_hold} held out, {cells} cells; {steps} steps of {batch} on {:?}", device.location());

    // the held-out set: each crop at three ISOs, with fixed noise
    let isos = [1600.0f32, 6400.0, 25600.0];
    let validate = |net: &Pmrid| -> Res<Vec<(f32, f64, f64)>> {
        let mut rng = Rng(77);
        let mut out = Vec::new();
        for &iso in &isos {
            let (mut noisy_db, mut db, mut n) = (0f64, 0f64, 0f64);
            for k in 0..n_hold.min(48) {
                let clean = holdout.get(k * per..(k + 1) * per).ok_or("held-out crop")?;
                let (input, _) = sample(clean, iso, 1.0, (1.0, 1.0), anchor, input_scale, &mut rng);
                let (s, o) = k_sigma(canon_noise(iso), anchor);
                let back = |v: &[f32]| v.iter().map(|y| (y / input_scale - o) / s).collect::<Vec<f32>>();
                let x = Tensor::from_vec(input.clone(), (1, 4, cells, cells), &device).map_err(err)?;
                let y = net.forward(&x).and_then(|y| y.flatten_all()).and_then(|y| y.to_vec1::<f32>()).map_err(err)?;
                noisy_db += psnr(&back(&input), clean);
                db += psnr(&back(&y), clean);
                n += 1.0;
            }
            out.push((iso, noisy_db / n, db / n));
        }
        Ok(out)
    };
    let report = |step: usize, v: &[(f32, f64, f64)]| {
        let parts: Vec<String> = v.iter().map(|(iso, a, b)| format!("ISO {iso}: {a:.2} → {b:.2} dB")).collect();
        println!("step {step}: held out {}", parts.join(" | "));
    };
    report(0, &validate(&net)?);

    let mut rng = Rng(12345);
    let (started, mut avg) = (Instant::now(), 0f64);
    let reference = k_sigma(canon_noise(1600.0), anchor).0;
    let mut weights = Vec::with_capacity(batch);
    for step in 1..=steps {
        let mut input = Vec::with_capacity(batch * per);
        let mut target = Vec::with_capacity(batch * per);
        for _ in 0..batch {
            let k = rng.below(n_train);
            let clean = train.get(k * per..(k + 1) * per).ok_or("training crop")?;
            let iso = rng.log_uniform(400.0, 51200.0);
            let gain = rng.log_uniform(0.5, 2.0);
            let jitter = (rng.log_uniform(0.8, 1.25), rng.log_uniform(0.8, 1.25));
            let (i, t) = sample(clean, iso, gain, jitter, anchor, input_scale, &mut rng);
            input.extend(i);
            target.extend(t);
            // The k-sigma map multiplies a crop by s: up to 16 at low ISO, well under 1 at high ISO, so an unweighted
            // loss is ruled by the cleanest crops and trades the high ISOs away. Weighting by 1 / √s makes an error
            // count by its size against the crop's own noise (√s at ISO 1600 keeps that weight at 1).
            weights.push((reference / k_sigma(canon_noise(iso), anchor).0).sqrt());
        }
        let x = Tensor::from_vec(input, (batch, 4, cells, cells), &device).map_err(err)?;
        let t = Tensor::from_vec(target, (batch, 4, cells, cells), &device).map_err(err)?;
        let w = Tensor::from_vec(std::mem::take(&mut weights), (batch, 1, 1, 1), &device).map_err(err)?;
        let loss = net.forward(&x).and_then(|y| (y - t)?.abs()?.broadcast_mul(&w)?.mean_all()).map_err(err)?;
        let grads = loss.backward().map_err(err)?;
        // warm up for 100 steps, then a cosine down to a tenth
        let warm = (step as f64 / 100.0).min(1.0);
        let progress = step as f64 / steps as f64;
        let rate = lr * warm * (0.1 + 0.9 * 0.5 * (1.0 + (std::f64::consts::PI * progress).cos()));
        adam.step(&trained, &grads, rate)?;
        let l = f64::from(loss.to_scalar::<f32>().map_err(err)?) / f64::from(input_scale);
        avg = if step == 1 { l } else { 0.98 * avg + 0.02 * l };
        if step % 25 == 0 || step <= 3 {
            println!("step {step}: loss {avg:.6}, {:.2} s a step", started.elapsed().as_secs_f64() / step as f64);
        }
        if step % every == 0 || step == steps {
            report(step, &validate(&net)?);
            save(&out.join(format!("step-{step:05}")), &vars, &m, &format!("ft-{step}"))?;
        }
    }
    Ok(())
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let rest = args.get(1..).unwrap_or_default();
    let r = match args.first().map(String::as_str) {
        Some("prepare") => prepare(rest),
        Some("train") => train(rest),
        _ => Err("usage: denoise-train prepare --from <raws> --out <data> | train --data <data> --init <pmrid> --out <models>".into()),
    };
    if let Err(e) = r {
        eprintln!("denoise-train: {e}");
        std::process::exit(1);
    }
}
