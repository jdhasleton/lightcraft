//! Opt-in quality check of the built-in denoiser on one public Bayer raw (CC0, from `cargo xtask corpus --download`).
//!
//! Noise of a known level is added to the raw's own mosaic, the built-in denoiser runs over it, and the result is
//! scored against the untouched mosaic demosaiced the same way. PSNR is measured on gamma-encoded camera RGB; the
//! detail figure is the gradient energy of the result over that of the truth (1 = as sharp, below 1 = smoother).
//!
//! LC_DENOISE_RAW=../../corpus/raw/arw-sony-a7m3-compressed.arw LC_DENOISE_OUT=/tmp/dn \
//!   cargo test --release -p lightcraft-engine --test denoise_builtin_quality -- --ignored --nocapture
//! LC_DENOISE_OUT, when set, receives side-by-side PNGs (truth | noisy | denoised) of a 100 % crop per level.
//!
//! Built with `--features denoise`, the installed models (`LC_DENOISE_MODELS`, else the user's models folder) are scored
//! beside the built-in denoiser. `LC_DENOISE_PAIRS=<folder>` scores real pairs instead of added noise (see
//! `denoise_quality/pairs.rs`): tripod shots of one scene at base ISO and at high ISOs.

#[path = "denoise_quality/pairs.rs"]
mod pairs;

use lightcraft_codecs::encode::{EncodeImage, EncodeMeta, Samples, encode_png};
use lightcraft_denoise_core::bayer::Layout;
use lightcraft_denoise_core::builtin::{Builtin, OVERLAP, TILE, demosaic, estimate_noise};
use lightcraft_denoise_core::manifest::Gain;
use lightcraft_denoise_core::nlm::Tuning;
use lightcraft_denoise_core::run::{Control, Params, TileRunner, denoise_bayer};
use std::path::{Path, PathBuf};
use std::time::Instant;

struct Rng(u64);

impl Rng {
    /// The next uniform number, strictly between 0 and 1.
    fn uniform(&mut self) -> f32 {
        self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        ((self.0 >> 40) as f32 + 0.5) / (1u64 << 24) as f32
    }

    /// The next standard normal number (Box-Muller).
    fn normal(&mut self) -> f32 {
        let (u, v) = (self.uniform(), self.uniform());
        (-2.0 * u.ln()).sqrt() * (std::f32::consts::TAU * v).cos()
    }
}

/// The plain demosaic of a mosaic (RGGB, even size) as camera RGB planes, through the tile packer the runner uses.
fn plain(mosaic: &[f32], w: usize, h: usize) -> Vec<[f32; 3]> {
    let (cw, ch) = (w / 2, h / 2);
    let mut planes = vec![vec![0f32; cw * ch]; 4];
    for cy in 0..ch {
        for cx in 0..cw {
            for (p, (dx, dy)) in [(0, 0), (1, 0), (0, 1), (1, 1)].into_iter().enumerate() {
                planes[p][cy * cw + cx] = mosaic[(2 * cy + dy) * w + 2 * cx + dx];
            }
        }
    }
    // `demosaic` takes square planes: run it on square blocks and stitch
    let side = cw.min(ch);
    let mut out = vec![[0f32; 3]; w * h];
    for by in (0..ch).step_by(side) {
        for bx in (0..cw).step_by(side) {
            let (x0, y0) = (bx.min(cw - side), by.min(ch - side));
            let block: Vec<Vec<f32>> = planes.iter().map(|pl| (0..side * side).map(|i| pl[(y0 + i / side) * cw + x0 + i % side]).collect()).collect();
            let rgb = demosaic(&block, side);
            let s2 = 2 * side;
            for y in 0..s2 {
                for x in 0..s2 {
                    let o = (2 * y0 + y) * w + 2 * x0 + x;
                    out[o] = [rgb[y * s2 + x], rgb[s2 * s2 + y * s2 + x], rgb[2 * s2 * s2 + y * s2 + x]];
                }
            }
        }
    }
    out
}

/// A linear value encoded for viewing: clamped to 0..1, then gamma 2.2.
fn gamma(v: f32) -> f32 {
    v.clamp(0.0, 1.0).powf(1.0 / 2.2)
}

/// PSNR in dB between two camera-RGB pictures after exposure `scale` and gamma, leaving out a 16-pixel border.
fn psnr(a: &[[f32; 3]], b: &[[f32; 3]], w: usize, h: usize, scale: f32) -> f64 {
    let (mut sum, mut n) = (0f64, 0f64);
    for y in 16..h - 16 {
        for x in 16..w - 16 {
            for c in 0..3 {
                let d = f64::from(gamma(a[y * w + x][c] * scale) - gamma(b[y * w + x][c] * scale));
                sum += d * d;
                n += 1.0;
            }
        }
    }
    -10.0 * (sum / n).log10()
}

/// The summed gradient of the picture's green channel after exposure `scale` and gamma, leaving out a 16-pixel border.
fn gradient(a: &[[f32; 3]], w: usize, h: usize, scale: f32) -> f64 {
    let mut sum = 0f64;
    for y in 16..h - 16 {
        for x in 16..w - 16 {
            let g = |x: usize, y: usize| f64::from(gamma(a[y * w + x][1] * scale));
            sum += (g(x + 1, y) - g(x, y)).abs() + (g(x, y + 1) - g(x, y)).abs();
        }
    }
    sum
}

/// How much of the true picture's fine structure the result keeps: the slope of the result's high-pass (green minus
/// its 5 x 5 mean) against the truth's. Noise is independent of the truth, so unlike gradient energy it can't inflate it.
fn texture(a: &[[f32; 3]], truth: &[[f32; 3]], w: usize, h: usize, scale: f32) -> f64 {
    let hp = |p: &[[f32; 3]], x: usize, y: usize| {
        let mut m = 0f64;
        for j in 0..5 {
            for i in 0..5 {
                m += f64::from(gamma(p[(y + j - 2) * w + x + i - 2][1] * scale));
            }
        }
        f64::from(gamma(p[y * w + x][1] * scale)) - m / 25.0
    };
    let (mut xy, mut xx) = (0f64, 0f64);
    for y in 16..h - 16 {
        for x in 16..w - 16 {
            let t = hp(truth, x, y);
            xy += t * hp(a, x, y);
            xx += t * t;
        }
    }
    xy / xx
}

/// A `side × side` crop of the picture from `(x0, y0)` as 8-bit RGB with a daylight white balance, for the PNGs.
fn crop_rgb8(a: &[[f32; 3]], w: usize, (x0, y0): (usize, usize), side: usize, scale: f32) -> Vec<u8> {
    let wb = [2.0f32, 1.0, 1.6];
    let mut out = Vec::with_capacity(side * side * 3);
    for y in 0..side {
        for x in 0..side {
            for c in 0..3 {
                out.push((gamma(a[(y0 + y) * w + x0 + x][c] * wb[c] * scale) * 255.0 + 0.5) as u8);
            }
        }
    }
    out
}

/// Adds noise of known levels to a public raw's mosaic, denoises it with the built-in denoiser and prints how close the
/// result is to the untouched mosaic. Opt-in: it needs the raw named by `LC_DENOISE_RAW`.
#[test]
#[ignore = "needs a public raw: LC_DENOISE_RAW=<file>"]
fn the_built_in_denoiser_scored_on_a_real_raw_with_added_noise() {
    let path = std::env::var("LC_DENOISE_RAW").unwrap_or_else(|_| "../../corpus/raw/arw-sony-a7m3-compressed.arw".into());
    let bytes = std::fs::read(&path).unwrap_or_else(|e| panic!("{path}: {e}"));
    let n = lightcraft_raw::decode(&bytes).unwrap().normalized().unwrap();
    let cfa = n.cfa.as_ref().unwrap();
    let layout = Layout::from_cell([cfa.color_at(0, 0), cfa.color_at(1, 0), cfa.color_at(0, 1), cfa.color_at(1, 1)]).unwrap();
    assert_eq!(layout, Layout { dx: 0, dy: 0 }, "the benchmark assumes RGGB");
    // an even-aligned window from the middle of the frame keeps the colour pattern
    let (w, h) = (1536usize, 1024usize);
    let (ox, oy) = (((n.width - w) / 2) & !1, ((n.height - h) / 2) & !1);
    let truth: Vec<f32> = (0..w * h).map(|i| n.data[(oy + i / w) * n.width + ox + i % w]).collect();
    // LC_BIN=1: average each cell with the three nearest of its own colour (a 2 x 2 binning of the colour pattern's
    // cells), which halves the reference's own noise so that noise added at a camera's low levels dominates it
    let (w, h, truth) = if std::env::var("LC_BIN").is_ok_and(|v| v != "0") {
        let (bw, bh) = (w / 2, h / 2);
        let binned: Vec<f32> = (0..bw * bh)
            .map(|i| {
                let (x, y) = (i % bw, i / bw);
                let (sx, sy, cx, cy) = (x / 2, y / 2, x % 2, y % 2);
                let at = |di: usize, dj: usize| truth[(2 * (2 * sy + dj) + cy) * w + 2 * (2 * sx + di) + cx];
                (at(0, 0) + at(1, 0) + at(0, 1) + at(1, 1)) / 4.0
            })
            .collect();
        (bw, bh, binned)
    } else {
        (w, h, truth)
    };
    let mean = truth.iter().map(|v| f64::from(*v)).sum::<f64>() / truth.len() as f64;
    let scale = (0.18 / mean) as f32;
    let truth_rgb = plain(&truth, w, h);
    let own = estimate_noise(&truth, w, h);
    println!(
        "the reference's own noise by the estimator: a {:.5} b {:.6} (sd at 0.18 {:.4}); a clean reference would read about 0",
        own.0,
        own.1,
        (own.0 * 0.18 + own.1).sqrt()
    );
    println!("{path}: {}x{}, window {w}x{h}, mean {mean:.4}", n.width, n.height);
    println!("{:<14}{:>10}{:>10}{:>9}{:>9}{:>9}{:>9}", "noise (a, b)", "noisy dB", "denoised", "gain", "detail", "texture", "seconds");
    let out_dir = std::env::var("LC_DENOISE_OUT").ok();
    let models = installed_models();
    println!("models scored beside the built-in denoiser: {}", models.iter().map(|m| m.name.as_str()).collect::<Vec<_>>().join(", "));
    #[cfg(feature = "denoise")]
    let gpu = std::env::var("LC_GPU").is_ok_and(|v| v != "0").then(|| {
        let p = Tuning::default().bm3d.unwrap();
        let g = lightcraft_gpu::bm3d::Bm3dGpu::new(&p, TILE as usize).unwrap();
        println!("on the graphics card: {} ({})", g.adapter(), g.build_report());
        std::sync::Arc::new(g)
    });
    // LC_REAL=1 runs the noise levels measured on real Canon EOS R / R6 Mark III raws instead (their standard deviation at
    // mid-grey is 0.003, 0.006, 0.0095, 0.015 and 0.038: ISO <= 400, 1600, 6400, 25600 and above)
    let levels: &[(&str, f32, f32)] = if std::env::var("LC_REAL").is_ok_and(|v| v != "0") {
        &[("iso400", 4e-5, 1e-7), ("iso1600", 2.1e-4, 3e-7), ("iso6400", 4.9e-4, 1e-6), ("iso25600", 1.19e-3, 1.7e-5), ("iso51200", 6.9e-3, 2.08e-4)]
    } else {
        &[("light", 0.002, 0.0002), ("medium", 0.008, 0.001), ("heavy", 0.025, 0.004), ("extreme", 0.06, 0.012)]
    };
    for &(name, a, b) in levels {
        let mut rng = Rng(11);
        let noisy: Vec<f32> = truth.iter().map(|&v| v + rng.normal() * (a * v.max(0.0) + b).sqrt()).collect();
        let started = Instant::now();
        let params = Params { tile: TILE as usize, overlap: OVERLAP as usize, gain: Gain::None, clip: Some(0.99), parallel: 4 };
        // LC_EST=v1 (the first estimator) | true (the noise that was added) | anything else: the current estimator
        // LC_FULL=1: measure the noise on the whole frame with the same noise added, as the app does (it measures the whole
        // photo), instead of on the crop
        let measure = |mosaic: &[f32], mw: usize, mh: usize| match std::env::var("LC_EST").as_deref() {
            Ok("v1") => lightcraft_denoise_core::builtin::estimate_noise_v1(mosaic, mw, mh),
            _ => estimate_noise(mosaic, mw, mh),
        };
        let est = match std::env::var("LC_EST").as_deref() {
            Ok("true") => (a, b),
            _ if std::env::var("LC_TRUE").is_ok_and(|v| v != "0") => (a, b),
            _ if std::env::var("LC_FULL").is_ok_and(|v| v != "0") => {
                let mut rng = Rng(23);
                let frame: Vec<f32> = n.data.iter().map(|&v| v + rng.normal() * (a * v.max(0.0) + b).sqrt()).collect();
                measure(&frame, n.width, n.height)
            }
            _ => measure(&noisy, w, h),
        };
        let mut tuning = Tuning::default();
        let env = |k: &str| std::env::var(k).ok().and_then(|v| v.parse::<f32>().ok());
        (tuning.search, tuning.search2) = (env("LC_S1").map_or(tuning.search, |v| v as usize), env("LC_S2").map_or(tuning.search2, |v| v as usize));
        (tuning.h_light, tuning.h_heavy, tuning.h2) =
            (env("LC_HL").unwrap_or(tuning.h_light), env("LC_HH").unwrap_or(tuning.h_heavy), env("LC_H2").unwrap_or(tuning.h2));
        (tuning.light, tuning.heavy) = (env("LC_LIGHT").unwrap_or(tuning.light), env("LC_HEAVY").unwrap_or(tuning.heavy));
        (tuning.half, tuning.half2) = (env("LC_P1").map_or(tuning.half, |v| v as usize), env("LC_P2").map_or(tuning.half2, |v| v as usize));
        (tuning.loose, tuning.few) = (env("LC_LOOSE").unwrap_or(tuning.loose), env("LC_FEW").unwrap_or(tuning.few));
        (tuning.search_c, tuning.h_c) = (env("LC_SC").map_or(tuning.search_c, |v| v as usize), env("LC_HC").unwrap_or(tuning.h_c));
        // LC_NLM=1 runs the old non-local-means filter instead; LC_B_* override the block-matching filter's parameters
        if env("LC_NLM").is_some_and(|v| v > 0.0) {
            tuning.bm3d = None;
        }
        if let Some(b) = tuning.bm3d.as_mut() {
            (b.block, b.step, b.search, b.group) = (
                env("LC_B_N").map_or(b.block, |v| v as usize),
                env("LC_B_STEP").map_or(b.step, |v| v as usize),
                env("LC_B_S").map_or(b.search, |v| v as usize),
                env("LC_B_K").map_or(b.group, |v| v as usize),
            );
            (b.tau1, b.tau2) = (env("LC_B_T1").unwrap_or(b.tau1), env("LC_B_T2").unwrap_or(b.tau2));
            // LC_B_LOFF adds to every point of the threshold curve (a sweep of how hard it cleans)
            let off = env("LC_B_LOFF").unwrap_or(0.0);
            for point in &mut b.lambda {
                point.1 += off;
            }
            b.sigma = env("LC_B_SIG").unwrap_or(b.sigma);
            b.wiener = env("LC_B_WIENER").is_none_or(|v| v > 0.0);
        }
        let runner = Builtin { noise: Some(est), tuning };
        // LC_GPU=1 (with `--features denoise`) runs the block-matching filter on the graphics card
        #[cfg(feature = "denoise")]
        let out = match &gpu {
            Some(g) => {
                let on_card = lightcraft_gpu::bm3d::BuiltinRunner::new(g.clone(), runner).unwrap();
                denoise_bayer(&noisy, w, h, layout, &on_card, &params, &Control::default()).unwrap()
            }
            None => denoise_bayer(&noisy, w, h, layout, &runner, &params, &Control::default()).unwrap(),
        };
        #[cfg(not(feature = "denoise"))]
        let out = denoise_bayer(&noisy, w, h, layout, &runner, &params, &Control::default()).unwrap();
        let secs = started.elapsed().as_secs_f64();
        // each installed model on the same noisy crop, told the same measured noise
        let others: Vec<(String, Vec<[f32; 3]>, f64)> = models
            .iter()
            .map(|m| {
                let runner = (m.runner)(Some(est));
                let p = Params { tile: m.tile, overlap: m.overlap, gain: m.gain, clip: Some(0.99), parallel: 4 };
                let started = Instant::now();
                let o = denoise_bayer(&noisy, w, h, layout, runner.as_ref(), &p, &Control::default()).unwrap();
                (m.name.clone(), o.data, started.elapsed().as_secs_f64())
            })
            .collect();
        let noisy_rgb = plain(&noisy, w, h);
        let (p_noisy, p_out) = (psnr(&noisy_rgb, &truth_rgb, w, h, scale), psnr(&out.data, &truth_rgb, w, h, scale));
        let detail = gradient(&out.data, w, h, scale) / gradient(&truth_rgb, w, h, scale);
        let tex = texture(&out.data, &truth_rgb, w, h, scale);
        println!(
            "{:<14}{:>10.2}{:>10.2}{:>+9.2}{:>9.3}{:>9.3}{:>9.2}   noise a {:.4} (true {a}), b {:.5} (true {b})",
            name,
            p_noisy,
            p_out,
            p_out - p_noisy,
            detail,
            tex,
            secs,
            est.0,
            est.1
        );
        for (name, data, secs) in &others {
            let p = psnr(data, &truth_rgb, w, h, scale);
            let detail = gradient(data, w, h, scale) / gradient(&truth_rgb, w, h, scale);
            println!(
                "{:<14}{:>10.2}{:>10.2}{:>+9.2}{:>9.3}{:>9.3}{:>9.2}   {name}",
                "",
                p_noisy,
                p,
                p - p_noisy,
                detail,
                texture(data, &truth_rgb, w, h, scale),
                secs
            );
        }
        if let Some(dir) = &out_dir {
            let side = 400;
            let at = (w / 2 - side / 2, h / 2 - side / 2);
            let mut tiles =
                vec![crop_rgb8(&truth_rgb, w, at, side, scale), crop_rgb8(&noisy_rgb, w, at, side, scale), crop_rgb8(&out.data, w, at, side, scale)];
            tiles.extend(others.iter().map(|(_, data, _)| crop_rgb8(data, w, at, side, scale)));
            let mut sheet = Vec::with_capacity(side * side * 3 * tiles.len());
            for y in 0..side {
                for t in &tiles {
                    sheet.extend_from_slice(&t[y * side * 3..(y + 1) * side * 3]);
                }
            }
            let img = EncodeImage::new((tiles.len() * side) as u32, side as u32, 3, Samples::U8(&sheet));
            std::fs::create_dir_all(dir).unwrap();
            std::fs::write(format!("{dir}/{name}.png"), encode_png(&img, &EncodeMeta::default()).unwrap()).unwrap();
        }
    }
}

// ---------------------------------------------------------------------------------------------------------------------
// the denoisers scored, and real pairs

/// The built-in denoiser, as the harness runs it: told the whole frame's noise.
fn builtin_denoiser() -> pairs::Denoiser {
    pairs::Denoiser {
        name: "builtin".into(),
        tile: TILE as usize,
        overlap: OVERLAP as usize,
        gain: Gain::None,
        runner: Box::new(|noise| Box::new(Builtin { noise, tuning: Tuning::default() }) as Box<dyn TileRunner>),
    }
}

/// The models installed in `LC_DENOISE_MODELS` (else the user's models folder), each as the app runs it: an ONNX model
/// on our CPU executor, a candle network on Metal when there is one (`LC_CPU=1`: the processor) told the frame's noise.
/// A model that does not load is said so and left out. Without the `denoise` feature: none.
fn installed_models() -> Vec<pairs::Denoiser> {
    #[cfg(feature = "denoise")]
    {
        use lightcraft_denoise::candle::{CandleRunner, Device, gpu_device};
        use lightcraft_denoise::manifest::Network;
        let Some(dir) = std::env::var_os("LC_DENOISE_MODELS").map(PathBuf::from).or_else(lightcraft_engine::config::default_denoise_models_dir)
        else {
            return Vec::new();
        };
        let mut out = Vec::new();
        let mut folders: Vec<PathBuf> = std::fs::read_dir(&dir).map(|r| r.flatten().map(|e| e.path()).collect()).unwrap_or_default();
        folders.sort();
        for f in folders {
            let Ok(bytes) = std::fs::read(f.join("denoise-model.json")) else { continue };
            let m = match lightcraft_denoise::manifest::parse(&bytes) {
                Ok(m) => m,
                Err(e) => {
                    eprintln!("{}: {e}", f.display());
                    continue;
                }
            };
            let file = f.join(m.network.file_name());
            let (tile, overlap, gain, name) = (m.tile as usize, m.overlap as usize, m.gain, m.id.clone());
            let runner: Box<dyn Fn(Option<(f32, f32)>) -> Box<dyn TileRunner>> = match m.network {
                Network::Onnx => match lightcraft_denoise::runtime::CpuRunner::load(&file, &m) {
                    Ok(r) => Box::new(move |_| Box::new(r.clone()) as Box<dyn TileRunner>),
                    Err(e) => {
                        eprintln!("{}: {e}", file.display());
                        continue;
                    }
                },
                Network::Safetensors { .. } => {
                    let device = if std::env::var("LC_CPU").is_ok_and(|v| v != "0") { Device::Cpu } else { gpu_device().unwrap_or(Device::Cpu) };
                    match CandleRunner::load(&file, &m, &device) {
                        Ok(r) => Box::new(move |noise| Box::new(r.with_noise(noise)) as Box<dyn TileRunner>),
                        Err(e) => {
                            eprintln!("{}: {e}", file.display());
                            continue;
                        }
                    }
                }
            };
            out.push(pairs::Denoiser { name, tile, overlap, gain, runner });
        }
        out
    }
    #[cfg(not(feature = "denoise"))]
    {
        Vec::new()
    }
}

/// Scores real pairs: `LC_DENOISE_PAIRS=<folder>` of tripod shots at base ISO and high ISOs (see `pairs.rs` for how they
/// are matched). Opt-in; prints a row per pair and denoiser, then the mean per ISO. `LC_DENOISE_OUT` receives a PNG per
/// pair (reference | noisy | each denoiser); `LC_PAIR_WINDOW=WxH` (default 1536x1024), `LC_PAIR_SHIFT` (cells, default 4)
/// and `LC_PAIR_GAP` (seconds, default 600) tune the matching and scoring.
#[test]
#[ignore = "needs a folder of real pairs: LC_DENOISE_PAIRS=<folder>"]
fn real_pairs_scored_against_their_base_iso_reference() {
    let Some(dir) = std::env::var_os("LC_DENOISE_PAIRS").map(PathBuf::from) else { return };
    let window = std::env::var("LC_PAIR_WINDOW")
        .ok()
        .and_then(|v| v.split_once('x').and_then(|(a, b)| Some((a.parse().ok()?, b.parse().ok()?))))
        .unwrap_or((1536, 1024));
    let env = |k: &str, d: i64| std::env::var(k).ok().and_then(|v| v.parse().ok()).unwrap_or(d);
    let opts = pairs::Opts {
        window,
        max_shift: env("LC_PAIR_SHIFT", 4) as usize,
        gap: env("LC_PAIR_GAP", 600),
        out: std::env::var_os("LC_DENOISE_OUT").map(PathBuf::from),
    };
    let mut denoisers = vec![builtin_denoiser()];
    denoisers.extend(installed_models());
    let rows = pairs::score_folder(&dir, &denoisers, &opts).unwrap_or_else(|e| panic!("{e}"));
    println!("{}", pairs::report(&rows));
}

// ---------------------------------------------------------------------------------------------------------------------
// the harness on synthetic pairs (always run)

/// A clean synthetic scene (texture, edges, a ramp) at `(x, y)` for colour plane `p` (RGGB), white 1.
fn scene(x: f32, y: f32, p: usize) -> f32 {
    let level = [0.45, 0.8, 0.8, 0.6][p];
    let waves = 0.5 + 0.25 * (x / 7.3).sin() * (y / 5.1).sin() + 0.08 * (x * 0.9 + y * 0.7).sin();
    let edge = if ((x / 40.0).floor() + (y / 32.0).floor()) as i64 % 2 == 0 { 0.15 } else { 0.0 };
    (level * (0.15 + 0.6 * waves + edge) * (0.6 + 0.4 * x / 512.0)).clamp(0.0, 1.0)
}

/// One synthetic shot of [`scene`]: seen from `shift`, `exposure` times as bright, with noise `(a, b)` (variance =
/// a · signal + b), at the EXIF `iso` and capture minute, its noise drawn from `seed`.
struct Take {
    shift: (f32, f32),
    exposure: f32,
    noise: (f32, f32),
    iso: u32,
    minute: u8,
    seed: u64,
}

/// A synthetic Bayer DNG (14-bit, RGGB, `w × h`) of a [`Take`], with the EXIF ISO, camera and capture time.
fn synthetic_dng(w: usize, h: usize, take: Take) -> Vec<u8> {
    let Take { shift: (dx, dy), exposure, noise: (a, b), iso, minute, seed } = take;
    use lightcraft_raw::{BlackLevel, Cfa, OpcodeLists, RawData, RawFormat, RawImage, Rect};
    let (black, white) = (512f32, 16383f32);
    let mut rng = Rng(seed);
    let data: Vec<u16> = (0..w * h)
        .map(|i| {
            let (x, y) = (i % w, i / w);
            let s = scene(x as f32 + dx, y as f32 + dy, x % 2 + 2 * (y % 2)) * exposure;
            let v = s + rng.normal() * (a * s + b).sqrt();
            (black + v * (white - black)).round().clamp(0.0, 16383.0) as u16
        })
        .collect();
    let full = Rect::new(0, 0, w, h);
    let raw = RawImage {
        format: RawFormat::Dng,
        width: w,
        height: h,
        cpp: 1,
        data: RawData::U16(data),
        cfa: Cfa::bayer("RGGB"),
        bits: 14,
        black: BlackLevel::uniform(black),
        white: vec![white],
        active_area: full,
        crop: full,
        orientation: lightcraft_raw::Orientation::Normal,
        color: Default::default(),
        wb_multipliers: None,
        linearized: false,
        pixel_aspect: 1.0,
        opcodes: OpcodeLists::default(),
        metadata: lightcraft_meta::Metadata {
            make: Some("LightCraft".into()),
            model: Some("Synthetic Pair".into()),
            iso: Some(iso),
            focal_length: Some(50.0),
            f_number: Some(8.0),
            capture_time: Some(lightcraft_meta::DateTime { year: 2026, month: 11, day: 9, hour: 15, minute, ..Default::default() }),
            ..Default::default()
        },
        lens_metadata: Default::default(),
    };
    lightcraft_raw::write_dng(&raw, &Default::default()).unwrap()
}

fn temp_dir(name: &str) -> PathBuf {
    let d = std::env::temp_dir().join(format!("lc-denoise-pairs-{name}-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&d);
    std::fs::create_dir_all(&d).unwrap();
    d
}

#[test]
fn iso_tokens_name_the_scene() {
    assert_eq!(pairs::iso_token("Kitchen_ISO12800"), Some(("kitchen".into(), 12800)));
    assert_eq!(pairs::iso_token("iso100-window"), Some(("window".into(), 100)));
    assert_eq!(pairs::iso_token("street iso 6400 b"), Some(("street_b".into(), 6400)));
    assert_eq!(pairs::iso_token("IMG_1234"), None);
    assert_eq!(pairs::iso_token("isolated"), None, "an `iso` without digits is not one");
    assert_eq!(pairs::iso_token("isolated_iso400"), Some(("isolated".into(), 400)));
}

#[test]
fn shots_are_grouped_by_folder_name_and_exif() {
    let shot = |name: &str, folder: Option<&str>, iso: u32, camera: &str, time: Option<i64>| pairs::Shot {
        path: PathBuf::from(format!("/p/{name}.cr3")),
        folder: folder.map(str::to_string),
        iso,
        named: pairs::iso_token(name).map(|t| t.0),
        camera: camera.into(),
        size: (6000, 4000),
        lens: (500, 80),
        time,
    };
    let shots = vec![
        shot("a", Some("street"), 100, "Canon EOS R", None),
        shot("b", Some("street"), 6400, "Canon EOS R", None),
        shot("desk_iso100", None, 100, "Canon EOS R6m3", None),
        shot("desk_iso400", None, 400, "Canon EOS R6m3", None),
        shot("desk_iso25600", None, 25600, "Canon EOS R6m3", None),
        // by EXIF: three shots a minute apart, then one an hour later (another scene), and one of another camera
        shot("IMG_1", None, 100, "Canon EOS R", Some(1000)),
        shot("IMG_2", None, 3200, "Canon EOS R", Some(1060)),
        shot("IMG_3", None, 12800, "Canon EOS R", Some(1120)),
        shot("IMG_9", None, 12800, "Canon EOS R", Some(5000)),
        shot("IMG_7", None, 800, "Canon EOS M50m2", Some(1000)),
        // two base-ISO shots of one scene are both reference
        shot("lamp_iso100_a", None, 100, "Canon EOS R", None),
        shot("lamp_iso100_b", None, 100, "Canon EOS R", None),
    ];
    let mut scenes = pairs::group(shots, 600);
    scenes.sort_by(|a, b| a.name.cmp(&b.name));
    let summary: Vec<(String, Vec<u32>, Vec<u32>)> =
        scenes.iter().map(|s| (s.name.clone(), s.reference.iter().map(|r| r.iso).collect(), s.noisy.iter().map(|r| r.iso).collect())).collect();
    assert_eq!(
        summary,
        vec![
            ("Canon EOS R IMG_1".to_string(), vec![100], vec![3200, 12800]),
            ("desk".to_string(), vec![100], vec![400, 25600]),
            ("street".to_string(), vec![100], vec![6400]),
        ],
        "IMG_9 (alone an hour later), IMG_7 (another camera) and the lamp (one ISO only) are no pair"
    );
}

#[test]
fn registration_finds_the_shift_and_the_exposure() {
    let dir = temp_dir("align");
    std::fs::write(
        dir.join("r.dng"),
        synthetic_dng(256, 192, Take { shift: (0.0, 0.0), exposure: 1.0, noise: (1e-6, 1e-8), iso: 100, minute: 0, seed: 1 }),
    )
    .unwrap();
    std::fs::write(
        dir.join("n.dng"),
        synthetic_dng(256, 192, Take { shift: (4.0, -2.0), exposure: 0.8, noise: (4.9e-4, 1e-6), iso: 6400, minute: 1, seed: 2 }),
    )
    .unwrap();
    let (r, n) = (pairs::load(&dir.join("r.dng")).unwrap(), pairs::load(&dir.join("n.dng")).unwrap());
    assert_eq!((r.width, r.height, r.layout), (256, 192, Layout { dx: 0, dy: 0 }));
    let win = n.centre(200, 160, 10).unwrap();
    assert_eq!((win.x % 2, win.y % 2), (0, 0), "the window starts on a red cell");
    assert_eq!(pairs::align(&r, &n, win, 4), (4, -2));
    let g = pairs::gains(&r.cut(win, 4, -2).unwrap(), &n.cut(win, 0, 0).unwrap(), win.w);
    assert!(g.iter().all(|g| (g - 0.8).abs() < 0.02), "{g:?}");
    let _ = std::fs::remove_dir_all(dir);
}

/// The whole harness on a folder of synthetic DNG pairs: matched by name, registered, scored, PNGs written. The built-in
/// denoiser must gain over the noisy shot and keep most of the texture.
#[test]
fn the_harness_scores_a_folder_of_synthetic_pairs() {
    let dir = temp_dir("folder");
    let out = dir.join("out");
    let (w, h) = (384, 256);
    std::fs::write(
        dir.join("bench_iso100.dng"),
        synthetic_dng(w, h, Take { shift: (0.0, 0.0), exposure: 1.0, noise: (4e-6, 1e-8), iso: 100, minute: 0, seed: 3 }),
    )
    .unwrap();
    std::fs::write(
        dir.join("bench_iso6400.dng"),
        synthetic_dng(w, h, Take { shift: (2.0, -2.0), exposure: 0.9, noise: (4.9e-4, 1e-6), iso: 6400, minute: 1, seed: 4 }),
    )
    .unwrap();
    std::fs::write(
        dir.join("bench_iso25600.dng"),
        synthetic_dng(w, h, Take { shift: (-2.0, 0.0), exposure: 1.1, noise: (1.19e-3, 1.7e-5), iso: 25600, minute: 2, seed: 5 }),
    )
    .unwrap();
    std::fs::write(dir.join("notes.txt"), b"not a raw").unwrap();
    let opts = pairs::Opts { window: (320, 224), max_shift: 3, gap: 600, out: Some(out.clone()) };
    let rows = pairs::score_folder(&dir, &[builtin_denoiser()], &opts).unwrap();
    println!("{}", pairs::report(&rows));
    assert_eq!(rows.len(), 2);
    for r in &rows {
        assert_eq!((r.scene.as_str(), r.reference_iso, r.denoiser.as_str()), ("bench", 100, "builtin"));
        assert_eq!(r.shift, if r.iso == 6400 { (2, -2) } else { (-2, 0) }, "{r:?}");
        assert!(r.db > r.noisy_db + 0.5, "{r:?}");
        assert!(r.texture > 0.3 && r.texture < 1.2 && r.detail.is_finite(), "{r:?}");
        assert!(r.noise.is_some_and(|(a, _)| a > 0.0), "the frame's noise is measured: {r:?}");
        assert!(out.join(format!("bench-iso{}.png", r.iso)).is_file());
    }
    let iso_25600 = rows.iter().find(|r| r.iso == 25600).unwrap();
    let iso_6400 = rows.iter().find(|r| r.iso == 6400).unwrap();
    assert!(iso_25600.noisy_db < iso_6400.noisy_db, "more noise at the higher ISO");
    assert!(iso_25600.db - iso_25600.noisy_db > iso_6400.db - iso_6400.noisy_db, "and more to gain there");
    assert!(pairs::report(&rows).contains("builtin"));
    // a folder with no pair is an error that says so
    let lonely = temp_dir("lonely");
    std::fs::write(
        lonely.join("a_iso100.dng"),
        synthetic_dng(128, 128, Take { shift: (0.0, 0.0), exposure: 1.0, noise: (1e-5, 1e-8), iso: 100, minute: 0, seed: 6 }),
    )
    .unwrap();
    assert!(pairs::score_folder(&lonely, &[builtin_denoiser()], &opts).unwrap_err().contains("no scene"));
    let _ = std::fs::remove_dir_all(dir);
    let _ = std::fs::remove_dir_all(lonely);
}

/// The harness keeps working on paths it is handed (it never needs them to be in the repository).
#[test]
fn a_missing_folder_is_an_error() {
    assert!(
        pairs::score_folder(Path::new("/does/not/exist"), &[builtin_denoiser()], &pairs::Opts { window: (64, 64), max_shift: 1, gap: 1, out: None })
            .is_err()
    );
}
