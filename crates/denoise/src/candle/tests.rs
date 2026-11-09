use super::pmrid::Pmrid;
use super::*;
use crate::licence::Licence;
use crate::manifest::{Domain, Gain};

fn manifest(tile: u32, noise: NoiseMap, input_scale: f32) -> DenoiserManifest {
    DenoiserManifest {
        id: "pmrid-test".into(),
        name: "PMRID test".into(),
        version: "1".into(),
        licence: Licence::default(),
        source: None,
        sha256: None,
        size_bytes: None,
        provenance: "random weights, for tests".into(),
        domain: Domain::BayerToRgb,
        tile,
        overlap: 16,
        gain: Gain::None,
        network: Network::Safetensors { architecture: Architecture::Pmrid, noise, input_scale },
    }
}

fn random_weights(seed: u64, scale: f32) -> Weights {
    super::pmrid::random_weights(seed, scale).unwrap()
}

fn tile(t: usize) -> Vec<f32> {
    crate::runtime::check_tile(t)
}

#[test]
fn zero_weights_give_the_input_back_through_the_mapping_and_the_demosaic() {
    // with every weight 0 the network predicts no noise: the answer is the (clipped) input, mapped and mapped back
    let t = 64;
    let m = manifest(t as u32, NoiseMap::KSigma { a: 1e-3, b: 3.2e-6 }, 256.0);
    let r = CandleRunner::from_weights(&mut random_weights(1, 0.0), &m, &Device::Cpu).unwrap().with_noise(Some((2e-4, 3e-7)));
    let input = tile(t);
    let out = r.run(&input).unwrap();
    let planes: Vec<Vec<f32>> = input.chunks_exact(t * t).map(|p| p.iter().map(|v| v.clamp(0.0, 1.0)).collect()).collect();
    let want = builtin::demosaic(&planes, t);
    assert_eq!(out.len(), 3 * 4 * t * t);
    let worst = out.iter().zip(&want).fold(0f32, |m, (a, b)| m.max((a - b).abs()));
    assert!(worst < 1e-5, "{worst}");
}

#[test]
fn the_k_sigma_transform_gives_the_training_noise() {
    let train = (1.0094e-3f32, 3.239e-6f32);
    // measured Canon EOS R levels, ISO 1600 to 51200 (docs/denoise.md)
    for photo in [(2.1e-4f32, 3e-7f32), (4.9e-4, 1e-6), (1.19e-3, 1.7e-5), (6.9e-3, 2.08e-4)] {
        let (s, o) = k_sigma(photo, train);
        for x in [0.01f32, 0.18, 0.7] {
            let var_mapped = s * s * (photo.0 * x + photo.1);
            let want = train.0 * (s * x + o) + train.1;
            assert!((var_mapped - want).abs() <= 1e-3 * want, "{photo:?} at {x}: {var_mapped} vs {want}");
        }
    }
    // a clean photo is brightened by at most MAX_GAIN, and noise that is all floor gets a bounded offset
    assert_eq!(k_sigma((1e-9, 1e-9), train).0, MAX_GAIN);
    let (s, o) = k_sigma((0.0, 8.3e-4), train);
    assert!(s.is_finite() && o.is_finite() && s < 1.0 && o < 0.2, "{s} {o}");
    for bad in [(f32::NAN, 0.0), (0.0, f32::INFINITY), (-1.0, -1.0)] {
        let (s, o) = k_sigma(bad, train);
        assert!(s.is_finite() && o.is_finite() && s > 0.0, "{bad:?}");
    }
    assert_eq!(k_sigma((1e-3, 1e-6), (0.0, 0.0)), (1.0, 0.0), "a training noise that is not one maps nothing");
}

#[test]
fn hostile_weights_are_errors_never_panics() {
    let m = manifest(64, NoiseMap::None, 1.0);
    let good = random_weights(3, 0.1);
    assert!(CandleRunner::from_weights(&mut Weights::parse(&good.to_bytes()).unwrap(), &m, &Device::Cpu).is_ok());
    // a missing tensor, a wrong shape, a tensor too many
    let names = Pmrid::parameters().unwrap();
    let (first, shape) = names.first().cloned().unwrap();
    let mut missing = Weights::parse(&good.to_bytes()).unwrap();
    missing.take(&first, &shape).unwrap();
    let e = CandleRunner::from_weights(&mut missing, &m, &Device::Cpu).err().unwrap().to_string();
    assert!(e.contains(&first), "{e}");
    let mut wrong = Weights::parse(&good.to_bytes()).unwrap();
    let v = wrong.take(&first, &shape).unwrap();
    wrong.insert(&first, &[v.len()], v).unwrap();
    assert!(CandleRunner::from_weights(&mut wrong, &m, &Device::Cpu).err().unwrap().to_string().contains("shape"));
    let mut extra = Weights::parse(&good.to_bytes()).unwrap();
    extra.insert("head.weight", &[2], vec![0.0; 2]).unwrap();
    assert!(CandleRunner::from_weights(&mut extra, &m, &Device::Cpu).err().unwrap().to_string().contains("head.weight"));
    // an ONNX manifest, or a tile the network cannot take
    let onnx = DenoiserManifest { network: Network::Onnx, ..m.clone() };
    assert!(CandleRunner::from_weights(&mut Weights::parse(&good.to_bytes()).unwrap(), &onnx, &Device::Cpu).is_err());
    let odd = DenoiserManifest { tile: 72, ..m.clone() };
    assert!(CandleRunner::from_weights(&mut Weights::parse(&good.to_bytes()).unwrap(), &odd, &Device::Cpu).is_err());
    // damaged files
    let bytes = good.to_bytes();
    let path = std::env::temp_dir().join(format!("lc-candle-hostile-{}.safetensors", std::process::id()));
    for cut in [0, 7, 8, 100, bytes.len() / 2, bytes.len() - 1] {
        std::fs::write(&path, &bytes[..cut]).unwrap();
        assert!(CandleRunner::load(&path, &m, &Device::Cpu).is_err(), "cut at {cut}");
    }
    std::fs::write(&path, &bytes).unwrap();
    let r = CandleRunner::load(&path, &m, &Device::Cpu).unwrap();
    // a tile of the wrong size
    assert!(matches!(r.run(&[0.5; 10]), Err(Error::Input(_))));
    let _ = std::fs::remove_file(path);
}

#[test]
fn the_runner_is_repeatable_and_finite_and_passes_the_shared_self_test_checks_it_can() {
    let m = manifest(64, NoiseMap::KSigma { a: 1e-3, b: 3.2e-6 }, 256.0);
    let r = CandleRunner::from_weights(&mut random_weights(5, 0.05), &m, &Device::Cpu).unwrap();
    let test = r.self_test();
    // random weights do not clean anything, but they run, give finite numbers and the same answer twice
    assert!(test.checks.iter().any(|(c, ok)| c.contains("finite") && *ok), "{test:?}");
    assert!(test.checks.iter().any(|(c, ok)| c.contains("same answer") && *ok), "{test:?}");
    // the photo's noise changes the answer; numbers that are not a noise are ignored
    let x = tile(64);
    let told = r.with_noise(Some((5e-4, 1e-6)));
    assert_ne!(told.run(&x).unwrap(), r.run(&x).unwrap());
    assert_eq!(r.with_noise(Some((f32::NAN, 1.0))).noise(), None);
}

#[test]
fn metal_and_the_processor_agree() {
    let device = match gpu_device() {
        Ok(d) => d,
        Err(why) => {
            lightcraft_testkit::skip!(Gpu, "no candle graphics card: {why}");
            return;
        }
    };
    let m = manifest(64, NoiseMap::KSigma { a: 1e-3, b: 3.2e-6 }, 256.0);
    let w = random_weights(9, 0.1);
    let cpu = CandleRunner::from_weights(&mut Weights::parse(&w.to_bytes()).unwrap(), &m, &Device::Cpu).unwrap();
    let gpu = CandleRunner::from_weights(&mut Weights::parse(&w.to_bytes()).unwrap(), &m, &device).unwrap();
    assert!(gpu.on_gpu());
    let (worst, mean, _, _) = compare(&cpu, &gpu, &tile(64)).unwrap();
    assert!(worst < 1e-3 && mean < 1e-5, "worst {worst}, mean {mean}");
    // and from several threads at once, as tiles run
    let x = tile(64);
    let want = gpu.run(&x).unwrap();
    std::thread::scope(|s| {
        let runs: Vec<_> = (0..4).map(|_| s.spawn(|| gpu.run(&x).unwrap())).collect();
        for r in runs {
            let got = r.join().unwrap();
            let worst = got.iter().zip(&want).fold(0f32, |m, (a, b)| m.max((a - b).abs()));
            assert!(worst < 1e-4, "threads change the answer by {worst}");
        }
    });
}

/// Opt-in: `LC_PMRID=<folder with model.safetensors and denoise-model.json> cargo test --release -p lightcraft-denoise
/// --features candle --lib real_pmrid -- --ignored --nocapture` (the folder `examples/pmrid-convert.rs` writes).
#[test]
#[ignore = "needs converted PMRID weights: set LC_PMRID"]
fn a_real_pmrid_model_passes_the_self_test_on_each_device() {
    let Some(dir) = std::env::var_os("LC_PMRID").map(std::path::PathBuf::from) else { return };
    let m = crate::manifest::parse(&std::fs::read(dir.join("denoise-model.json")).unwrap()).unwrap();
    let mut devices = vec![Device::Cpu];
    devices.extend(gpu_device().ok());
    for d in devices {
        let r = CandleRunner::load(&dir.join("model.safetensors"), &m, &d).unwrap();
        let t = r.self_test();
        println!("{:?}: load {:.0} ms | {:.0} ms per {} tile | {:?}", d.location(), t.load_ms, t.tile_ms, m.tile, t.checks);
        assert!(t.ok, "{t:?}");
    }
}

/// Opt-in, with `LC_PMRID` as above: five warm tiles on each device, in milliseconds.
#[test]
#[ignore = "timing: set LC_PMRID"]
fn a_real_pmrid_tile_timed_on_each_device() {
    let Some(dir) = std::env::var_os("LC_PMRID").map(std::path::PathBuf::from) else { return };
    let m = crate::manifest::parse(&std::fs::read(dir.join("denoise-model.json")).unwrap()).unwrap();
    let mut devices = vec![Device::Cpu];
    devices.extend(gpu_device().ok());
    for d in devices {
        let r = CandleRunner::load(&dir.join("model.safetensors"), &m, &d).unwrap();
        let x = tile(m.tile as usize);
        let times: Vec<f64> = (0..5)
            .map(|_| {
                let s = std::time::Instant::now();
                r.run(&x).unwrap();
                s.elapsed().as_secs_f64() * 1e3
            })
            .collect();
        println!("{:?}: {times:.0?}", d.location());
    }
}

#[test]
fn the_blur_weights_make_a_network_that_passes_the_self_test() {
    let m = manifest(64, NoiseMap::None, 1.0);
    let r = CandleRunner::from_weights(&mut super::pmrid::blur_weights().unwrap(), &m, &Device::Cpu).unwrap();
    let t = r.self_test();
    assert!(t.ok, "{t:?}");
    // a flat tile stays flat inside (the blur's edges see the zero padding)
    let out = r.run(&vec![0.3; 4 * 64 * 64]).unwrap();
    assert!((out[128 * 128 + 64 * 128 + 64] - 0.3).abs() < 1e-4, "{}", out[128 * 128 + 64 * 128 + 64]);
}
