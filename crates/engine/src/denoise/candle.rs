//! A denoise model that is a candle network (`lightcraft_denoise::candle`, safetensors weights; PMRID): on the
//! processor, and on the graphics card (Metal on macOS) once it has been set up for it, checked against the processor
//! on a check tile and timed, like an ONNX model's card ([`super::model::Network`]).
//!
//! The network adapts to the photo's noise (the k-sigma transform), so each photo gets a model of its own
//! ([`Model::for_photo`]) that shares the loaded weights: [`Model::prepare`] measures the whole photo's noise once and
//! every tile is mapped by it. Without it (the self-test, a check tile) each tile measures its own.

use lightcraft_denoise::candle::{CandleRunner, gpu_device};
use lightcraft_denoise::runtime::check_tile;

use super::gpu::{GPU_SETUP_LIMIT, GPU_SETUP_MARKER, card_tile_ms, set_up_card};
use super::model::{Model, RunOn, worth_retrying};
use super::{Arc, DenoiserManifest, OnceLock, Path, PathBuf, RunError, TileRunner, Value, json};

/// How far the card's answer on the check tile may be from the processor's (white = 1), at the worst place and on
/// average, before it is not used. The two round differently through some fifty layers (random weights: under 1e-3 at
/// worst, `metal_and_the_processor_agree`); overlapping passes were off by 1.3, and a wrong kernel would be too.
const AGREE_WORST: f32 = 1e-2;
const AGREE_MEAN: f32 = 1e-4;
/// Tiles at once on the card: its network passes run one at a time, and a second tile is mapped and demosaiced meanwhile.
const CARD_PARALLEL: usize = 2;

struct Card {
    runner: CandleRunner,
    adapter: String,
    /// The card's warm time for the check tile, and the processor's for the same tile alone (ms).
    card_ms: f64,
    cpu_ms: f64,
}

/// What every photo's model shares: the network on the processor, and on the card once it is set up.
struct Shared {
    cpu: CandleRunner,
    path: PathBuf,
    card: OnceLock<Result<Card, String>>,
}

impl Shared {
    /// The card when it can be used now, else why not; `start` sets it up when that has not been tried.
    fn card(&self, start: bool) -> Result<&Card, String> {
        let got = if start { Some(self.card.get_or_init(|| make_card(&self.cpu, &self.path))) } else { self.card.get() };
        match got {
            None => Err("the graphics card is not set up yet".into()),
            Some(Err(why)) => Err(why.clone()),
            Some(Ok(c)) => Ok(c),
        }
    }

    /// The card, when a photo made under `run_on` runs on it; else why the processor does.
    fn choose(&self, run_on: RunOn, start: bool) -> Result<&Card, String> {
        if run_on == RunOn::Cpu {
            return Err("the processor is chosen in Settings".into());
        }
        let c = self.card(start)?;
        // candle's processor kernels already use every core for one tile, so tiles at once do not add up as they do for
        // the single-threaded ONNX executor (`card_is_faster`): the card's tile against the processor's
        if run_on == RunOn::Auto && !(c.card_ms.is_finite() && c.card_ms <= c.cpu_ms) {
            return Err(format!("the processor is faster than {} on this computer", c.adapter));
        }
        Ok(c)
    }
}

/// Set the card up for the network at `path`: loaded there, its kernels built on the check tile, compared with the
/// processor's answer and both timed, on a thread of its own with the set-up marker and time limit of every card.
fn make_card(cpu: &CandleRunner, path: &Path) -> Result<Card, String> {
    let manifest: DenoiserManifest = cpu.manifest().clone();
    let tile = Arc::new(check_tile(manifest.tile as usize));
    let (on_card, file) = (tile.clone(), path.to_path_buf());
    let card = move || -> Result<(CandleRunner, Vec<f32>, f64), String> {
        let device = gpu_device()?;
        let runner = CandleRunner::load(&file, &manifest, &device).map_err(|e| e.to_string())?;
        let got = runner.run(&on_card).map_err(|e| format!("the graphics card could not run the check tile: {e}"))?;
        let card_ms = card_tile_ms(&runner, &on_card);
        Ok((runner, got, card_ms))
    };
    let on_cpu = || {
        let started = web_time::Instant::now();
        (cpu.run(&tile), started.elapsed().as_secs_f64() * 1000.0)
    };
    let made = set_up_card(&path.with_file_name(GPU_SETUP_MARKER), GPU_SETUP_LIMIT, card, on_cpu);
    let ((runner, got, card_ms), (want, cpu_ms)) = match made {
        Ok(m) => m,
        Err(why) => {
            log::info!("denoise: the candle network runs on the processor ({why})");
            return Err(why);
        }
    };
    let want = want.map_err(|e| format!("the processor could not check the graphics card: {e}"))?;
    let (worst, sum) = want.iter().zip(&got).fold((0f32, 0f64), |(w, s), (a, b)| ((a - b).abs().max(w), s + f64::from((a - b).abs())));
    let mean = (sum / want.len().max(1) as f64) as f32;
    if got.len() != want.len() || worst > AGREE_WORST || mean > AGREE_MEAN {
        return Err(format!("its answer on the check tile is not the processor's (off by {worst:.1e} at worst, {mean:.1e} on average): not used"));
    }
    Ok(Card { runner, adapter: "Metal (candle)".into(), card_ms, cpu_ms })
}

/// One photo's tiles: on the card when it was chosen, with the processor running any tile the card gets wrong.
struct PhotoRunner {
    card: Option<CandleRunner>,
    cpu: CandleRunner,
}

impl TileRunner for PhotoRunner {
    fn run(&self, input: &[f32]) -> Result<Vec<f32>, RunError> {
        if let Some(c) = &self.card {
            match c.run(input) {
                Ok(out) => return Ok(out),
                Err(e) => log::warn!("denoise: a tile came back wrong from the graphics card ({e}); the processor runs it"),
            }
        }
        self.cpu.run(input)
    }
}

/// A candle network as the engine's [`Model`].
pub(super) struct CandleModel {
    shared: Arc<Shared>,
    /// The photo's noise, measured by [`Model::prepare`].
    noise: OnceLock<(f32, f32)>,
    /// The photo's runner, on the card and on the processor (made on first use, with the noise then known).
    on_card: OnceLock<PhotoRunner>,
    on_cpu: OnceLock<PhotoRunner>,
}

/// Load the candle network at `path` as `manifest` describes it, on the processor (the card is set up for the first
/// photo that may use it).
pub(super) fn load(path: &Path, manifest: &DenoiserManifest) -> Result<CandleModel, String> {
    let cpu = CandleRunner::load(path, manifest, &lightcraft_denoise::candle::Device::Cpu).map_err(|e| e.to_string())?;
    let shared = Shared { cpu, path: path.to_path_buf(), card: OnceLock::new() };
    Ok(CandleModel::of(Arc::new(shared)))
}

impl CandleModel {
    fn of(shared: Arc<Shared>) -> Self {
        CandleModel { shared, noise: OnceLock::new(), on_card: OnceLock::new(), on_cpu: OnceLock::new() }
    }

    fn photo_runner(&self, card: Option<&Card>) -> &PhotoRunner {
        let noise = self.noise.get().copied();
        let cpu = || self.shared.cpu.with_noise(noise);
        match card {
            Some(c) => self.on_card.get_or_init(|| PhotoRunner { card: Some(c.runner.with_noise(noise)), cpu: cpu() }),
            None => self.on_cpu.get_or_init(|| PhotoRunner { card: None, cpu: cpu() }),
        }
    }
}

impl Model for CandleModel {
    fn runner(&self, run_on: RunOn, threads: usize) -> (&dyn TileRunner, usize) {
        let threads = threads.max(1);
        match self.shared.choose(run_on, true) {
            Ok(card) => (self.photo_runner(Some(card)), threads.min(CARD_PARALLEL)),
            Err(_) => (self.photo_runner(None), threads),
        }
    }

    fn for_photo(&self) -> Option<Arc<dyn Model>> {
        Some(Arc::new(CandleModel::of(self.shared.clone())))
    }

    fn prepare(&self, mosaic: &[f32], width: usize, height: usize) {
        if let Some(noise) = lightcraft_denoise::builtin::measure_noise(mosaic, width, height) {
            let _ = self.noise.set(noise);
        }
    }

    fn self_test(&self, run_on: RunOn) -> Result<Value, String> {
        let t = self.shared.cpu.self_test();
        if !t.ok {
            let failed: Vec<&str> = t.checks.iter().filter(|(_, ok)| !*ok).map(|(c, _)| c.as_str()).collect();
            return Err(format!("the model failed its self-test: it does not {}", failed.join(", and does not ")));
        }
        let mut v = serde_json::to_value(&t).map_err(|e| e.to_string())?;
        if run_on != RunOn::Cpu {
            let _ = self.shared.card(true);
        }
        if let Some(o) = v.as_object_mut() {
            o.insert("device".into(), self.device(run_on));
            o.insert("runtime".into(), json!("candle"));
        }
        Ok(v)
    }

    fn bytes_per_tile(&self) -> usize {
        self.shared.cpu.bytes_per_tile()
    }

    fn device(&self, run_on: RunOn) -> Value {
        if run_on != RunOn::Cpu && self.shared.card.get().is_none() {
            return json!({"kind": "pending"});
        }
        let mut v = match self.shared.choose(run_on, false) {
            Ok(c) => json!({"kind": "gpu", "adapter": c.adapter}),
            Err(why) => json!({"kind": "cpu", "retry": worth_retrying(&why), "reason": why}),
        };
        if let (Some(Ok(c)), Some(o)) = (self.shared.card.get(), v.as_object_mut()) {
            o.insert("cardMs".into(), json!(c.card_ms));
            o.insert("cpuMs".into(), json!(c.cpu_ms));
        }
        v
    }
}
