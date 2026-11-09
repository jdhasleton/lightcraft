//! The candle runner: a denoise model that is a network written for LightCraft in candle (pure Rust; Metal on macOS,
//! the processor elsewhere), with its weights in a safetensors file, instead of an ONNX file of the operators our own
//! executor reads ([`crate::onnx`]).
//!
//! It is a [`TileRunner`] with the contract every model has (`bayerToRgb`: four packed planes in, camera RGB at twice
//! their size out), so tiling, scale matching, highlight protection, the cache, the queue and the exports are the same.
//! The manifest says which network the weights are for ([`Architecture`]) and how the photo's noise is brought to the
//! noise it was trained on ([`NoiseMap`]):
//!
//! - **PMRID** ([`pmrid`]) cleans the four planes and gives them back; LightCraft demosaics them with the built-in
//!   denoiser's demosaic ([`builtin::demosaic`]). Its input is clipped to 0..1, as its authors' benchmark does.
//! - **k-sigma** ([`k_sigma`]): the photo's noise (variance = `a · signal + b`) is measured once from the whole photo
//!   (the engine calls [`CandleRunner::with_noise`]; a tile measures its own otherwise) and the samples are mapped so
//!   that their noise is the training noise, then mapped back after the network.
//!
//! Weights and manifests are hostile: the file is bounded and checked ([`weights`]), every tensor must be the one the
//! network asks for with the shape it needs and none may be left over, and an error or a panic inside candle while
//! building or running is an error, never a crash.

/// The safetensors reader the weights go through.
pub use lightcraft_denoise_core::safetensors as weights;
/// The PMRID network (its own crate: Apache-2.0 only, see NOTICE).
pub use lightcraft_denoise_pmrid as pmrid;

use std::panic::{AssertUnwindSafe, catch_unwind};
use std::path::Path;
use std::sync::{Arc, Mutex, PoisonError};

pub use candle_core::Device;
use candle_core::Tensor;
use web_time::Instant;

use crate::builtin;
use crate::manifest::{Architecture, DenoiserManifest, Network, NoiseMap};
use crate::run::{Error, TileRunner};
use crate::runtime::SelfTest;
use weights::Weights;

/// The most the k-sigma transform brightens a photo by (a clean photo would otherwise be scaled far beyond what the
/// network saw: PMRID's training reached 14 times at its sensor's ISO 100).
pub const MAX_GAIN: f32 = 16.0;
/// The most read noise against shot noise (`b / a²`, the read noise in photoelectrons squared) the transform maps:
/// a photo whose measured noise is almost all floor (`a` near 0) would get a huge offset. PMRID's sensor ranges from
/// 2 to 40, Canon EOS R bodies from 4 to 60.
pub const MAX_READ: f32 = 100.0;

/// The k-sigma transform `x' = s · x + o` (as `(s, o)`) that gives a photo with noise `photo` the noise `train`
/// (both `(a, b)`: variance = `a · signal + b`, black 0, white 1): `s = a_train / a_photo`,
/// `o = (s² · b_photo − b_train) / a_train`, so that the variance of `x'` is `a_train · x' + b_train`. The photo's shot
/// noise is taken as at least `a_train / MAX_GAIN` and `sqrt(b_photo / MAX_READ)` (see those).
pub fn k_sigma(photo: (f32, f32), train: (f32, f32)) -> (f32, f32) {
    let (ta, tb) = train;
    if !(ta.is_finite() && ta > 0.0 && tb.is_finite()) {
        return (1.0, 0.0);
    }
    let (pa, pb) = if photo.0.is_finite() && photo.1.is_finite() { (photo.0.max(0.0), photo.1.max(0.0)) } else { train };
    let pa = pa.max(ta / MAX_GAIN).max((pb / MAX_READ).sqrt());
    let s = ta / pa;
    (s, (s * s * pb - tb) / ta)
}

/// Run `f`, turning a panic inside candle into an error.
fn guarded<T>(what: &str, f: impl FnOnce() -> candle_core::Result<T>) -> Result<T, Error> {
    match catch_unwind(AssertUnwindSafe(f)) {
        Ok(Ok(v)) => Ok(v),
        Ok(Err(e)) => Err(Error::Runtime(format!("{what}: {e}"))),
        Err(_) => Err(Error::Runtime(format!("{what}: the network stopped unexpectedly"))),
    }
}

/// Whether this build can run a candle network on a graphics card (Metal on macOS).
pub const GPU_BUILD: bool = cfg!(target_os = "macos");

/// The graphics card candle can use here (Metal on macOS), or why not.
pub fn gpu_device() -> Result<Device, String> {
    #[cfg(target_os = "macos")]
    {
        match catch_unwind(|| Device::new_metal(0)) {
            Ok(Ok(d)) => Ok(d),
            Ok(Err(e)) => Err(format!("no Metal device: {e}")),
            Err(_) => Err("setting up Metal stopped unexpectedly".into()),
        }
    }
    #[cfg(not(target_os = "macos"))]
    {
        Err("this build runs candle networks on the processor only".into())
    }
}

/// A network LightCraft has written, with its weights.
enum Net {
    Pmrid(pmrid::Pmrid),
}

impl Net {
    fn build(architecture: Architecture, w: &mut Weights, device: &Device) -> Result<Net, Error> {
        let built = catch_unwind(AssertUnwindSafe(|| match architecture {
            Architecture::Pmrid => pmrid::Pmrid::new(w, device).map(Net::Pmrid),
        }));
        let net = built.map_err(|_| Error::Load("building the network stopped unexpectedly".into()))??;
        let left = w.left();
        if !left.is_empty() {
            let some: Vec<&str> = left.iter().take(3).copied().collect();
            return Err(Error::Load(format!(
                "the weights hold {} tensor(s) the {} network does not have ({}): is this file for it?",
                left.len(),
                architecture.id(),
                some.join(", ")
            )));
        }
        Ok(net)
    }

    fn forward(&self, x: &Tensor) -> candle_core::Result<Tensor> {
        match self {
            Net::Pmrid(n) => n.forward(x),
        }
    }
}

/// A candle network, on one device, as a tile runner.
#[derive(Clone)]
pub struct CandleRunner {
    net: Arc<Net>,
    device: Device,
    manifest: DenoiserManifest,
    architecture: Architecture,
    noise_map: NoiseMap,
    input_scale: f32,
    /// The whole photo's noise, `(a, b)`; `None`: each tile measures its own.
    noise: Option<(f32, f32)>,
    load_ms: f64,
    /// On a graphics card, one network pass at a time (shared by the clones): candle's Metal backend gives wrong
    /// answers when passes from several threads overlap (`metal_and_the_processor_agree`). The mapping and the
    /// demosaic around it still run in parallel.
    one_pass: Option<Arc<Mutex<()>>>,
}

impl CandleRunner {
    /// Load the weights at `path` for the network `manifest` names, on `device`.
    pub fn load(path: &Path, manifest: &DenoiserManifest, device: &Device) -> Result<Self, Error> {
        let started = Instant::now();
        crate::manifest::validate(manifest).map_err(|e| Error::Load(e.to_string()))?;
        if !matches!(manifest.network, Network::Safetensors { .. }) {
            return Err(Error::Load("the manifest does not describe a safetensors network".into()));
        }
        let mut w = Weights::read(path)?;
        let mut r = Self::from_weights(&mut w, manifest, device)?;
        r.load_ms = started.elapsed().as_secs_f64() * 1e3;
        Ok(r)
    }

    /// [`CandleRunner::load`] from weights in memory.
    pub fn from_weights(w: &mut Weights, manifest: &DenoiserManifest, device: &Device) -> Result<Self, Error> {
        let started = Instant::now();
        crate::manifest::validate(manifest).map_err(|e| Error::Load(e.to_string()))?;
        let Network::Safetensors { architecture, noise, input_scale } = manifest.network else {
            return Err(Error::Load("the manifest does not describe a safetensors network".into()));
        };
        if !(manifest.tile as usize).is_multiple_of(pmrid::MULTIPLE) {
            return Err(Error::Load(format!("{} needs a tile that is a multiple of {}", architecture.id(), pmrid::MULTIPLE)));
        }
        let net = Net::build(architecture, w, device)?;
        Ok(CandleRunner {
            net: Arc::new(net),
            device: device.clone(),
            manifest: manifest.clone(),
            architecture,
            noise_map: noise,
            input_scale,
            noise: None,
            load_ms: started.elapsed().as_secs_f64() * 1e3,
            one_pass: (!device.is_cpu()).then(|| Arc::new(Mutex::new(()))),
        })
    }

    /// The same network, told the whole photo's noise `(a, b)` (variance = `a · signal + b`; see
    /// [`builtin::measure_noise`]). `None` or numbers that are not finite: each tile measures its own.
    pub fn with_noise(&self, noise: Option<(f32, f32)>) -> Self {
        CandleRunner { noise: noise.filter(|(a, b)| a.is_finite() && b.is_finite() && *a >= 0.0 && *b >= 0.0), ..self.clone() }
    }

    pub fn noise(&self) -> Option<(f32, f32)> {
        self.noise
    }

    pub fn manifest(&self) -> &DenoiserManifest {
        &self.manifest
    }

    pub fn architecture(&self) -> Architecture {
        self.architecture
    }

    pub fn device(&self) -> &Device {
        &self.device
    }

    /// Whether it runs on a graphics card.
    pub fn on_gpu(&self) -> bool {
        !self.device.is_cpu()
    }

    /// The installation self-test every model passes (see [`crate::runtime::self_test_of`]).
    pub fn self_test(&self) -> SelfTest {
        crate::runtime::self_test_of(self, &self.manifest, self.load_ms)
    }

    /// Working memory one tile may take: the network's features at the tile's size and the unfolded copy the largest
    /// depthwise convolution makes (25 taps of 16 channels at the tile's size: 1600 bytes a cell), with room to spare.
    pub fn bytes_per_tile(&self) -> usize {
        let t = self.manifest.tile as usize;
        t.saturating_mul(t).saturating_mul(2048)
    }

    /// How the samples of a tile are mapped before the network: `(s, o)` of `x' = s · x + o`.
    pub fn mapping(&self, input: &[f32]) -> (f32, f32) {
        match self.noise_map {
            NoiseMap::None => (1.0, 0.0),
            NoiseMap::KSigma { a, b } => {
                let photo = self.noise.or_else(|| builtin::tile_noise(input)).unwrap_or((a, b));
                k_sigma(photo, (a, b))
            }
        }
    }

    /// Run the network on four planes of `t × t` (already mapped and scaled); gives its four planes.
    fn network(&self, x: Vec<f32>, t: usize) -> Result<Vec<f32>, Error> {
        let _one = self.one_pass.as_ref().map(|m| m.lock().unwrap_or_else(PoisonError::into_inner));
        guarded("the network could not run", || {
            let x = Tensor::from_vec(x, (1, pmrid::PLANES, t, t), &self.device)?;
            self.net.forward(&x)?.flatten_all()?.to_vec1::<f32>()
        })
    }
}

impl TileRunner for CandleRunner {
    fn run(&self, input: &[f32]) -> Result<Vec<f32>, Error> {
        let t = self.manifest.tile as usize;
        let n = t * t;
        if input.len() != pmrid::PLANES * n {
            return Err(Error::Input(format!("a tile of this model is {} values, not {}", pmrid::PLANES * n, input.len())));
        }
        let (s, o) = self.mapping(input);
        let scale = self.input_scale;
        let clip = |v: f32| match self.architecture {
            // the authors' benchmark clips the mosaic to 0..1 before the transform
            Architecture::Pmrid if v.is_finite() => v.clamp(0.0, 1.0),
            Architecture::Pmrid => 0.0,
        };
        let x: Vec<f32> = input.iter().map(|&v| (s * clip(v) + o) * scale).collect();
        let out = self.network(x, t)?;
        if out.len() != pmrid::PLANES * n || out.iter().any(|v| !v.is_finite()) {
            return Err(Error::Model(format!("{} values, some not finite, instead of {}", out.len(), pmrid::PLANES * n)));
        }
        let planes: Vec<Vec<f32>> = out.chunks_exact(n).map(|p| p.iter().map(|&y| (y / scale - o) / s).collect()).collect();
        Ok(builtin::demosaic(&planes, t))
    }
}

/// How two runners of one network differ on `tile` (four planes): the largest difference relative to the tile's
/// range, the mean difference, and each one's time (ms).
pub fn compare(a: &CandleRunner, b: &CandleRunner, tile: &[f32]) -> Result<(f32, f32, f64, f64), Error> {
    let time = |r: &CandleRunner| -> Result<(Vec<f32>, f64), Error> {
        let started = Instant::now();
        let out = r.run(tile)?;
        Ok((out, started.elapsed().as_secs_f64() * 1e3))
    };
    let (x, ta) = time(a)?;
    let (y, tb) = time(b)?;
    if x.len() != y.len() || x.is_empty() {
        return Err(Error::Model("the two runners gave answers of different sizes".into()));
    }
    let (worst, sum) = x.iter().zip(&y).fold((0f32, 0f64), |(w, s), (p, q)| {
        let d = (p - q).abs();
        (w.max(d), s + f64::from(d))
    });
    Ok((worst, (sum / x.len() as f64) as f32, ta, tb))
}

#[cfg(test)]
mod tests;
