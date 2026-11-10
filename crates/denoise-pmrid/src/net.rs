//! The network itself (see the crate's notes). Modified work under the Apache License 2.0: see the crate root and NOTICE.

pub use candle_core::backprop::GradStore;
pub use candle_core::{Device, Tensor, Var};

use lightcraft_denoise_core::run::Error;
use lightcraft_denoise_core::safetensors::Weights;

type R<T> = candle_core::Result<T>;

/// Planes a tile has, in and out.
pub const PLANES: usize = 4;
/// The tile must be a multiple of this (four stages that halve it).
pub const MULTIPLE: usize = 16;

/// Where the parameters come from: a file's weights, or zeros while listing what the network needs.
pub trait Source {
    /// The values of parameter `name`, which must have `shape`.
    fn get(&mut self, name: &str, shape: &[usize]) -> Result<Vec<f32>, Error>;
}

impl Source for Weights {
    fn get(&mut self, name: &str, shape: &[usize]) -> Result<Vec<f32>, Error> {
        self.take(name, shape)
    }
}

/// Lists the parameters the network asks for, giving each zeros (or what `fill` makes of its index).
pub struct Listing {
    pub names: Vec<(String, Vec<usize>)>,
    fill: Option<Box<dyn FnMut(usize) -> f32>>,
}

impl Listing {
    pub fn zeros() -> Self {
        Listing { names: Vec::new(), fill: None }
    }

    /// Values from `fill(i)` for the `i`-th value handed out (over all parameters).
    pub fn with(fill: impl FnMut(usize) -> f32 + 'static) -> Self {
        Listing { names: Vec::new(), fill: Some(Box::new(fill)) }
    }
}

impl Source for Listing {
    fn get(&mut self, name: &str, shape: &[usize]) -> Result<Vec<f32>, Error> {
        self.names.push((name.to_string(), shape.to_vec()));
        let n = shape.iter().try_fold(1usize, |a, &d| a.checked_mul(d)).ok_or_else(|| Error::Load(format!("`{name}` is too large")))?;
        let start: usize = self.names.iter().rev().skip(1).map(|(_, s)| s.iter().product::<usize>()).sum();
        Ok(match &mut self.fill {
            Some(f) => (start..start + n).map(f).collect(),
            None => vec![0.0; n],
        })
    }
}

fn load(e: candle_core::Error) -> Error {
    Error::Load(e.to_string())
}

/// How the depthwise convolutions are computed (the same numbers either way, to rounding).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Depthwise {
    /// One multiply-add of the shifted picture per tap: little memory, many small steps (the processor).
    Shifted,
    /// The picture unfolded into its taps (one contiguous copy each), then one batched matrix product: few large steps,
    /// a copy of the features per tap.
    Unfold,
}

impl Depthwise {
    /// The way networks are built here: unfolding is faster on both the processor and Metal (see docs/denoise.md).
    pub fn for_device(_dev: &Device) -> Self {
        Depthwise::Unfold
    }
}

struct Build<'a> {
    src: &'a mut dyn Source,
    dev: &'a Device,
    depthwise: Depthwise,
    /// When the network is built to be trained: every parameter as a variable, by name.
    vars: Option<Vec<(String, Var)>>,
}

impl Build<'_> {
    /// Parameter `name` of `shape` on the device (a variable when training). Every form a layer keeps is derived from it
    /// by tensor operations, so gradients reach it.
    fn param(&mut self, name: &str, shape: &[usize]) -> Result<Tensor, Error> {
        let values = self.src.get(name, shape)?;
        match &mut self.vars {
            Some(vars) => {
                let v = Var::from_vec(values, shape, self.dev).map_err(load)?;
                let t = v.as_tensor().clone();
                vars.push((name.to_string(), v));
                Ok(t)
            }
            None => Tensor::from_vec(values, shape, self.dev).map_err(load),
        }
    }

    fn separable(&mut self, p: &str, cin: usize, cout: usize, k: usize) -> Result<Separable, Error> {
        let dw = self.param(&format!("{p}.depthwise.weight"), &[cin, 1, k, k])?;
        let pw = self.param(&format!("{p}.pointwise.weight"), &[cout, cin, 1, 1])?;
        let pb = self.param(&format!("{p}.pointwise.bias"), &[cout])?;
        let shaped = || -> R<(Kernel, Tensor, Tensor)> {
            let per_tap = dw.reshape((cin, k * k))?;
            let depthwise = match self.depthwise {
                Depthwise::Shifted => {
                    Kernel::Shifted((0..k * k).map(|tap| per_tap.narrow(1, tap, 1)?.reshape((1, cin, 1, 1))).collect::<R<Vec<_>>>()?)
                }
                Depthwise::Unfold => Kernel::Unfold(dw.reshape((cin, 1, k * k))?),
            };
            Ok((depthwise, pw.reshape((cout, cin))?, pb.reshape((1, cout, 1, 1))?))
        };
        let (depthwise, pw, pb) = shaped().map_err(load)?;
        Ok(Separable { k, depthwise, pw, pb, cout })
    }

    fn conv(&mut self, p: &str, cin: usize, cout: usize, k: usize) -> Result<Conv, Error> {
        let w = self.param(&format!("{p}.conv.weight"), &[cout, cin, k, k])?;
        let b = self.param(&format!("{p}.conv.bias"), &[cout])?.reshape((1, cout, 1, 1)).map_err(load)?;
        Ok(Conv { w, b, k })
    }

    /// A 2 × 2 transposed convolution of stride 2.
    fn up(&mut self, p: &str, cin: usize, cout: usize) -> Result<Up, Error> {
        let w = self.param(&format!("{p}.weight"), &[cin, cout, 2, 2])?;
        let b = self.param(&format!("{p}.bias"), &[cout])?.reshape((1, cout, 1, 1)).map_err(load)?;
        Ok(Up { w, b, cout })
    }

    fn encoder_stage(&mut self, p: &str, cin: usize, cout: usize, blocks: usize) -> Result<Vec<EncoderBlock>, Error> {
        let mut out = Vec::with_capacity(blocks);
        for i in 0..blocks {
            let (inc, stride) = if i == 0 { (cin, 2) } else { (cout, 1) };
            let mid = cout / 4;
            let q = format!("{p}.{i}");
            let conv1 = self.separable(&format!("{q}.conv1"), inc, mid, 5)?;
            let conv2 = self.separable(&format!("{q}.conv2"), mid, cout, 5)?;
            let proj = if stride == 1 && inc == cout { None } else { Some(self.separable(&format!("{q}.proj"), inc, cout, 3)?) };
            out.push(EncoderBlock { conv1, conv2, proj, stride });
        }
        Ok(out)
    }

    fn decoder_block(&mut self, p: &str, c: usize) -> Result<DecoderBlock, Error> {
        Ok(DecoderBlock { conv0: self.separable(&format!("{p}.conv0"), c, c, 3)?, conv1: self.separable(&format!("{p}.conv1"), c, c, 3)? })
    }

    fn decoder_stage(&mut self, p: &str, cin: usize, skip: usize, cout: usize) -> Result<DecoderStage, Error> {
        Ok(DecoderStage {
            decode: self.decoder_block(&format!("{p}.decode_conv"), cin)?,
            up: self.up(&format!("{p}.upsample"), cin, cout)?,
            proj: self.separable(&format!("{p}.proj_conv"), skip, cout, 3)?,
        })
    }
}

enum Kernel {
    /// Per tap, the channels' weights `[1, C, 1, 1]`.
    Shifted(Vec<Tensor>),
    /// The channels' weights `[C, 1, k²]`.
    Unfold(Tensor),
}

/// The depthwise convolution unfolded. Each channel, padded (one row more below) and laid out as one line of
/// `(h + k) · w'` samples (`w'` the padded width), holds every tap's shifted picture as one contiguous run of `h · w'`
/// samples starting at `i · w' + j`: the runs are copied side by side (`[C, k², h · w']`), one batched product with each
/// channel's weights sums them, and the padding columns are dropped. A stride of 2 takes every other place after.
fn unfolded(x: &Tensor, w: &Tensor, k: usize, stride: usize) -> R<Tensor> {
    let (n, c, h, wd) = x.dims4()?;
    let p = k / 2;
    let wp = wd + 2 * p;
    let flat = x.pad_with_zeros(2, p, p + 1)?.pad_with_zeros(3, p, p)?.reshape((n, c, (h + 2 * p + 1) * wp))?;
    // `cat` of views along a middle dimension gives a transposed layout, which candle's Metal product refuses, and the
    // processor's too for a batch: there each run is copied out first, so that `cat` lays them side by side plainly (the
    // processor's product takes one tile's as it is)
    let copy = !x.device().is_cpu() || n > 1;
    let mut runs = Vec::with_capacity(k * k);
    for i in 0..k {
        for j in 0..k {
            let run = flat.narrow(2, i * wp + j, h * wp)?;
            runs.push(if copy { run.contiguous()? } else { run }.unsqueeze(2)?);
        }
    }
    let cols = Tensor::cat(&runs, 2)?;
    // one product per picture of the batch: the weights laid out for each (small), as the Metal product wants
    let w = if n == 1 { w.unsqueeze(0)? } else { w.unsqueeze(0)?.broadcast_as((n, c, 1, k * k))?.contiguous()? };
    let y = w.matmul(&cols)?.reshape((n, c, h, wp))?.narrow(3, 0, wd)?.contiguous()?;
    if stride == 2 { every_other(&y) } else { Ok(y) }
}

/// `w` (`[rows, cols]`) times each picture of `x` (`[n, cols, l]`): `[n, rows, l]`.
fn batched(w: &Tensor, x: &Tensor) -> R<Tensor> {
    let (n, _, _) = x.dims3()?;
    if n == 1 {
        return w.matmul(&x.squeeze(0)?)?.unsqueeze(0);
    }
    let (rows, cols) = w.dims2()?;
    w.unsqueeze(0)?.broadcast_as((n, rows, cols))?.contiguous()?.matmul(x)
}

/// One tap of a depthwise kernel per channel, multiplied into the shifted picture and summed.
fn depthwise(x: &Tensor, taps: &[Tensor], k: usize) -> R<Tensor> {
    let (_, _, h, w) = x.dims4()?;
    let p = k / 2;
    let padded = x.pad_with_zeros(2, p, p)?.pad_with_zeros(3, p, p)?;
    let mut acc: Option<Tensor> = None;
    for (tap, weight) in taps.iter().enumerate() {
        let (i, j) = (tap / k, tap % k);
        let term = padded.narrow(2, i, h)?.narrow(3, j, w)?.broadcast_mul(weight)?;
        acc = Some(match acc {
            Some(a) => (a + term)?,
            None => term,
        });
    }
    acc.ok_or_else(|| candle_core::Error::Msg("a depthwise kernel with no taps".into()))
}

/// Every other row and column, from the first: a stride-2 convolution is its stride-1 sum taken there.
fn every_other(x: &Tensor) -> R<Tensor> {
    let (b, c, h, w) = x.dims4()?;
    x.reshape((b, c, h / 2, 2, w / 2, 2))?.narrow(3, 0, 1)?.narrow(5, 0, 1)?.reshape((b, c, h / 2, w / 2))
}

/// A 1 × 1 convolution: `w` is `[cout, cin]`.
fn pointwise(x: &Tensor, w: &Tensor, b: &Tensor, cout: usize) -> R<Tensor> {
    let (n, cin, h, wd) = x.dims4()?;
    batched(w, &x.reshape((n, cin, h * wd))?)?.reshape((n, cout, h, wd))?.broadcast_add(b)
}

struct Separable {
    k: usize,
    depthwise: Kernel,
    pw: Tensor,
    pb: Tensor,
    cout: usize,
}

impl Separable {
    fn forward(&self, x: &Tensor, stride: usize, relu: bool) -> R<Tensor> {
        let y = match &self.depthwise {
            Kernel::Shifted(taps) => {
                let y = depthwise(x, taps, self.k)?;
                if stride == 2 { every_other(&y)? } else { y }
            }
            Kernel::Unfold(w) => unfolded(x, w, self.k, stride)?,
        };
        let y = pointwise(&y, &self.pw, &self.pb, self.cout)?;
        if relu { y.relu() } else { Ok(y) }
    }
}

struct Conv {
    w: Tensor,
    b: Tensor,
    k: usize,
}

impl Conv {
    fn forward(&self, x: &Tensor, relu: bool) -> R<Tensor> {
        let y = x.conv2d(&self.w, self.k / 2, 1, 1, 1)?.broadcast_add(&self.b)?;
        if relu { y.relu() } else { Ok(y) }
    }
}

struct Up {
    /// `[cin, cout, 2, 2]` as published. It is rearranged into a matrix on every pass, not once when built: a rearranged
    /// copy would not follow a variable the optimizer updates, so training would run on stale weights and save others.
    w: Tensor,
    b: Tensor,
    cout: usize,
}

impl Up {
    fn forward(&self, x: &Tensor) -> R<Tensor> {
        let (n, cin, h, w) = x.dims4()?;
        // row `(co · 2 + dy) · 2 + dx` is what input channel `ci` adds to output pixel `(2y + dy, 2x + dx)` of channel `co`
        let wt = self.w.permute((1, 2, 3, 0))?.reshape((self.cout * 4, cin))?;
        let m = batched(&wt, &x.reshape((n, cin, h * w))?)?;
        m.reshape((n, self.cout, 2, 2, h, w))?.permute((0, 1, 4, 2, 5, 3))?.reshape((n, self.cout, 2 * h, 2 * w))?.broadcast_add(&self.b)
    }
}

struct EncoderBlock {
    conv1: Separable,
    conv2: Separable,
    proj: Option<Separable>,
    stride: usize,
}

impl EncoderBlock {
    fn forward(&self, x: &Tensor) -> R<Tensor> {
        let proj = match &self.proj {
            Some(p) => p.forward(x, self.stride, false)?,
            None => x.clone(),
        };
        let y = self.conv2.forward(&self.conv1.forward(x, self.stride, true)?, 1, false)?;
        (y + proj)?.relu()
    }
}

struct DecoderBlock {
    conv0: Separable,
    conv1: Separable,
}

impl DecoderBlock {
    fn forward(&self, x: &Tensor) -> R<Tensor> {
        self.conv1.forward(&self.conv0.forward(x, 1, true)?, 1, false)? + x
    }
}

struct DecoderStage {
    decode: DecoderBlock,
    up: Up,
    proj: Separable,
}

impl DecoderStage {
    fn forward(&self, x: &Tensor, skip: &Tensor) -> R<Tensor> {
        self.up.forward(&self.decode.forward(x)?)? + self.proj.forward(skip, 1, true)?
    }
}

/// The network, its parameters on one device.
pub struct Pmrid {
    conv0: Conv,
    enc: [Vec<EncoderBlock>; 4],
    encdec: Separable,
    dec: [DecoderStage; 4],
    out0: DecoderBlock,
    out1: Conv,
}

impl Pmrid {
    /// Build the network on `dev` from `src`, which must hold every parameter with the authors' name and shape.
    pub fn new(src: &mut dyn Source, dev: &Device) -> Result<Self, Error> {
        Self::with(src, dev, Depthwise::for_device(dev))
    }

    /// [`Pmrid::new`] with the depthwise convolutions computed as `depthwise` says.
    pub fn with(src: &mut dyn Source, dev: &Device, depthwise: Depthwise) -> Result<Self, Error> {
        Self::build(&mut Build { src, dev, depthwise, vars: None })
    }

    /// The network built to be trained: every parameter is a variable (named as in [`Pmrid::parameters`]) that the
    /// forward pass's gradients reach.
    pub fn trainable(src: &mut dyn Source, dev: &Device) -> Result<(Self, Vec<(String, Var)>), Error> {
        let mut b = Build { src, dev, depthwise: Depthwise::Unfold, vars: Some(Vec::new()) };
        let net = Self::build(&mut b)?;
        Ok((net, b.vars.unwrap_or_default()))
    }

    /// The variables of a trained network as weights (to save as safetensors).
    pub fn weights_of(vars: &[(String, Var)]) -> Result<Weights, Error> {
        let mut w = Weights::default();
        for (name, v) in vars {
            let values = v.as_tensor().flatten_all().and_then(|t| t.to_vec1::<f32>()).map_err(load)?;
            w.insert(name, v.as_tensor().dims(), values)?;
        }
        Ok(w)
    }

    fn build(b: &mut Build<'_>) -> Result<Self, Error> {
        Ok(Pmrid {
            conv0: b.conv("conv0", PLANES, 16, 3)?,
            enc: [
                b.encoder_stage("enc1", 16, 64, 2)?,
                b.encoder_stage("enc2", 64, 128, 2)?,
                b.encoder_stage("enc3", 128, 256, 4)?,
                b.encoder_stage("enc4", 256, 512, 4)?,
            ],
            encdec: b.separable("encdec", 512, 64, 3)?,
            dec: [
                b.decoder_stage("dec1", 64, 256, 64)?,
                b.decoder_stage("dec2", 64, 128, 32)?,
                b.decoder_stage("dec3", 32, 64, 32)?,
                b.decoder_stage("dec4", 32, 16, 16)?,
            ],
            out0: b.decoder_block("out0", 16)?,
            out1: b.conv("out1", 16, PLANES, 3)?,
        })
    }

    /// Every parameter the network has: name and shape, in the order they are read.
    pub fn parameters() -> Result<Vec<(String, Vec<usize>)>, Error> {
        let mut list = Listing::zeros();
        Pmrid::new(&mut list, &Device::Cpu)?;
        Ok(list.names)
    }

    /// Run tiles: `[n, 4, h, w]` in, the same out, `h` and `w` multiples of [`MULTIPLE`].
    pub fn forward(&self, inp: &Tensor) -> R<Tensor> {
        let (n, c, h, w) = inp.dims4()?;
        if n == 0 || c != PLANES || h % MULTIPLE != 0 || w % MULTIPLE != 0 || h == 0 || w == 0 {
            candle_core::bail!("PMRID takes [n, 4, h, w] with h and w multiples of {MULTIPLE}, not {:?}", inp.dims())
        }
        let stage = |blocks: &[EncoderBlock], x: Tensor| blocks.iter().try_fold(x, |x, b| b.forward(&x));
        let c0 = self.conv0.forward(inp, true)?;
        let [e1, e2, e3, e4] = &self.enc;
        let c1 = stage(e1, c0.clone())?;
        let c2 = stage(e2, c1.clone())?;
        let c3 = stage(e3, c2.clone())?;
        let c4 = stage(e4, c3.clone())?;
        let c5 = self.encdec.forward(&c4, 1, true)?;
        let [d1, d2, d3, d4] = &self.dec;
        let x = d1.forward(&c5, &c3)?;
        let x = d2.forward(&x, &c2)?;
        let x = d3.forward(&x, &c1)?;
        let x = d4.forward(&x, &c0)?;
        let x = self.out1.forward(&self.out0.forward(&x)?, false)?;
        inp + x
    }
}

/// PMRID's parameters set so that the network is a 3 × 3 box blur of each plane (through its first convolution, the last
/// decoder's skip connection and the last convolution; everything else zero): a real network that cleans noise, for
/// tests of the layers that install and run candle networks. Nothing here is a model anyone would use.
#[doc(hidden)]
pub fn blur_weights() -> Result<Weights, Error> {
    let mut w = Weights::default();
    for (name, shape) in Pmrid::parameters()? {
        let n: usize = shape.iter().product();
        let mut v = vec![0f32; n];
        let mut set = |i: usize, x: f32| {
            if let Some(slot) = v.get_mut(i) {
                *slot = x;
            }
        };
        match name.as_str() {
            // the four planes into the first four channels, as they are
            "conv0.conv.weight" => (0..PLANES).for_each(|c| set(((c * PLANES + c) * 3 + 1) * 3 + 1, 1.0)),
            // and through the last skip connection unchanged
            "dec4.proj_conv.depthwise.weight" => (0..PLANES).for_each(|c| set(c * 9 + 4, 1.0)),
            "dec4.proj_conv.pointwise.weight" => (0..PLANES).for_each(|c| set(c * 16 + c, 1.0)),
            // the residual: the 3 × 3 mean minus the sample, so input + residual is the mean
            "out1.conv.weight" => {
                for p in 0..PLANES {
                    for tap in 0..9 {
                        set((p * 16 + p) * 9 + tap, 1.0 / 9.0 - if tap == 4 { 1.0 } else { 0.0 });
                    }
                }
            }
            _ => {}
        }
        w.insert(&name, &shape, v)?;
    }
    Ok(w)
}

/// PMRID's parameters with deterministic small random values in `-scale..scale` (`scale` 0: zeros), for tests of the
/// layers that build and run the network. Nothing here is a model anyone would use.
#[doc(hidden)]
pub fn random_weights(seed: u64, scale: f32) -> Result<Weights, Error> {
    let fill = move |i: usize| {
        let mut x = (i as u64).wrapping_mul(0x9e37_79b9_7f4a_7c15) ^ seed;
        x ^= x >> 31;
        x = x.wrapping_mul(0xbf58_476d_1ce4_e5b9);
        x ^= x >> 29;
        ((x >> 40) as f32 / (1u64 << 24) as f32 * 2.0 - 1.0) * scale
    };
    let mut w = Weights::default();
    let mut i = 0usize;
    for (name, shape) in Pmrid::parameters()? {
        let n: usize = shape.iter().product();
        w.insert(&name, &shape, (i..i + n).map(fill).collect())?;
        i += n;
    }
    Ok(w)
}

#[cfg(test)]
mod tests;
