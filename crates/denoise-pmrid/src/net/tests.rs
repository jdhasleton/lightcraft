use super::*;

/// A deterministic tile of four planes: a ramp with ripples, from 0 to about 1.
fn tile(t: usize) -> Vec<f32> {
    (0..4 * t * t)
        .map(|i| {
            let (x, y, p) = ((i % (t * t)) % t, (i % (t * t)) / t, i / (t * t));
            ((x + y) as f32 / (2 * t) as f32 + 0.1 * ((x * 7 + y * 3 + p * 5) as f32).sin()).clamp(0.0, 1.0)
        })
        .collect()
}

// ---------------------------------------------------------------------------------------------------------------------
// a scalar reference of the network, straight from the definitions of PyTorch's layers (no shifted sums, no
// matrices), to check the candle version against

#[derive(Clone)]
struct Map {
    c: usize,
    h: usize,
    w: usize,
    v: Vec<f32>,
}

impl Map {
    fn at(&self, c: usize, y: isize, x: isize) -> f32 {
        if y < 0 || x < 0 || y as usize >= self.h || x as usize >= self.w {
            return 0.0;
        }
        self.v[(c * self.h + y as usize) * self.w + x as usize]
    }
}

/// `nn.Conv2d(cin, cout, k, stride, padding = k / 2, groups)`, weights `[cout, cin / groups, k, k]`.
fn conv(x: &Map, w: &[f32], b: Option<&[f32]>, cout: usize, k: usize, stride: usize, groups: usize) -> Map {
    let (h, wd) = (x.h / stride, x.w / stride);
    let per = x.c / groups;
    let p = (k / 2) as isize;
    let mut v = vec![0f32; cout * h * wd];
    for co in 0..cout {
        let g = co / (cout / groups);
        for y in 0..h {
            for xx in 0..wd {
                let mut s = b.map_or(0.0, |b| b[co]);
                for ci in 0..per {
                    for i in 0..k {
                        for j in 0..k {
                            let wv = w[((co * per + ci) * k + i) * k + j];
                            s += wv * x.at(g * per + ci, (y * stride + i) as isize - p, (xx * stride + j) as isize - p);
                        }
                    }
                }
                v[(co * h + y) * wd + xx] = s;
            }
        }
    }
    Map { c: cout, h, w: wd, v }
}

/// `nn.ConvTranspose2d(cin, cout, 2, stride = 2)`, weights `[cin, cout, 2, 2]`.
fn up(x: &Map, w: &[f32], b: &[f32], cout: usize) -> Map {
    let (h, wd) = (2 * x.h, 2 * x.w);
    let mut v = vec![0f32; cout * h * wd];
    for co in 0..cout {
        for y in 0..h {
            for xx in 0..wd {
                let mut s = b[co];
                for ci in 0..x.c {
                    s += w[((ci * cout + co) * 2 + y % 2) * 2 + xx % 2] * x.at(ci, (y / 2) as isize, (xx / 2) as isize);
                }
                v[(co * h + y) * wd + xx] = s;
            }
        }
    }
    Map { c: cout, h, w: wd, v }
}

fn relu(mut m: Map) -> Map {
    m.v.iter_mut().for_each(|v| *v = v.max(0.0));
    m
}

fn add(mut a: Map, b: &Map) -> Map {
    a.v.iter_mut().zip(&b.v).for_each(|(x, y)| *x += y);
    a
}

struct Ref<'a>(&'a Weights);

impl Ref<'_> {
    fn t(&self, name: &str) -> Vec<f32> {
        self.0.get(name).unwrap_or_else(|| panic!("{name}")).values.clone()
    }

    fn sep(&self, p: &str, x: &Map, cout: usize, k: usize, stride: usize, has_relu: bool) -> Map {
        let d = conv(x, &self.t(&format!("{p}.depthwise.weight")), None, x.c, k, stride, x.c);
        let y = conv(&d, &self.t(&format!("{p}.pointwise.weight")), Some(&self.t(&format!("{p}.pointwise.bias"))), cout, 1, 1, 1);
        if has_relu { relu(y) } else { y }
    }

    fn stage(&self, p: &str, mut x: Map, cout: usize, blocks: usize) -> Map {
        for i in 0..blocks {
            let stride = if i == 0 { 2 } else { 1 };
            let q = format!("{p}.{i}");
            let proj = if i == 0 { self.sep(&format!("{q}.proj"), &x, cout, 3, 2, false) } else { x.clone() };
            let y = self.sep(&format!("{q}.conv1"), &x, cout / 4, 5, stride, true);
            let y = self.sep(&format!("{q}.conv2"), &y, cout, 5, 1, false);
            x = relu(add(y, &proj));
        }
        x
    }

    fn block(&self, p: &str, x: &Map) -> Map {
        let y = self.sep(&format!("{p}.conv0"), x, x.c, 3, 1, true);
        add(self.sep(&format!("{p}.conv1"), &y, x.c, 3, 1, false), x)
    }

    fn dec(&self, p: &str, x: &Map, skip: &Map, cout: usize) -> Map {
        let y = self.block(&format!("{p}.decode_conv"), x);
        let y = up(&y, &self.t(&format!("{p}.upsample.weight")), &self.t(&format!("{p}.upsample.bias")), cout);
        add(y, &self.sep(&format!("{p}.proj_conv"), skip, cout, 3, 1, true))
    }

    fn forward(&self, inp: &Map) -> Map {
        let c0 = relu(conv(inp, &self.t("conv0.conv.weight"), Some(&self.t("conv0.conv.bias")), 16, 3, 1, 1));
        let c1 = self.stage("enc1", c0.clone(), 64, 2);
        let c2 = self.stage("enc2", c1.clone(), 128, 2);
        let c3 = self.stage("enc3", c2.clone(), 256, 4);
        let c4 = self.stage("enc4", c3.clone(), 512, 4);
        let c5 = self.sep("encdec", &c4, 64, 3, 1, true);
        let x = self.dec("dec1", &c5, &c3, 64);
        let x = self.dec("dec2", &x, &c2, 32);
        let x = self.dec("dec3", &x, &c1, 32);
        let x = self.dec("dec4", &x, &c0, 16);
        let x = self.block("out0", &x);
        let x = conv(&x, &self.t("out1.conv.weight"), Some(&self.t("out1.conv.bias")), 4, 3, 1, 1);
        add(x, inp)
    }
}

#[test]
fn the_network_matches_a_scalar_reference_of_its_layers() {
    let t = 32;
    let w = random_weights(7, 0.2).unwrap();
    let input = tile(t);
    let reference = Ref(&w).forward(&Map { c: 4, h: t, w: t, v: input.clone() });
    for way in [Depthwise::Shifted, Depthwise::Unfold] {
        let mut weights = Weights::parse(&w.to_bytes()).unwrap();
        let net = Pmrid::with(&mut weights, &Device::Cpu, way).unwrap();
        assert!(weights.left().is_empty());
        let x = Tensor::from_vec(input.clone(), (1, 4, t, t), &Device::Cpu).unwrap();
        let out = net.forward(&x).unwrap().flatten_all().unwrap().to_vec1::<f32>().unwrap();
        assert_eq!(out.len(), reference.v.len());
        let range = reference.v.iter().fold(0f32, |m, v| m.max(v.abs()));
        let worst = out.iter().zip(&reference.v).fold(0f32, |m, (a, b)| m.max((a - b).abs()));
        assert!(range > 0.1, "random weights give a non-trivial answer ({range})");
        assert!(worst <= 1e-4 * range.max(1.0), "{way:?}: worst difference {worst} of {range}");
    }
}

#[test]
fn the_parameters_are_the_published_ones() {
    let p = Pmrid::parameters().unwrap();
    let count: usize = p.iter().map(|(_, s)| s.iter().product::<usize>()).sum();
    // the authors' torch_pretrained.ckp holds these 141 tensors, 1,031,908 numbers (the converter in
    // crates/denoise/examples/pmrid-convert.rs checks every name and shape against it)
    assert_eq!((p.len(), count), (141, 1_031_908));
    for name in ["conv0.conv.weight", "enc1.0.proj.depthwise.weight", "enc4.3.conv2.pointwise.bias", "dec4.upsample.weight", "out1.conv.bias"] {
        assert!(p.iter().any(|(n, _)| n == name), "{name}");
    }
    assert!(!p.iter().any(|(n, _)| n == "enc1.1.proj.depthwise.weight"), "a block that keeps its size has no projection");
}

#[test]
fn a_tile_the_network_cannot_take_is_an_error() {
    let net = Pmrid::new(&mut random_weights(1, 0.1).unwrap(), &Device::Cpu).unwrap();
    for dims in [(1, 4, 24, 24), (1, 3, 32, 32), (0, 4, 32, 32)] {
        let x = Tensor::zeros(dims, candle_core::DType::F32, &Device::Cpu).unwrap();
        assert!(net.forward(&x).is_err(), "{dims:?}");
    }
}

#[test]
fn a_batch_gives_what_its_tiles_give_one_at_a_time() {
    let net = Pmrid::new(&mut random_weights(4, 0.1).unwrap(), &Device::Cpu).unwrap();
    let (a, b) = (tile(32), tile(32).iter().map(|v| 1.0 - v).collect::<Vec<f32>>());
    let one = |x: &[f32]| {
        net.forward(&Tensor::from_vec(x.to_vec(), (1, 4, 32, 32), &Device::Cpu).unwrap()).unwrap().flatten_all().unwrap().to_vec1::<f32>().unwrap()
    };
    let both = net.forward(&Tensor::from_vec([a.clone(), b.clone()].concat(), (2, 4, 32, 32), &Device::Cpu).unwrap()).unwrap();
    let both = both.flatten_all().unwrap().to_vec1::<f32>().unwrap();
    let want = [one(&a), one(&b)].concat();
    let worst = both.iter().zip(&want).fold(0f32, |m, (x, y)| m.max((x - y).abs()));
    assert!(worst < 1e-4, "{worst}");
}

#[test]
fn a_trained_network_runs_on_the_weights_it_saves() {
    // The optimizer changes each variable in place. Every layer must follow it, or training runs on stale weights and
    // saves others (the transposed convolutions once kept a rearranged copy made when the network was built).
    let (net, vars) = Pmrid::trainable(&mut random_weights(9, 0.05).unwrap(), &Device::Cpu).unwrap();
    for (i, (_, v)) in vars.iter().enumerate() {
        v.set(&(v.as_tensor() * (1.0 + 0.5 * (i % 3) as f64)).unwrap()).unwrap();
    }
    let x = Tensor::from_vec(tile(32), (1, 4, 32, 32), &Device::Cpu).unwrap();
    let mut saved = lightcraft_denoise_core::safetensors::Weights::parse(&Pmrid::weights_of(&vars).unwrap().to_bytes()).unwrap();
    let reloaded = Pmrid::new(&mut saved, &Device::Cpu).unwrap();
    let (a, b) = (net.forward(&x).unwrap(), reloaded.forward(&x).unwrap());
    let scale = a.abs().unwrap().max_all().unwrap().to_scalar::<f32>().unwrap();
    let diff = (a - b).unwrap().abs().unwrap().max_all().unwrap().to_scalar::<f32>().unwrap();
    assert!(diff <= 1e-5 * scale.max(1.0), "trained and reloaded networks differ by {diff} (answers up to {scale})");
}

#[test]
fn a_trainable_network_learns_and_saves_its_weights() {
    let (net, vars) = Pmrid::trainable(&mut random_weights(5, 0.05).unwrap(), &Device::Cpu).unwrap();
    assert_eq!(vars.len(), 141);
    let x = Tensor::from_vec([tile(32), tile(32)].concat(), (2, 4, 32, 32), &Device::Cpu).unwrap();
    // the target: the input, so the network must learn to predict no noise
    let loss = |net: &Pmrid| (net.forward(&x).unwrap() - &x).unwrap().sqr().unwrap().mean_all().unwrap();
    let before = loss(&net);
    let grads = before.backward().unwrap();
    let mut reached = 0;
    for (_, v) in &vars {
        if let Some(g) = grads.get(v) {
            let g = g.flatten_all().unwrap().to_vec1::<f32>().unwrap();
            assert!(g.iter().all(|v| v.is_finite()));
            reached += 1;
            v.set(&(v.as_tensor() - (grads.get(v).unwrap() * 0.05).unwrap()).unwrap()).unwrap();
        }
    }
    assert_eq!(reached, vars.len(), "every parameter gets a gradient");
    let after = loss(&net);
    let (b, a) = (before.to_scalar::<f32>().unwrap(), after.to_scalar::<f32>().unwrap());
    assert!(a < b, "one step down the gradient lowers the loss: {b} -> {a}");
    // the trained values come back as weights the runner loads
    let w = Pmrid::weights_of(&vars).unwrap();
    let mut again = lightcraft_denoise_core::safetensors::Weights::parse(&w.to_bytes()).unwrap();
    let reloaded = Pmrid::new(&mut again, &Device::Cpu).unwrap();
    let diff = (reloaded.forward(&x).unwrap() - net.forward(&x).unwrap()).unwrap().abs().unwrap().max_all().unwrap().to_scalar::<f32>().unwrap();
    assert!(diff < 1e-5, "{diff}");
}
