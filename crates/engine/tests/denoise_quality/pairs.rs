//! Real-noise pairs for the denoise benchmark: tripod shots of one scene at its lowest ISO (the reference) and at
//! higher ISOs, from a folder the user names (`LC_DENOISE_PAIRS`; no photo is in the repository).
//!
//! - **Matching** ([`group`]): a subfolder is one scene; else an `iso<N>` in the file name names the scene (what is
//!   left of the name, e.g. `kitchen_iso100.cr3` and `kitchen_iso12800.cr3`); else the EXIF does (the same camera,
//!   frame size, focal length and aperture, shot less than `gap` apart). The ISO comes from the EXIF, else the name.
//!   Several shots at the lowest ISO are averaged into the reference.
//! - **Registration** ([`align`]): the reference is shifted by whole colour cells (two pixels, so the colour pattern
//!   lines up) to the place where its first green plane correlates best with the noisy shot's (a tripod still moves a
//!   pixel or two between shots), and scaled plane by plane to the noisy shot's mean ([`gains`]: the exposures of a
//!   real pair are never exactly equal).
//! - **Scoring**: each denoiser runs on the noisy shot's window, told the noise measured on its whole frame as the
//!   app does, and is scored like the synthetic benchmark (PSNR on gamma-encoded camera RGB, the texture figure, the
//!   detail ratio). The reference has a little noise of its own, which caps the PSNR a perfect result could reach.

use std::path::{Path, PathBuf};

use lightcraft_denoise_core::bayer::Layout;
use lightcraft_denoise_core::manifest::Gain;
use lightcraft_denoise_core::run::{Control, Params, TileRunner, denoise_bayer};

use super::{crop_rgb8, gradient, plain, psnr, texture};

/// One raw file of the folder.
#[derive(Clone, Debug, PartialEq)]
pub struct Shot {
    pub path: PathBuf,
    /// The subfolder it is in, when it is not at the top.
    pub folder: Option<String>,
    pub iso: u32,
    /// The scene its name gives (an `iso<N>` taken out), if it has one.
    pub named: Option<String>,
    pub camera: String,
    /// Frame size, focal length and f-number (tenths), as the EXIF says.
    pub size: (u32, u32),
    pub lens: (i64, i64),
    /// Capture time, seconds.
    pub time: Option<i64>,
}

/// A scene: its reference shots (all at its lowest ISO) and the noisy shots.
#[derive(Debug)]
pub struct Scene {
    pub name: String,
    pub reference: Vec<Shot>,
    pub noisy: Vec<Shot>,
}

/// The `iso<N>` in a file name (any case, digits after it, optionally after `-`, `_` or a space) and what is left of
/// the name around it (separators trimmed), lowercased: `Kitchen_ISO12800` → `("kitchen", 12800)`.
pub fn iso_token(stem: &str) -> Option<(String, u32)> {
    let lower = stem.to_lowercase();
    let mut from = 0;
    while let Some(at) = lower.get(from..).and_then(|s| s.find("iso")).map(|i| i + from) {
        let after = lower.get(at + 3..).unwrap_or("");
        let gap = after.len() - after.trim_start_matches(['-', '_', ' ']).len();
        let digits: String = after.get(gap..).unwrap_or("").chars().take_while(char::is_ascii_digit).collect();
        if let Ok(iso) = digits.parse::<u32>()
            && iso > 0
        {
            let before = lower.get(..at).unwrap_or("");
            let rest = lower.get(at + 3 + gap + digits.len()..).unwrap_or("");
            let sep = ['-', '_', ' ', '.'];
            let name =
                format!("{}{}", before.trim_end_matches(sep), if rest.trim_matches(sep).is_empty() { "" } else { "_" }) + rest.trim_matches(sep);
            return Some((name.trim_matches(sep).to_string(), iso));
        }
        from = at + 3;
    }
    None
}

/// Seconds since 1970 of an EXIF time (through chrono, as every date in LightCraft).
fn seconds(t: &lightcraft_meta::DateTime) -> Option<i64> {
    let date = chrono::NaiveDate::from_ymd_opt(i32::from(t.year), u32::from(t.month), u32::from(t.day))?;
    let time = date.and_hms_opt(u32::from(t.hour), u32::from(t.minute), u32::from(t.second))?;
    Some(time.and_utc().timestamp() - i64::from(t.offset_minutes.unwrap_or(0)) * 60)
}

/// The raw files of `dir` and of its subfolders (one level), with what [`group`] needs; files that are not raws, or
/// whose ISO is in neither the EXIF nor the name, are left out (and said so on stderr).
pub fn discover(dir: &Path) -> Result<Vec<Shot>, String> {
    let mut files: Vec<(PathBuf, Option<String>)> = Vec::new();
    let entries = std::fs::read_dir(dir).map_err(|e| format!("{}: {e}", dir.display()))?;
    for e in entries.flatten() {
        let p = e.path();
        if p.is_dir() {
            let folder = p.file_name().map(|n| n.to_string_lossy().into_owned());
            for f in std::fs::read_dir(&p).map_err(|e| format!("{}: {e}", p.display()))?.flatten() {
                if f.path().is_file() {
                    files.push((f.path(), folder.clone()));
                }
            }
        } else if p.is_file() {
            files.push((p, None));
        }
    }
    files.sort();
    let mut shots = Vec::new();
    for (path, folder) in files {
        let Ok(bytes) = std::fs::read(&path) else { continue };
        if lightcraft_raw::probe(&bytes).is_none() {
            continue;
        }
        let meta = lightcraft_meta::extract(&bytes);
        let stem = path.file_stem().map(|s| s.to_string_lossy().into_owned()).unwrap_or_default();
        let token = iso_token(&stem);
        let Some(iso) = meta.iso.filter(|&i| i > 0).or(token.as_ref().map(|t| t.1)) else {
            eprintln!("{}: no ISO in its EXIF or its name, left out", path.display());
            continue;
        };
        let tenths = |v: Option<f64>| v.map_or(0, |v| (v * 10.0).round() as i64);
        shots.push(Shot {
            path,
            folder,
            iso,
            named: token.map(|t| t.0).filter(|n| !n.is_empty()),
            camera: format!("{} {}", meta.make.unwrap_or_default(), meta.model.unwrap_or_default()).trim().to_string(),
            size: (meta.width.unwrap_or(0), meta.height.unwrap_or(0)),
            lens: (tenths(meta.focal_length), tenths(meta.f_number)),
            time: meta.capture_time.as_ref().and_then(seconds),
        });
    }
    Ok(shots)
}

/// Sort shots into scenes (see the module's notes): by subfolder, then by the scene in their names, then by EXIF with
/// shots less than `gap` seconds apart together. Scenes with only one ISO are left out.
pub fn group(shots: Vec<Shot>, gap: i64) -> Vec<Scene> {
    let mut groups: Vec<(String, Vec<Shot>)> = Vec::new();
    let add = |name: String, shot: Shot, groups: &mut Vec<(String, Vec<Shot>)>| match groups.iter_mut().find(|(n, _)| *n == name) {
        Some((_, g)) => g.push(shot),
        None => groups.push((name, vec![shot])),
    };
    let mut by_exif: Vec<Shot> = Vec::new();
    for s in shots {
        if let Some(f) = &s.folder {
            add(f.clone(), s.clone(), &mut groups);
        } else if let Some(n) = &s.named {
            add(n.clone(), s.clone(), &mut groups);
        } else {
            by_exif.push(s);
        }
    }
    by_exif.sort_by(|a, b| (&a.camera, a.size, a.lens, a.time, &a.path).cmp(&(&b.camera, b.size, b.lens, b.time, &b.path)));
    let mut run: Vec<Shot> = Vec::new();
    let flush = |run: &mut Vec<Shot>, groups: &mut Vec<(String, Vec<Shot>)>| {
        if let Some(first) = run.first() {
            let stem = first.path.file_stem().map(|s| s.to_string_lossy().into_owned()).unwrap_or_default();
            groups.push((format!("{} {stem}", first.camera).trim().to_string(), std::mem::take(run)));
        }
    };
    for s in by_exif {
        let joins = run.last().is_some_and(|l| {
            l.camera == s.camera && l.size == s.size && l.lens == s.lens && l.time.zip(s.time).is_none_or(|(a, b)| (b - a).abs() <= gap)
        });
        if !joins {
            flush(&mut run, &mut groups);
        }
        run.push(s);
    }
    flush(&mut run, &mut groups);
    groups
        .into_iter()
        .filter_map(|(name, shots)| {
            let base = shots.iter().map(|s| s.iso).min()?;
            let (reference, noisy): (Vec<Shot>, Vec<Shot>) = shots.into_iter().partition(|s| s.iso == base);
            (!noisy.is_empty()).then_some(Scene { name, reference, noisy })
        })
        .collect()
}

/// A decoded raw: its normalised mosaic (black 0, white 1) and colour layout.
pub struct Mosaic {
    pub data: Vec<f32>,
    pub width: usize,
    pub height: usize,
    pub layout: Layout,
}

pub fn load(path: &Path) -> Result<Mosaic, String> {
    let bytes = std::fs::read(path).map_err(|e| format!("{}: {e}", path.display()))?;
    let n = lightcraft_raw::decode(&bytes).and_then(|r| r.normalized()).map_err(|e| format!("{}: {e}", path.display()))?;
    let cfa = n.cfa.as_ref().filter(|_| n.cpp == 1).ok_or_else(|| format!("{}: not a Bayer raw", path.display()))?;
    let layout = Layout::from_cell([cfa.color_at(0, 0), cfa.color_at(1, 0), cfa.color_at(0, 1), cfa.color_at(1, 1)])
        .ok_or_else(|| format!("{}: not a Bayer raw", path.display()))?;
    Ok(Mosaic { data: n.data, width: n.width, height: n.height, layout })
}

/// A window of a mosaic: origin (on a red cell) and size, in pixels.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Window {
    pub x: usize,
    pub y: usize,
    pub w: usize,
    pub h: usize,
}

impl Mosaic {
    /// The window's samples, from `(x + dx, y + dy)`; `None` when it does not fit.
    pub fn cut(&self, win: Window, dx: isize, dy: isize) -> Option<Vec<f32>> {
        let (x0, y0) = (win.x.checked_add_signed(dx)?, win.y.checked_add_signed(dy)?);
        if x0 + win.w > self.width || y0 + win.h > self.height {
            return None;
        }
        Some((0..win.h).flat_map(|y| self.data[(y0 + y) * self.width + x0..(y0 + y) * self.width + x0 + win.w].iter().copied()).collect())
    }

    /// A centred window of at most `w × h` (even, starting on a red cell) that leaves `margin` pixels on every side.
    pub fn centre(&self, w: usize, h: usize, margin: usize) -> Option<Window> {
        let fit = |want: usize, have: usize| want.min(have.saturating_sub(2 * margin + 2)) & !1;
        let (w, h) = (fit(w, self.width), fit(h, self.height));
        if w < 64 || h < 64 {
            return None;
        }
        let x = (((self.width - w) / 2) & !1) + self.layout.dx;
        let y = (((self.height - h) / 2) & !1) + self.layout.dy;
        Some(Window { x, y, w, h })
    }
}

/// The shift (pixels, even) of `reference` within `max` cells that best matches `noisy` over `win`: the highest
/// correlation of the first green planes (every other cell of them, which is plenty).
pub fn align(reference: &Mosaic, noisy: &Mosaic, win: Window, max: usize) -> (isize, isize) {
    let plane = |m: &Mosaic, dx: isize, dy: isize| -> Option<Vec<f64>> {
        let (x0, y0) = (win.x.checked_add_signed(dx)? + 1, win.y.checked_add_signed(dy)?);
        let mut v = Vec::with_capacity(win.w * win.h / 16);
        for cy in (0..win.h / 2).step_by(2) {
            for cx in (0..win.w / 2).step_by(2) {
                v.push(f64::from(*m.data.get((y0 + 2 * cy) * m.width + x0 + 2 * cx)?));
            }
        }
        Some(v)
    };
    let Some(b) = plane(noisy, 0, 0) else { return (0, 0) };
    let mean_b = b.iter().sum::<f64>() / b.len().max(1) as f64;
    let max = max as isize;
    let mut best = (f64::NEG_INFINITY, (0, 0));
    for sy in -max..=max {
        for sx in -max..=max {
            let Some(a) = plane(reference, 2 * sx, 2 * sy) else { continue };
            let mean_a = a.iter().sum::<f64>() / a.len().max(1) as f64;
            let (mut ab, mut aa, mut bb) = (0f64, 0f64, 0f64);
            for (x, y) in a.iter().zip(&b) {
                let (p, q) = (x - mean_a, y - mean_b);
                ab += p * q;
                aa += p * p;
                bb += q * q;
            }
            let c = ab / (aa * bb).sqrt().max(1e-30);
            if c > best.0 {
                best = (c, (2 * sx, 2 * sy));
            }
        }
    }
    best.1
}

/// What each of the four colour planes of `reference` (RGGB, `w` wide) is multiplied by to have the mean of `noisy`'s
/// (clipped samples left out).
pub fn gains(reference: &[f32], noisy: &[f32], w: usize) -> [f32; 4] {
    let mut sums = [(0f64, 0f64); 4];
    for (i, (&r, &n)) in reference.iter().zip(noisy).enumerate() {
        if r < 0.95 && n < 0.95 {
            let p = (i % w) % 2 + 2 * ((i / w) % 2);
            sums[p].0 += f64::from(r);
            sums[p].1 += f64::from(n);
        }
    }
    sums.map(|(r, n)| if r > 1e-6 { (n / r) as f32 } else { 1.0 })
}

/// A denoiser to score: its tile, and its runner for a photo whose whole-frame noise is `noise`.
pub struct Denoiser {
    pub name: String,
    pub tile: usize,
    pub overlap: usize,
    pub gain: Gain,
    #[allow(clippy::type_complexity)]
    pub runner: Box<dyn Fn(Option<(f32, f32)>) -> Box<dyn TileRunner>>,
}

/// How to score a folder.
pub struct Opts {
    /// The window scored, at most (pixels).
    pub window: (usize, usize),
    /// Largest registration shift tried, in colour cells.
    pub max_shift: usize,
    /// Shots further apart than this (seconds) are not one scene, when matched by EXIF.
    pub gap: i64,
    /// Where the side-by-side PNG crops go (reference | noisy | each denoiser).
    pub out: Option<PathBuf>,
}

/// One denoiser on one pair.
#[derive(Clone, Debug)]
pub struct Row {
    pub scene: String,
    pub iso: u32,
    pub reference_iso: u32,
    pub denoiser: String,
    pub noisy_db: f64,
    pub db: f64,
    pub texture: f64,
    pub detail: f64,
    pub secs: f64,
    /// The registration shift found (pixels) for the first reference shot, and the noise measured on the noisy frame.
    pub shift: (isize, isize),
    pub noise: Option<(f32, f32)>,
}

/// Score every pair of `dir` with every denoiser.
pub fn score_folder(dir: &Path, denoisers: &[Denoiser], opts: &Opts) -> Result<Vec<Row>, String> {
    let scenes = group(discover(dir)?, opts.gap);
    if scenes.is_empty() {
        return Err(format!("{}: no scene with shots at two ISOs (see the matching rules in pairs.rs)", dir.display()));
    }
    let mut rows = Vec::new();
    for scene in &scenes {
        let refs: Vec<Mosaic> = scene.reference.iter().map(|s| load(&s.path)).collect::<Result<_, _>>()?;
        for shot in &scene.noisy {
            rows.extend(score_pair(scene, &refs, shot, denoisers, opts)?);
        }
    }
    Ok(rows)
}

fn score_pair(scene: &Scene, refs: &[Mosaic], shot: &Shot, denoisers: &[Denoiser], opts: &Opts) -> Result<Vec<Row>, String> {
    let noisy = load(&shot.path)?;
    let first = refs.first().ok_or("a scene without a reference")?;
    if (first.width, first.height, first.layout) != (noisy.width, noisy.height, noisy.layout) {
        return Err(format!("{}: not the size or colour layout of its reference", shot.path.display()));
    }
    let win = noisy.centre(opts.window.0, opts.window.1, 2 * opts.max_shift + 2).ok_or("the frame is too small to score")?;
    let noisy_win = noisy.cut(win, 0, 0).ok_or("the window does not fit")?;
    // the reference: each base-ISO shot registered to this one, matched in exposure, averaged
    let mut truth = vec![0f32; win.w * win.h];
    let mut first_shift = (0, 0);
    for (k, r) in refs.iter().enumerate() {
        let (dx, dy) = align(r, &noisy, win, opts.max_shift);
        if k == 0 {
            first_shift = (dx, dy);
        }
        let cut = r.cut(win, dx, dy).ok_or("the registered reference does not fit")?;
        let g = gains(&cut, &noisy_win, win.w);
        for (i, (t, v)) in truth.iter_mut().zip(&cut).enumerate() {
            *t += v * g[(i % win.w) % 2 + 2 * ((i / win.w) % 2)] / refs.len() as f32;
        }
    }
    // the noise of the whole noisy frame, as the app measures it
    let noise = lightcraft_denoise_core::builtin::measure_noise(&noisy.data, noisy.width, noisy.height);
    let mean = truth.iter().map(|v| f64::from(*v)).sum::<f64>() / truth.len() as f64;
    let scale = (0.18 / mean.max(1e-6)) as f32;
    let (truth_rgb, noisy_rgb) = (plain(&truth, win.w, win.h), plain(&noisy_win, win.w, win.h));
    let noisy_db = psnr(&noisy_rgb, &truth_rgb, win.w, win.h, scale);
    let truth_grad = gradient(&truth_rgb, win.w, win.h, scale);
    let mut rows = Vec::new();
    let mut crops = Vec::new();
    let side = 400.min(win.w).min(win.h);
    let at = (win.w / 2 - side / 2, win.h / 2 - side / 2);
    crops.push(crop_rgb8(&truth_rgb, win.w, at, side, scale));
    crops.push(crop_rgb8(&noisy_rgb, win.w, at, side, scale));
    for d in denoisers {
        let runner = (d.runner)(noise);
        let params = Params { tile: d.tile, overlap: d.overlap, gain: d.gain, clip: Some(0.99), parallel: 4 };
        let started = std::time::Instant::now();
        let out = denoise_bayer(&noisy_win, win.w, win.h, Layout { dx: 0, dy: 0 }, runner.as_ref(), &params, &Control::default())
            .map_err(|e| format!("{}: {e}", d.name))?;
        let secs = started.elapsed().as_secs_f64();
        crops.push(crop_rgb8(&out.data, win.w, at, side, scale));
        rows.push(Row {
            scene: scene.name.clone(),
            iso: shot.iso,
            reference_iso: scene.reference.first().map_or(0, |s| s.iso),
            denoiser: d.name.clone(),
            noisy_db,
            db: psnr(&out.data, &truth_rgb, win.w, win.h, scale),
            texture: texture(&out.data, &truth_rgb, win.w, win.h, scale),
            detail: gradient(&out.data, win.w, win.h, scale) / truth_grad,
            secs,
            shift: first_shift,
            noise,
        });
    }
    if let Some(dir) = &opts.out {
        let mut sheet = Vec::with_capacity(crops.len() * side * side * 3);
        for y in 0..side {
            for c in &crops {
                sheet.extend_from_slice(&c[y * side * 3..(y + 1) * side * 3]);
            }
        }
        let img =
            lightcraft_codecs::encode::EncodeImage::new((crops.len() * side) as u32, side as u32, 3, lightcraft_codecs::encode::Samples::U8(&sheet));
        let png = lightcraft_codecs::encode::encode_png(&img, &lightcraft_codecs::encode::EncodeMeta::default()).map_err(|e| e.to_string())?;
        let safe: String = scene.name.chars().map(|c| if c.is_ascii_alphanumeric() || c == '-' { c } else { '_' }).collect();
        std::fs::create_dir_all(dir).map_err(|e| e.to_string())?;
        std::fs::write(dir.join(format!("{safe}-iso{}.png", shot.iso)), png).map_err(|e| e.to_string())?;
    }
    Ok(rows)
}

/// The rows as a table, then the mean of each denoiser at each ISO.
pub fn report(rows: &[Row]) -> String {
    let mut s = format!(
        "{:<28}{:>8}{:>18}{:>10}{:>10}{:>8}{:>9}{:>8}{:>8}{:>10}\n",
        "scene", "ISO", "denoiser", "noisy dB", "dB", "gain", "texture", "detail", "secs", "shift"
    );
    for r in rows {
        s += &format!(
            "{:<28}{:>8}{:>18}{:>10.2}{:>10.2}{:>+8.2}{:>9.3}{:>8.3}{:>8.2}{:>10}\n",
            r.scene.chars().take(27).collect::<String>(),
            r.iso,
            r.denoiser,
            r.noisy_db,
            r.db,
            r.db - r.noisy_db,
            r.texture,
            r.detail,
            r.secs,
            format!("{},{}", r.shift.0, r.shift.1)
        );
    }
    let mut keys: Vec<(String, u32)> = rows.iter().map(|r| (r.denoiser.clone(), r.iso)).collect();
    keys.sort();
    keys.dedup();
    s += &format!("\n{:<18}{:>8}{:>6}{:>10}{:>10}{:>8}{:>9}\n", "denoiser", "ISO", "pairs", "noisy dB", "dB", "gain", "texture");
    for (d, iso) in keys {
        let of: Vec<&Row> = rows.iter().filter(|r| r.denoiser == d && r.iso == iso).collect();
        let n = of.len() as f64;
        let mean = |f: &dyn Fn(&Row) -> f64| of.iter().map(|r| f(r)).sum::<f64>() / n;
        s += &format!(
            "{:<18}{:>8}{:>6}{:>10.2}{:>10.2}{:>+8.2}{:>9.3}\n",
            d,
            iso,
            of.len(),
            mean(&|r| r.noisy_db),
            mean(&|r| r.db),
            mean(&|r| r.db - r.noisy_db),
            mean(&|r| r.texture)
        );
    }
    s
}
