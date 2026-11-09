//! A denoise model's manifest: what the model takes and gives, how big its tiles are and who may use it, as
//! plain data.
//!
//! It comes from a `denoise-model.json` next to the model file (`.onnx`, or `.safetensors` for a network LightCraft has
//! written itself, see [`Network`]) or from the user's `catalog.json`. Both are
//! untrusted: [`validate`] must pass before a manifest is used, and [`parse`] applies it after reading JSON of
//! bounded size. The licence, id and address rules are the face models' (`crate::licence`).

use crate::licence::{Licence, valid_id};
use serde::{Deserialize, Serialize};

/// Largest `denoise-model.json` we read.
pub const MAX_MANIFEST_BYTES: usize = 64 * 1024;
/// Largest model file we accept (bytes).
pub const MAX_MODEL_BYTES: u64 = 2 * 1024 * 1024 * 1024;

/// What the model takes and what it gives.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub enum Domain {
    /// The sensor's Bayer mosaic (black subtracted, white at 1, not white balanced) packed as four planes
    /// `[R, G1, G2, B]` of its 2 × 2 cells, forced to RGGB; gives camera RGB at the mosaic's resolution, so
    /// the picture is denoised and demosaiced in one step and must not be demosaiced again.
    BayerToRgb,
}

/// How the model's output scale is brought back to the input's.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "camelCase")]
pub enum Gain {
    /// The output already has the input's scale.
    None,
    /// The output has a scale of its own, `nominal` times the input's (a million, for one model) and a little
    /// different from tile to tile: each tile is scaled so that its channel means equal the input's. A tile too
    /// dark to measure, or whose measured scale is further than `maxDeviation` (a fraction, 0.05 = 5 %) from
    /// `nominal`, is scaled as if it were within that limit.
    MatchMean { nominal: f32, max_deviation: f32 },
}

/// The networks LightCraft has written itself (`crates/denoise/src/candle`), which a [`Network::Safetensors`] model
/// gives the weights of.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub enum Architecture {
    /// PMRID's network (Wang et al., "Practical Deep Raw Image Denoising on Mobile Devices", ECCV 2020): a U-Net of
    /// separable convolutions that takes the four packed planes and gives them back cleaned (it predicts the noise and
    /// subtracts it). LightCraft demosaics its answer with the built-in denoiser's demosaic, so the model still gives
    /// camera RGB like every other ([`Domain::BayerToRgb`]).
    Pmrid,
}

impl Architecture {
    pub fn id(self) -> &'static str {
        match self {
            Architecture::Pmrid => "pmrid",
        }
    }
}

/// How a photo's noise is brought to the noise the network was trained on, before it runs (and back after).
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "camelCase")]
pub enum NoiseMap {
    /// Use the samples as they are.
    None,
    /// The k-sigma transform (PMRID's paper, eq. 7): the photo's noise (variance = `a · signal + b`, measured from the
    /// photo) is mapped onto the training noise `a`, `b` (the same units: black 0, white 1) by `x' = s · x + o` with
    /// `s = a / a_photo`, and the answer is mapped back.
    KSigma { a: f32, b: f32 },
}

/// What the model file is and how it is run.
#[derive(Clone, Copy, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(tag = "format", rename_all = "camelCase")]
pub enum Network {
    /// `model.onnx`: an ONNX file of the operators LightCraft's own executor reads (`crates/denoise/src/onnx.rs`).
    #[default]
    Onnx,
    /// `model.safetensors`: the weights of a network LightCraft has written in candle (`architecture`), run on the
    /// graphics card (Metal on macOS) or the processor.
    #[serde(rename_all = "camelCase")]
    Safetensors {
        architecture: Architecture,
        #[serde(default = "no_noise_map")]
        noise: NoiseMap,
        /// The samples (after `noise`) are multiplied by this before the network and divided after (PMRID: 256).
        #[serde(default = "unit_scale")]
        input_scale: f32,
    },
}

fn no_noise_map() -> NoiseMap {
    NoiseMap::None
}

fn unit_scale() -> f32 {
    1.0
}

impl Network {
    pub fn is_onnx(&self) -> bool {
        matches!(self, Network::Onnx)
    }

    /// The name the model file has in its folder of the models folder.
    pub fn file_name(&self) -> &'static str {
        match self {
            Network::Onnx => "model.onnx",
            Network::Safetensors { .. } => "model.safetensors",
        }
    }

    /// The file extension (lowercase, without the dot) of a model file of this kind.
    pub fn extension(&self) -> &'static str {
        match self {
            Network::Onnx => "onnx",
            Network::Safetensors { .. } => "safetensors",
        }
    }
}

/// Model file extensions a `denoise-model.json` may sit beside.
pub const MODEL_EXTENSIONS: [&str; 2] = ["onnx", "safetensors"];

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct DenoiserManifest {
    /// Lowercase letters, digits, `.` `_` `-`.
    pub id: String,
    pub name: String,
    pub version: String,
    #[serde(default)]
    pub licence: Licence,
    /// Where the weights come from (an `https://` page or file), for the user's information.
    #[serde(default)]
    pub source: Option<String>,
    /// Lowercase hex SHA-256 of the `.onnx` file, when known.
    #[serde(default)]
    pub sha256: Option<String>,
    #[serde(default)]
    pub size_bytes: Option<u64>,
    /// What the model was trained on, in plain words; says "undisclosed" when it is.
    #[serde(default)]
    pub provenance: String,
    pub domain: Domain,
    /// Side of the square the model runs on, in packed cells (the file's fixed input size).
    pub tile: u32,
    /// Cells neighbouring tiles share; the picture is blended across them.
    pub overlap: u32,
    pub gain: Gain,
    /// What the model file is (absent: an ONNX file, as every manifest before this field).
    #[serde(default, skip_serializing_if = "Network::is_onnx")]
    pub network: Network,
}

#[derive(Debug, PartialEq, thiserror::Error)]
pub enum ManifestError {
    #[error("`{0}` is not valid: {1}")]
    Field(&'static str, String),
    #[error("not a denoise-model manifest: {0}")]
    Parse(String),
    #[error("a manifest may be at most {MAX_MANIFEST_BYTES} bytes")]
    TooLarge,
}

fn bad<T>(field: &'static str, why: impl Into<String>) -> Result<T, ManifestError> {
    Err(ManifestError::Field(field, why.into()))
}

fn text(field: &'static str, s: &str, min: usize, max: usize) -> Result<(), ManifestError> {
    let n = s.chars().count();
    if n < min || n > max {
        return bad(field, format!("must be {min} to {max} characters"));
    }
    if s.chars().any(|c| c.is_control() && c != '\n') {
        return bad(field, "must not contain control characters");
    }
    Ok(())
}

fn url(field: &'static str, u: &Option<String>) -> Result<(), ManifestError> {
    match u {
        None => Ok(()),
        Some(u) if u.len() <= 500 && u.starts_with("https://") && !u.chars().any(|c| c.is_control() || c.is_whitespace()) => Ok(()),
        Some(_) => bad(field, "must be an https:// address of at most 500 characters"),
    }
}

/// Check a manifest from outside before trusting it.
pub fn validate(m: &DenoiserManifest) -> Result<(), ManifestError> {
    if !valid_id(&m.id) {
        return bad("id", "1 to 64 lowercase letters, digits, `.`, `_` or `-`, starting with a letter or digit");
    }
    text("name", &m.name, 1, 80)?;
    text("version", &m.version, 1, 40)?;
    text("provenance", &m.provenance, 0, 1000)?;
    text("licence.name", &m.licence.name, 0, 80)?;
    text("licence.notice", &m.licence.notice, 0, 2000)?;
    url("licence.url", &m.licence.url)?;
    url("source", &m.source)?;
    if let Some(h) = &m.sha256
        && !(h.len() == 64 && h.bytes().all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b)))
    {
        return bad("sha256", "must be 64 lowercase hex digits");
    }
    if let Some(n) = m.size_bytes
        && (n == 0 || n > MAX_MODEL_BYTES)
    {
        return bad("sizeBytes", format!("must be between 1 and {MAX_MODEL_BYTES}"));
    }
    // a U-Net with four poolings wants a multiple of 16; nothing smaller than 64 or larger than 2048 is a tile
    if !(64..=2048).contains(&m.tile) || !m.tile.is_multiple_of(16) {
        return bad("tile", "must be a multiple of 16 from 64 to 2048");
    }
    // tiles must overlap by less than half, or one tile's blend ramps would run into its other neighbour's
    if !m.overlap.is_multiple_of(2) || m.overlap < 16 || m.overlap.saturating_mul(2) >= m.tile {
        return bad("overlap", "must be even, at least 16 and less than half the tile");
    }
    if let Gain::MatchMean { nominal, max_deviation } = m.gain {
        if !(nominal.is_finite() && (1e-6..=1e12).contains(&nominal)) {
            return bad("gain.nominal", "must be between 0.000001 and 1000000000000");
        }
        if !(max_deviation.is_finite() && (0.0..=0.5).contains(&max_deviation)) {
            return bad("gain.maxDeviation", "must be between 0 and 0.5");
        }
    }
    if let Network::Safetensors { noise, input_scale, .. } = m.network {
        if !(input_scale.is_finite() && (1e-3..=1e6).contains(&input_scale)) {
            return bad("network.inputScale", "must be between 0.001 and 1000000");
        }
        if let NoiseMap::KSigma { a, b } = noise {
            // the training noise at white: a deviation of 0.03 % to 100 % of white, and a floor of at most a tenth
            if !(a.is_finite() && (1e-7..=1.0).contains(&a)) {
                return bad("network.noise.a", "must be between 0.0000001 and 1");
            }
            if !(b.is_finite() && (0.0..=0.1).contains(&b)) {
                return bad("network.noise.b", "must be between 0 and 0.1");
            }
        }
        // a network that gives the input back cleaned is demosaiced by LightCraft, so it must take the four planes
        if m.domain != Domain::BayerToRgb {
            return bad("domain", "a safetensors network must be bayerToRgb");
        }
    }
    Ok(())
}

/// Read and validate a `denoise-model.json`.
pub fn parse(bytes: &[u8]) -> Result<DenoiserManifest, ManifestError> {
    if bytes.len() > MAX_MANIFEST_BYTES {
        return Err(ManifestError::TooLarge);
    }
    let bytes = bytes.strip_prefix(&[0xEF, 0xBB, 0xBF]).unwrap_or(bytes);
    let m: DenoiserManifest = serde_json::from_slice(bytes).map_err(|e| ManifestError::Parse(e.to_string()))?;
    validate(&m)?;
    Ok(m)
}

#[cfg(test)]
mod tests {
    use super::*;

    pub(crate) fn sample() -> DenoiserManifest {
        DenoiserManifest {
            id: "nind-bayer".into(),
            name: "Bayer denoiser".into(),
            version: "1".into(),
            licence: Licence::default(),
            source: Some("https://example.org/model.onnx".into()),
            sha256: None,
            size_bytes: Some(31_000_000),
            provenance: "trained on real raw noise pairs".into(),
            domain: Domain::BayerToRgb,
            tile: 512,
            overlap: 64,
            gain: Gain::MatchMean { nominal: 1.0e6, max_deviation: 0.05 },
            network: Network::Onnx,
        }
    }

    fn pmrid() -> DenoiserManifest {
        DenoiserManifest {
            gain: Gain::None,
            network: Network::Safetensors { architecture: Architecture::Pmrid, noise: NoiseMap::KSigma { a: 1.0e-3, b: 3.2e-6 }, input_scale: 256.0 },
            ..sample()
        }
    }

    #[test]
    fn a_good_manifest_round_trips() {
        let m = sample();
        let json = serde_json::to_vec(&m).unwrap();
        assert_eq!(parse(&json).unwrap(), m);
    }

    #[test]
    fn bad_fields_are_named() {
        let cases: Vec<(&str, Box<dyn Fn(&mut DenoiserManifest)>)> = vec![
            ("id", Box::new(|m| m.id = "../evil".into())),
            ("id", Box::new(|m| m.id = String::new())),
            ("name", Box::new(|m| m.name = "x\u{7}".into())),
            ("source", Box::new(|m| m.source = Some("http://insecure".into()))),
            ("sha256", Box::new(|m| m.sha256 = Some("XYZ".into()))),
            ("sizeBytes", Box::new(|m| m.size_bytes = Some(0))),
            ("tile", Box::new(|m| m.tile = 500)),
            ("tile", Box::new(|m| m.tile = 4096)),
            ("overlap", Box::new(|m| m.overlap = 300)),
            ("overlap", Box::new(|m| m.overlap = 63)),
            ("gain.maxDeviation", Box::new(|m| m.gain = Gain::MatchMean { nominal: 1.0, max_deviation: f32::NAN })),
            ("gain.maxDeviation", Box::new(|m| m.gain = Gain::MatchMean { nominal: 1.0, max_deviation: 2.0 })),
            ("gain.nominal", Box::new(|m| m.gain = Gain::MatchMean { nominal: 0.0, max_deviation: 0.1 })),
            ("gain.nominal", Box::new(|m| m.gain = Gain::MatchMean { nominal: f32::INFINITY, max_deviation: 0.1 })),
        ];
        for (field, edit) in cases {
            let mut m = sample();
            edit(&mut m);
            match validate(&m) {
                Err(ManifestError::Field(f, _)) => assert_eq!(f, field),
                other => panic!("{field}: {other:?}"),
            }
        }
    }

    #[test]
    fn an_onnx_manifest_reads_and_writes_as_before_the_network_field() {
        // a manifest written before `network` existed is an ONNX model, and is written back without the field
        let mut v = serde_json::to_value(sample()).unwrap();
        assert!(v.get("network").is_none(), "{v}");
        v.as_object_mut().unwrap().remove("network");
        let m: DenoiserManifest = serde_json::from_value(v).unwrap();
        assert_eq!(m.network, Network::Onnx);
        assert_eq!(m.network.file_name(), "model.onnx");
    }

    #[test]
    fn a_safetensors_manifest_round_trips_and_is_checked() {
        let m = pmrid();
        let json = serde_json::to_vec(&m).unwrap();
        assert_eq!(parse(&json).unwrap(), m);
        assert_eq!(m.network.file_name(), "model.safetensors");
        let text = String::from_utf8(json).unwrap();
        assert!(
            text.contains(
                r#""network":{"format":"safetensors","architecture":"pmrid","noise":{"kind":"kSigma","a":0.001,"b":0.0000032},"inputScale":256.0}"#
            ),
            "{text}"
        );
        // the defaults: no noise map, a scale of 1
        let short = text.replace(r#","noise":{"kind":"kSigma","a":0.001,"b":0.0000032},"inputScale":256.0"#, "");
        let read = parse(short.as_bytes()).unwrap();
        assert_eq!(read.network, Network::Safetensors { architecture: Architecture::Pmrid, noise: NoiseMap::None, input_scale: 1.0 });
        // an architecture LightCraft has not written is refused, naming the ones it has
        let unknown = text.replace(r#""architecture":"pmrid""#, r#""architecture":"resnet9000""#);
        match parse(unknown.as_bytes()) {
            Err(ManifestError::Parse(why)) => assert!(why.contains("pmrid"), "{why}"),
            other => panic!("{other:?}"),
        }
        let cases: Vec<(&str, Network)> = vec![
            ("network.inputScale", Network::Safetensors { architecture: Architecture::Pmrid, noise: NoiseMap::None, input_scale: f32::NAN }),
            ("network.inputScale", Network::Safetensors { architecture: Architecture::Pmrid, noise: NoiseMap::None, input_scale: 0.0 }),
            (
                "network.noise.a",
                Network::Safetensors { architecture: Architecture::Pmrid, noise: NoiseMap::KSigma { a: 0.0, b: 0.0 }, input_scale: 1.0 },
            ),
            (
                "network.noise.a",
                Network::Safetensors { architecture: Architecture::Pmrid, noise: NoiseMap::KSigma { a: f32::INFINITY, b: 0.0 }, input_scale: 1.0 },
            ),
            (
                "network.noise.b",
                Network::Safetensors { architecture: Architecture::Pmrid, noise: NoiseMap::KSigma { a: 0.001, b: -1.0 }, input_scale: 1.0 },
            ),
            (
                "network.noise.b",
                Network::Safetensors { architecture: Architecture::Pmrid, noise: NoiseMap::KSigma { a: 0.001, b: f32::NAN }, input_scale: 1.0 },
            ),
        ];
        for (field, network) in cases {
            match validate(&DenoiserManifest { network, ..pmrid() }) {
                Err(ManifestError::Field(f, _)) => assert_eq!(f, field),
                other => panic!("{field}: {other:?}"),
            }
        }
    }

    #[test]
    fn parse_is_bounded_and_tolerant() {
        assert_eq!(parse(&vec![b' '; MAX_MANIFEST_BYTES + 1]), Err(ManifestError::TooLarge));
        assert!(matches!(parse(b"{ nope"), Err(ManifestError::Parse(_))));
        assert!(matches!(parse(b""), Err(ManifestError::Parse(_))));
        let mut with_bom = vec![0xEF, 0xBB, 0xBF];
        with_bom.extend(serde_json::to_vec(&sample()).unwrap());
        assert!(parse(&with_bom).is_ok());
    }
}
