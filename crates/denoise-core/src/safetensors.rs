//! Safetensors weights, read whole from a file of bounded size and checked before anything is built from them (the
//! weights of candle networks, `lightcraft_denoise::candle`; plain Rust, no tensor library).
//!
//! The format (<https://github.com/huggingface/safetensors>): an 8-byte little-endian header length, a JSON header
//! naming each tensor's dtype, shape and byte range, then the bytes. Everything in it is hostile: the header is
//! bounded, every range must lie in the file and match its shape, and every number must be finite. Only `F32` is read.

use std::collections::BTreeMap;
use std::io::Read;
use std::path::Path;

use crate::manifest::MAX_MODEL_BYTES;
use crate::run::Error;

/// Largest JSON header we read.
const MAX_HEADER: u64 = 16 << 20;
/// Most tensors one file may hold.
const MAX_TENSORS: usize = 100_000;
/// Most dimensions one tensor may have.
const MAX_DIMS: usize = 8;

/// One tensor: its shape and its values, row-major.
#[derive(Clone, Debug, PartialEq)]
pub struct Stored {
    pub shape: Vec<usize>,
    pub values: Vec<f32>,
}

/// The tensors of a safetensors file, by name.
#[derive(Debug, Default)]
pub struct Weights {
    tensors: BTreeMap<String, Stored>,
}

fn bad(why: impl Into<String>) -> Error {
    Error::Load(why.into())
}

impl Weights {
    /// Read the file at `path` (at most [`MAX_MODEL_BYTES`]).
    pub fn read(path: &Path) -> Result<Self, Error> {
        let file = std::fs::File::open(path).map_err(|e| bad(format!("{}: {e}", path.display())))?;
        let mut bytes = Vec::new();
        file.take(MAX_MODEL_BYTES + 1).read_to_end(&mut bytes).map_err(|e| bad(format!("{}: {e}", path.display())))?;
        if bytes.len() as u64 > MAX_MODEL_BYTES {
            return Err(bad(format!("{}: a model file may be at most {MAX_MODEL_BYTES} bytes", path.display())));
        }
        Self::parse(&bytes)
    }

    /// Read a safetensors file from memory.
    pub fn parse(bytes: &[u8]) -> Result<Self, Error> {
        let (len, rest) = bytes.split_first_chunk::<8>().ok_or_else(|| bad("not a safetensors file (too short)"))?;
        let n = u64::from_le_bytes(*len);
        if n > MAX_HEADER || n > rest.len() as u64 {
            return Err(bad(format!("not a safetensors file (a header of {n} bytes)")));
        }
        let (header, data) = rest.split_at(n as usize);
        let header: serde_json::Map<String, serde_json::Value> =
            serde_json::from_slice(header).map_err(|e| bad(format!("not a safetensors file: {e}")))?;
        if header.len() > MAX_TENSORS + 1 {
            return Err(bad(format!("the file names more than {MAX_TENSORS} tensors")));
        }
        let mut tensors = BTreeMap::new();
        for (name, entry) in header {
            if name == "__metadata__" {
                continue;
            }
            let dtype = entry.get("dtype").and_then(|d| d.as_str()).unwrap_or("?");
            if dtype != "F32" {
                return Err(bad(format!("tensor `{name}` is {dtype}: LightCraft reads F32 weights (convert them to F32)")));
            }
            let dims = entry.get("shape").and_then(|s| s.as_array()).ok_or_else(|| bad(format!("tensor `{name}` has no shape")))?;
            if dims.len() > MAX_DIMS {
                return Err(bad(format!("tensor `{name}` has more than {MAX_DIMS} dimensions")));
            }
            let shape = dims
                .iter()
                .map(|d| d.as_u64().and_then(|d| usize::try_from(d).ok()))
                .collect::<Option<Vec<usize>>>()
                .ok_or_else(|| bad(format!("tensor `{name}` has a shape that is not a list of sizes")))?;
            let count = shape.iter().try_fold(1usize, |a, &d| a.checked_mul(d)).ok_or_else(|| bad(format!("tensor `{name}` is too large")))?;
            let offsets = entry.get("data_offsets").and_then(|o| o.as_array()).map(|o| o.iter().map(|v| v.as_u64()).collect::<Vec<_>>());
            let (start, end) = match offsets.as_deref() {
                Some([Some(s), Some(e)]) => (*s, *e),
                _ => return Err(bad(format!("tensor `{name}` has no data_offsets [start, end]"))),
            };
            let size = count.checked_mul(4).ok_or_else(|| bad(format!("tensor `{name}` is too large")))?;
            if end.checked_sub(start) != Some(size as u64) {
                return Err(bad(format!("tensor `{name}`: its bytes do not match its shape {shape:?}")));
            }
            let raw = usize::try_from(start)
                .ok()
                .zip(usize::try_from(end).ok())
                .and_then(|(s, e)| data.get(s..e))
                .ok_or_else(|| bad(format!("tensor `{name}` lies outside the file (incomplete or damaged?)")))?;
            let values: Vec<f32> = raw.as_chunks::<4>().0.iter().map(|c| f32::from_le_bytes(*c)).collect();
            if values.iter().any(|v| !v.is_finite()) {
                return Err(bad(format!("tensor `{name}` has values that are not finite")));
            }
            tensors.insert(name, Stored { shape, values });
        }
        Ok(Weights { tensors })
    }

    /// Take tensor `name`, which must have `shape`.
    pub fn take(&mut self, name: &str, shape: &[usize]) -> Result<Vec<f32>, Error> {
        let t = self.tensors.remove(name).ok_or_else(|| bad(format!("the weights have no tensor `{name}`: is this file for this network?")))?;
        if t.shape != shape {
            return Err(bad(format!("tensor `{name}` has the shape {:?}, the network needs {shape:?}", t.shape)));
        }
        Ok(t.values)
    }

    /// Tensor `name`, left in place.
    pub fn get(&self, name: &str) -> Option<&Stored> {
        self.tensors.get(name)
    }

    /// Names not taken yet.
    pub fn left(&self) -> Vec<&str> {
        self.tensors.keys().map(String::as_str).collect()
    }

    /// Add a tensor (for building weights in memory). A shape whose size differs from `values` is an error.
    pub fn insert(&mut self, name: &str, shape: &[usize], values: Vec<f32>) -> Result<(), Error> {
        if shape.iter().try_fold(1usize, |a, &d| a.checked_mul(d)) != Some(values.len()) {
            return Err(bad(format!("tensor `{name}`: {} values do not make the shape {shape:?}", values.len())));
        }
        self.tensors.insert(name.to_string(), Stored { shape: shape.to_vec(), values });
        Ok(())
    }

    /// The tensors as a safetensors file (F32, sorted by name).
    pub fn to_bytes(&self) -> Vec<u8> {
        let mut header = serde_json::Map::new();
        let mut offset = 0usize;
        for (name, t) in &self.tensors {
            let end = offset + 4 * t.values.len();
            header.insert(name.clone(), serde_json::json!({"dtype": "F32", "shape": t.shape, "data_offsets": [offset, end]}));
            offset = end;
        }
        let mut head = serde_json::Value::Object(header).to_string().into_bytes();
        // the data starts 8-byte aligned, as other writers do
        while !(8 + head.len()).is_multiple_of(8) {
            head.push(b' ');
        }
        let mut out = Vec::with_capacity(8 + head.len() + offset);
        out.extend_from_slice(&(head.len() as u64).to_le_bytes());
        out.extend_from_slice(&head);
        for t in self.tensors.values() {
            for v in &t.values {
                out.extend_from_slice(&v.to_le_bytes());
            }
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample() -> Weights {
        let mut w = Weights::default();
        w.insert("a.weight", &[2, 3], (0..6).map(|i| i as f32).collect()).unwrap();
        w.insert("b", &[4], vec![0.5; 4]).unwrap();
        w
    }

    #[test]
    fn weights_round_trip() {
        let bytes = sample().to_bytes();
        let mut w = Weights::parse(&bytes).unwrap();
        assert_eq!(w.left(), ["a.weight", "b"]);
        assert!(w.take("a.weight", &[3, 2]).is_err(), "the wrong shape is refused");
        let mut w2 = Weights::parse(&bytes).unwrap();
        assert_eq!(w2.take("a.weight", &[2, 3]).unwrap(), vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0]);
        assert_eq!(w2.left(), ["b"]);
        assert!(w.take("missing", &[1]).is_err());
        assert!(w.insert("c", &[3], vec![1.0; 2]).is_err());
    }

    #[test]
    fn hostile_files_are_errors_never_panics() {
        let good = sample().to_bytes();
        // every truncation and every flipped byte
        for n in 0..good.len() {
            let _ = Weights::parse(&good[..n]);
        }
        for at in 0..good.len() {
            let mut bad = good.clone();
            bad[at] ^= 0x5a;
            let _ = Weights::parse(&bad);
        }
        let header = |h: &str, data: &[u8]| {
            let mut out = (h.len() as u64).to_le_bytes().to_vec();
            out.extend_from_slice(h.as_bytes());
            out.extend_from_slice(data);
            out
        };
        let cases = [
            header(r#"{"x":{"dtype":"F16","shape":[2],"data_offsets":[0,4]}}"#, &[0; 4]),
            header(r#"{"x":{"dtype":"F32","shape":[2],"data_offsets":[0,4]}}"#, &[0; 8]),
            header(r#"{"x":{"dtype":"F32","shape":[2],"data_offsets":[0,8]}}"#, &[0; 4]),
            header(r#"{"x":{"dtype":"F32","shape":[2],"data_offsets":[8,0]}}"#, &[0; 8]),
            header(r#"{"x":{"dtype":"F32","shape":[18446744073709551615,2],"data_offsets":[0,8]}}"#, &[0; 8]),
            header(r#"{"x":{"dtype":"F32","shape":[-1],"data_offsets":[0,4]}}"#, &[0; 4]),
            header(r#"{"x":{"dtype":"F32","shape":[1,1,1,1,1,1,1,1,1],"data_offsets":[0,4]}}"#, &[0; 4]),
            header(r#"{"x":{"dtype":"F32","shape":[1],"data_offsets":[0]}}"#, &[0; 4]),
            header(r#"{"x":{"dtype":"F32","shape":[1],"data_offsets":[0,4]}}"#, &f32::NAN.to_le_bytes()),
            header(r#"{"x":{"dtype":"F32","shape":[1],"data_offsets":[0,4]}}"#, &f32::INFINITY.to_le_bytes()),
            header(r#"[1,2]"#, &[]),
            header(r#"{"x":5}"#, &[]),
            u64::MAX.to_le_bytes().to_vec(),
            vec![1, 2, 3],
        ];
        for (i, c) in cases.iter().enumerate() {
            assert!(Weights::parse(c).is_err(), "case {i}");
        }
        // a metadata entry is not a tensor
        let ok = header(r#"{"__metadata__":{"format":"pt"},"x":{"dtype":"F32","shape":[1],"data_offsets":[0,4]}}"#, &1f32.to_le_bytes());
        assert_eq!(Weights::parse(&ok).unwrap().left(), ["x"]);
    }

    #[test]
    fn an_oversized_or_missing_file_is_an_error() {
        assert!(Weights::read(Path::new("/does/not/exist.safetensors")).is_err());
    }
}
