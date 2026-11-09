//! PMRID's network (Wang et al., "Practical Deep Raw Image Denoising on Mobile Devices", ECCV 2020) in candle, for
//! `lightcraft_denoise::candle` (which runs it as a denoise model) and the converter in
//! `crates/denoise/examples/pmrid-convert.rs`.
//!
//! The layers, their wiring and the parameter names follow the authors' PyTorch definition
//! (`models/net_torch.py` in <https://github.com/MegEngine/PMRID>, Apache-2.0), so that their published weights,
//! converted to safetensors, load unchanged. It takes the four packed planes `[R, G1, G2, B]` (after the k-sigma
//! transform and the input scale, see [`super`]) and gives them back cleaned: an encoder of four stages of separable
//! 5 × 5 convolutions, each halving the size, a decoder of transposed convolutions that adds back the encoder's
//! features of the same size, and a last convolution whose answer is added to the input (the network predicts the
//! noise to take away).
//!
//! A separable convolution is a depthwise one (each channel by its own small kernel) and a pointwise 1 × 1 one. The
//! depthwise convolution is written here as one multiply-add of the shifted picture per kernel tap, which runs well on
//! the graphics card (a grouped convolution would be a call per channel), and a stride of two is the same sum taken at
//! every other place. Nothing here reads a file: the parameters come from a [`Source`].
//!
//! Modified work (Apache License 2.0, §4(b)): written by the LightCraft contributors in 2026 after `models/net_torch.py`
//! of MegEngine/PMRID (by the PMRID authors, Wang et al.; Apache License 2.0); translated from Python/PyTorch to Rust on
//! candle, with the depthwise and transposed convolutions computed another way. This crate is therefore licensed under
//! the Apache License 2.0 only (not the MIT option of the rest of LightCraft). See NOTICE.
#![forbid(unsafe_code)]
#![deny(clippy::unwrap_used, clippy::expect_used, clippy::panic, clippy::unimplemented, clippy::todo, clippy::unreachable)]

// candle runs natively only (the web build has no AI denoise models), as for SAM 3
#[cfg(not(target_arch = "wasm32"))]
mod net;
#[cfg(not(target_arch = "wasm32"))]
pub use net::{Depthwise, Device, GradStore, Listing, MULTIPLE, PLANES, Pmrid, Source, Tensor, Var, blur_weights, random_weights};
