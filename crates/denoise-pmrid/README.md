# lightcraft-denoise-pmrid

L1: the PMRID raw denoising network (Wang et al., ECCV 2020) in candle, which
`lightcraft-denoise`'s `candle` runner runs as an AI Denoise model. It follows the
authors' PyTorch definition (Apache-2.0), so this crate is **Apache-2.0 only**, not
the MIT option of the rest of LightCraft (see `NOTICE`). No weights are included:
see [docs/denoise.md → PMRID](../../docs/denoise.md#pmrid) for the licence of the
published weights and how a user converts and installs them.

Run `cargo test -p lightcraft-denoise-pmrid`: the network is checked against a scalar
version of its layers written from their PyTorch definitions.
