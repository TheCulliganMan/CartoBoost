# LSTTN CUDA tensor backend target design

This is an implementation note, not user-facing performance documentation.
It defines the target architecture required before CUDA LSTTN training may be
advertised as available.

## Tensor and CSR layout

CUDA training values will be contiguous `f32`. A supervised input batch is
`[B, T, N, C]` (row-major), where `B <= 32`; forecasts and targets are
`[B, H, N]`. Structural graph buffers are CSR `(u32[N + 1], u32[E], f32[E])`.
Forward and reverse graphs are independently row-normalized. Adaptive
adjacency is sparse over structural edges plus one self candidate per row.

## Required executor boundaries

The completed executor must keep a CUDA context, graph buffers, parameters,
moments, compiled modules, and shape-keyed workspaces alive across batches.
It must implement explicit tensor forward/backward for patch embedding,
Transformer attention/FFN/norm, causal convolution, sparse diffusion,
fusion, masked MAE, clipping, and AdamW—never a scalar tape on CUDA.

Pretraining must retain deterministic whole-patch masking from serialized
step state and freeze MST ranges during supervised training. Checkpoints must
contain only portable model/optimizer/scheduling state and recreate device
buffers on resume.

Parity targets are `1e-4` absolute/relative for deterministic small fixtures
and `2e-3` for sparse central-difference gradients.
