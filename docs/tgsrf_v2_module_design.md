# TGS-RF V2 Module Design Notes

This document explains what the current TGS-RF V2 initializer does, how each module is implemented, which parameters it changes, and how the current experiments should be interpreted.

Current code branch:

```text
wavelet-triplane-research
```

Main entry files:

```text
train_rfid.py
scene/tgsrf/initializer.py
scene/tgsrf/rf_spectrum_encoder.py
scene/tgsrf/rf_geometry_encoder.py
scene/tgsrf/rf_point_decoder.py
scene/tgsrf/rf_triplane_field.py
scene/tgsrf/rf_geometry_aware_triplane.py
scene/tgsrf/rf_gaussian_decoder.py
scene/tgsrf/encoding.py
arguments/configs/rfid/exp_tgsrf.yaml
run_rfid_triplane_experiments.sh
```

## 1. Why V2 Was Added

The earlier triplane experiments mostly adjusted Gaussian attenuation, or used triplane features as a lightweight gate. That was too weak for the current research goal, because it did not change the Gaussian representation deeply enough.

V2 tries to move closer to a TGS / MVSplat / pixelSplat style design:

```text
RF observation + TX/RX geometry
        -> spectrum tokens
        -> candidate/anchor features
        -> geometry-aware triplane query
        -> Gaussian parameter decoder
        -> materialized Gaussian initialization
        -> standard GSRF optimization
```

The main idea is no longer only:

```text
triplane predicts attenuation correction
```

Instead V2 tries to initialize multiple Gaussian parameters:

```text
xyz center
attenuation
scale
FLE features / rotation passthrough
```

The current dense 30K V2 runs use `refine` mode, so V2 starts from the baseline GSRF Gaussian points and predicts bounded residuals. The more aggressive non-grid `generate` mode exists in code, but it is not the current stable experimental setting.

## 2. High-Level Pipeline

The training entry is in `train_rfid.py`.

When `use_tgsrf_initializer: true`, the code runs:

```python
run_tgsrf_init_warmup(scene, gaussians, model_args, optimization_para_args, pipeline_para_args, render)
```

This happens before:

```python
gaussians.training_setup(...)
```

That means V2 is an initialization module, not a persistent renderer replacement. During warm-up, V2 temporarily overrides Gaussian parameters inside `render_rfid`. After warm-up, it writes one static initialized Gaussian set back into `GaussianModel`. Then normal GSRF training and densification continue.

The warm-up loop is:

```text
1. choose one train spectrum viewpoint
2. encode its RF spectrum and TX/RX
3. predict temporary Gaussian parameters
4. render with override_xyz / override_attenuation / override_scaling / override_features / override_rotation
5. compute GSRF-style reconstruction loss
6. update the V2 initializer
7. after warm-up, average predictions over several train views
8. materialize averaged Gaussian parameters into GaussianModel
```

The materialization step is important:

```text
V2 is used only before normal training.
Inference does not run the V2 network.
Inference only loads the saved Gaussian checkpoint.
```

## 3. V2 Modes

V2 supports two modes in `TGSRFInitializer`.

### 3.1 `refine` Mode

Current dense V2 experiments use:

```yaml
tgsrf_mode: "refine"
```

In this mode:

```text
seed_xyz = baseline GSRF Gaussian xyz
num_points = number of baseline points
```

So the point count and initial point layout still come from the original GSRF initializer. V2 predicts residual changes around those points.

This is safer because it does not destroy the baseline initialization. It also makes the comparison more controlled:

```text
baseline GSRF
vs
baseline GSRF points + V2 RF-aware parameter refinement
```

### 3.2 `generate` Mode

The code also supports:

```yaml
tgsrf_mode: "generate"
```

In this mode, V2 builds new non-grid anchors using `build_rf_ray_seeds`:

```text
high-energy angular spectrum rays
TX-RX path samples
Sobol/random volume samples
```

This was added to avoid relying on the original grid/cube-like point creation. However, current stable dense experiments did not use this mode. Earlier generated-anchor attempts were less stable, so the current V2 dense runs use `refine`.

## 4. Module 1: RF Spectrum Encoder

File:

```text
scene/tgsrf/rf_spectrum_encoder.py
```

Class:

```python
RFSpectrumEncoder
```

Purpose:

Encode the current RF spatial spectrum into tokens, similar to how image-conditioned Gaussian methods encode image features.

Input:

```text
spectrum: [H, W], RFID angular spectrum
tx: [3]
rx: [3]
```

For RFID:

```text
H = 90 elevation bins
W = 360 azimuth bins
```

The encoder augments the spectrum with angular coordinate channels:

```text
spectrum
sin(azimuth)
cos(azimuth)
sin(elevation)
cos(elevation)
```

So the CNN input has 5 channels:

```text
[spectrum, sin(az), cos(az), sin(el), cos(el)]
```

Then a small CNN produces a feature map:

```text
Conv2d(5 -> hidden_dim/2)
Conv2d(hidden_dim/2 -> hidden_dim)
Conv2d(hidden_dim -> hidden_dim, kernel=stride=patch_stride)
```

With the current default:

```yaml
tgsrf_hidden_dim: 128
tgsrf_spectrum_patch_stride: 8
```

The spectrum becomes a token grid of roughly:

```text
90 x 360 -> 11 x 45 tokens
```

TX/RX are Fourier encoded and projected:

```text
concat(tx, rx) -> Fourier encoding -> MLP -> txrx embedding
```

The TX/RX embedding is added to every spectrum token. The output is:

```text
tokens:      [1, N_tokens, hidden_dim]
global:      [1, hidden_dim]
feature_map: [1, hidden_dim, H_patch, W_patch]
```

Role in the method:

```text
tokens: cross-attention memory for point decoder
global: global RF condition for Gaussian decoder
feature_map: local angular feature lookup for each candidate point
```

Important limitation:

During warm-up this encoder uses the training spectrum itself as conditioning input. After warm-up, the V2 network is not used at inference. Therefore the spectrum encoder currently helps learn a static scene initialization; it is not yet a test-time feed-forward predictor.

## 5. Module 2: RF Geometry Encoder

File:

```text
scene/tgsrf/rf_geometry_encoder.py
```

Class:

```python
RFGeometryEncoder
```

Purpose:

Encode RF propagation geometry for each candidate Gaussian point.

Input:

```text
xyz: [N, 3]
tx:  [3]
rx:  [3]
```

Raw geometry features:

```text
scaled xyz
unit direction from point to RX
unit direction from point to TX
unit direction from RX to point
unit direction from TX to point
d_rx / scene_scale
d_tx / scene_scale
(d_rx + d_tx) / scene_scale
sin(2*pi*path_length / wavelength)
cos(2*pi*path_length / wavelength)
```

The phase proxy uses:

```text
wavelength = c / frequency
frequency = 915 MHz by default
```

These raw features are Fourier encoded:

```text
raw -> fourier_encode(raw, tgsrf_geometry_freqs)
```

Then an MLP maps them to:

```text
geometry_feature: [N, hidden_dim]
```

Why this matters:

The encoder gives the decoder explicit RF-relevant information:

```text
where the point is
how far it is from TX/RX
whether its path length has phase-sensitive structure
which propagation direction it corresponds to
```

This is the NeRF2-style RF geometry component in V2.

## 6. Module 3: Point Decoder / Candidate Transformer

File:

```text
scene/tgsrf/rf_point_decoder.py
```

Classes/functions:

```python
RFPointDecoder
CrossAttentionBlock
point_to_local_spectrum
build_rf_ray_seeds
```

Purpose:

Produce point-level features by letting candidate Gaussian anchors attend to RF spectrum tokens.

The input per point is:

```text
normalized point coordinate
learnable point latent
RF geometry feature
local spectrum feature
```

This is projected to a query token. The query token attends to all RF spectrum tokens:

```text
point query -> MultiheadAttention -> spectrum tokens
```

The output is:

```text
point_features: [N, hidden_dim]
point_delta:    [N, 3]
point_xyz:      seed_xyz + point_delta
```

The point delta is bounded:

```text
point_delta = tanh(delta_head(point_features)) * tgsrf_point_offset_radius
```

Current dense V2 setting:

```bash
--tgsrf_point_refine_start 999999
```

So in the dense V2 experiments:

```text
point_delta is disabled
point_xyz = seed_xyz
```

However, the cross-attention point features are still used by the Gaussian decoder. So the point decoder is active as a feature extractor, but not as a point mover in the current dense V2 runs.

## 7. Module 4: RF Triplane Field

File:

```text
scene/tgsrf/rf_triplane_field.py
```

Class:

```python
RFTriplaneField
```

Purpose:

Provide a compact 3D spatial feature prior through three axis-aligned 2D planes:

```text
P_xy
P_xz
P_yz
```

The point coordinate is normalized using a padded bounding box:

```text
xyz_min, xyz_max = scene_bounds_from_gaussians(gaussians, tgsrf_bbox_pad_scale)
xyz_norm in [-1, 1]^3
```

Then the field samples:

```text
f_xy = sample(P_xy, x, y)
f_xz = sample(P_xz, x, z)
f_yz = sample(P_yz, y, z)
```

The output is concatenated:

```text
triplane_features = concat(f_xy, f_xz, f_yz)
```

With current defaults:

```yaml
tgsrf_triplane_channels: 32
```

the decoder receives:

```text
3 * 32 = 96 triplane feature dimensions
```

### 7.1 Direct Triplane

When:

```yaml
tgsrf_triplane_type: "direct"
```

the module directly optimizes:

```text
planes: [3, C, R, R]
```

Current dense direct V2 used:

```text
resolution = 128
channels = 32
```

### 7.2 Wavelet Triplane

When:

```yaml
tgsrf_triplane_type: "haar"
```

the module stores:

```text
LL coarse coefficients
LH / HL / HH high-frequency coefficients per level
```

Then it reconstructs full-resolution planes using inverse Haar steps.

Current defaults:

```yaml
tgsrf_wavelet_levels: 2
tgsrf_wavelet_high_init: 0.0
```

So high-frequency bands start from zero unless explicitly changed.

The loss:

```text
tgsrf_wavelet_l1 * mean(abs(high-frequency coefficients))
```

encourages sparse high-frequency plane coefficients.

### 7.3 C2F Schedule

When:

```yaml
tgsrf_wavelet_c2f: true
```

the warm-up progressively activates wavelet high-frequency levels:

```text
early warm-up: coarse LL only / fewer high-frequency levels
later warm-up: more high-frequency levels
end: all levels active
```

In code:

```python
active = ((step - 1) * (levels + 1)) // warmup_iters
```

### 7.4 Important Note About `bior44`

The code currently accepts:

```text
haar
bior44
bior4.4
```

But the implementation still uses the stable Haar reconstruction path for both Haar and bior44 switches:

```python
# V1 uses stable Haar reconstruction for both Haar and bior44 switches.
```

So current `bior44` experiments should be described carefully as a bior44 ablation flag, not a complete biorthogonal 4.4 filter-bank implementation.

## 8. Module 5: Geometry-Aware Triplane Encoder

File:

```text
scene/tgsrf/rf_geometry_aware_triplane.py
```

Class:

```python
RFGeometryAwareTriplaneEncoder
```

Purpose:

Inject point-level transformer features back into three triplane grids.

The module projects point features:

```text
point_features [N, hidden_dim] -> feat [N, channels]
```

Then scatters them into three axis-aligned grids:

```text
xy plane
xz plane
yz plane
```

The scattered planes are scaled by a learnable scalar:

```text
extra_planes = scale * scattered_feature_planes
```

Then `RFTriplaneField.query` adds them to the learned triplane before sampling:

```text
planes = learned_planes + geometry_aware_extra_planes
```

Why this was added:

The learned triplane alone is static and scene-specific. The geometry-aware triplane gives it view/sample-conditioned information from:

```text
spectrum tokens
TX/RX geometry
point-level cross-attention
```

Known limitation:

The scatter uses rounded grid indices. Gradients flow to the point features, but not smoothly through point coordinates. This is acceptable for V2 because current dense runs disable point offset anyway.

## 9. Module 6: Gaussian Decoder

File:

```text
scene/tgsrf/rf_gaussian_decoder.py
```

Class:

```python
RFGaussianDecoder
```

Purpose:

Predict bounded residual updates for Gaussian parameters.

Input:

```text
point_features
triplane_features
geometry_features
local_spectrum_features
global_spectrum_token
```

These are concatenated and passed through a shared MLP trunk:

```text
shared MLP -> hidden state
```

Then three separate heads predict:

```text
delta_xyz
attenuation_delta
log_scale_delta
```

The residuals are bounded:

```text
delta_xyz = tanh(xyz_head) * tgsrf_gaussian_offset_radius
attenuation_delta = tanh(att_head) * tgsrf_att_radius
log_scale_delta = tanh(scale_head) * tgsrf_scale_radius
```

Current dense V2 settings:

```text
tgsrf_gaussian_offset_radius = 0.05
tgsrf_att_radius = 0.05
tgsrf_scale_radius = 0.10
```

The final Gaussian parameters are:

```text
final_xyz = point_xyz + gaussian_delta
attenuation_raw = base_attenuation_raw + attenuation_delta
scaling_raw = base_scaling_raw + log_scale_delta
```

Then clamps are applied:

```text
attenuation_raw in [-8, 2]
scaling_raw in [-6, 0]
final_xyz inside bbox
```

Current limitation:

The decoder has separate heads, but it still uses one shared trunk for physically different quantities. This is stronger than the older single-output gate, but it is not yet a fully disentangled per-parameter decoder.

## 10. Which Gaussian Parameters V2 Changes

Current V2 can affect:

```text
xyz center
attenuation
scale
```

Current V2 does not truly learn:

```text
rotation residual
FLE feature residual
top-K pruning / keep probability
new densification policy
renderer changes
```

Details:

### 10.1 XYZ

In current dense runs:

```text
point offset disabled
Gaussian decoder offset enabled after half warm-up
```

For warm-up 2000:

```text
gaussian offset starts at step 1000
```

So only the Gaussian decoder's bounded local offset changes position:

```text
xyz <- xyz + delta_xyz
```

### 10.2 Attenuation

V2 predicts a residual in raw/logit space:

```text
attenuation_raw <- base_attenuation_raw + attenuation_delta
attenuation <- sigmoid(attenuation_raw)
```

This is more expressive than old gate-only attenuation modulation because it is coupled with geometry, spectrum tokens, triplane features, and scale/xyz residuals.

### 10.3 Scale

V2 predicts:

```text
scaling_raw <- base_scaling_raw + log_scale_delta
scale <- exp(scaling_raw)
```

This means V2 can change Gaussian spatial support before standard GSRF optimization.

### 10.4 Rotation

In refine mode, rotation is copied from baseline:

```text
base_rotation_raw = gaussians._rotation
```

No rotation residual is predicted.

### 10.5 FLE Features

In refine mode:

```yaml
tgsrf_refine_features: false
```

So warm-up does not train FLE features. It passes through the initial baseline features. After materialization, normal GSRF training still optimizes FLE features as usual.

## 11. Warm-Up Loss

Warm-up loss is in `scene/tgsrf/initializer.py`.

The reconstruction loss follows GSRF:

```text
L_recon = (1 - lambda_ssim - lambda_fourier) * L1
          + lambda_ssim * SSIM_loss
          + lambda_fourier * Fourier_loss
```

Then V2 adds regularizers:

```text
L = L_recon
    + lambda_point * ||point_delta||^2
    + lambda_gauss * ||gaussian_delta||^2
    + lambda_att * ||attenuation_delta||^2
    + lambda_scale * ||log_scale_delta||^2
    + lambda_wavelet * wavelet_l1
```

Current defaults:

```yaml
tgsrf_point_offset_l2: 0.02
tgsrf_gaussian_offset_l2: 0.02
tgsrf_att_l2: 0.02
tgsrf_scale_l2: 0.01
tgsrf_wavelet_l1: 0.0 for direct
tgsrf_wavelet_l1: 0.001 for Haar dense experiment
```

Gradient clipping:

```yaml
tgsrf_grad_clip: 1.0
```

Non-finite loss handling:

```text
if loss is non-finite:
    skip the step
```

The code also clamps base attenuation and scaling and cleans non-finite FLE features.

## 12. Materialization

At the end of warm-up:

```text
select tgsrf_finalize_views train views
predict parameters for each selected view
average xyz / attenuation_raw / scaling_raw across those views
write averaged values into GaussianModel
```

Current default:

```yaml
tgsrf_finalize_views: 16
```

This design makes V2 produce a single static Gaussian initialization for the whole scene.

Advantage:

```text
standard GSRF training and inference remain unchanged
```

Disadvantage:

```text
view-conditioned predictions are averaged away
```

This may be one reason V2 does not show a strong dense-sample gain: the spectrum-conditioned network learns per-view corrections during warm-up, but only a static average is kept.

## 13. Current Dense V2 Experiment Settings

The dense V2 entries are in `run_rfid_triplane_experiments.sh`.

### 13.1 Direct Dense 30K

Command entry:

```bash
GPU=0 V2_WARMUP=2000 ./run_rfid_triplane_experiments.sh tgsrf_v2_direct_dense30k
```

Effective key flags:

```text
ratio_train = 0.8
iterations = 30000
tgsrf_mode = refine
tgsrf_triplane_type = direct
tgsrf_warmup_iters = 2000
tgsrf_gaussian_offset_radius = 0.05
tgsrf_gaussian_offset_start = 1000
tgsrf_point_refine_start = 999999
```

### 13.2 Haar Dense 30K

Command entry:

```bash
GPU=1 V2_WARMUP=2000 ./run_rfid_triplane_experiments.sh tgsrf_v2_haar_dense30k
```

Additional flags:

```text
tgsrf_triplane_type = haar
tgsrf_wavelet_l1 = 0.001
tgsrf_wavelet_c2f = true
```

## 14. Current Dense Results

Use `inference_rfid.py` results, not `training_report` random 5-sample logs.

| Method | Iter | PSNR | MSE | SSIM |
|---|---:|---:|---:|---:|
| baseline dense | 30K | 22.0571 | 0.009428 | 0.8165 |
| old triplane gate dense | 30K | 21.9720 | 0.009467 | 0.8148 |
| old bior44 dense | 30K | 21.9392 | 0.009644 | 0.8144 |
| V2 direct dense | 30K | 21.9376 | 0.009458 | 0.8128 |
| V2 Haar dense | 30K | 21.9296 | 0.009542 | 0.8128 |
| V2 direct dense | 20K | 21.3880 | 0.010404 | 0.7985 |
| V2 Haar dense | 20K | 21.4003 | 0.010498 | 0.7976 |

Interpretation:

```text
V2 dense does not beat the dense baseline.
Haar does not improve the final dense result.
The dense setting is currently not the strongest paper story.
```

Training logs showed 20K test PSNR around 25 dB, but that was caused by `training_report` using only 5 random test samples. Full inference over all 1225 test samples gives about 21.39 dB at 20K.

## 15. What V2 Has Already Tried

V2 has already implemented these research ideas:

```text
NeRF2-style RF geometry features
RF spectrum encoder
candidate-wise anchor representation
point-to-spectrum local angular feature lookup
cross-attention point feature decoding
TGS-style triplane Gaussian representation
optional wavelet-domain triplane coefficients
coarse-to-fine wavelet activation
multi-parameter Gaussian initialization
warm-up then materialize into standard GSRF
```

Compared with the earlier gate/triplane method, V2 is a major architectural expansion.

## 16. What V2 Has Not Yet Done

V2 still does not implement several important ideas from MVSplat / pixelSplat / TGS-like papers:

```text
candidate-wise keep probability
top-K pruning
cost-volume-style explicit RF candidate scoring
separate decoders for xyz / scale / attenuation / FLE
rotation residual prediction
persistent conditional modulation during main training
test-time feed-forward Gaussian prediction
true bior4.4 wavelet filter-bank reconstruction
wavelet subband reconstruction metric/loss on final spectrum
```

Most importantly, the current V2 does not yet "learn to select points." It only refines all existing points.

## 17. Main Technical Weaknesses Observed

### 17.1 Dense Dataset Leaves Little Room

The dense 80/20 baseline is already around 22.06 dB. Small initialization changes are easily washed out by 30K standard GSRF optimization.

### 17.2 Spectrum Conditioning Is Averaged Away

During warm-up, V2 predicts view-conditioned parameters. But at materialization time, predictions from 16 train views are averaged into one static Gaussian set.

This can weaken the value of:

```text
spectrum encoder
TX/RX-aware point features
geometry-aware triplane
```

### 17.3 Current V2 Does Not Select or Prune Candidates

MVSplat/pixelSplat-style methods are strong because they change candidate selection:

```text
candidate score
depth probability
keep probability
top-K
opacity as probability
```

V2 currently does not have this mechanism.

### 17.4 Physical Quantities Still Share One Trunk

The Gaussian decoder has separate output heads, but a shared trunk. This may be too coupled for RF:

```text
xyz controls geometry
scale controls support
attenuation controls RF strength
FLE controls directional complex response
```

A stronger V3 should likely use parameter-specific decoders or at least parameter-specific adapters.

### 17.5 No Persistent V2 During Main Training

After materialization, V2 disappears. If the main benefit is conditional modulation, the current design may discard it too early.

## 18. Recommended Next Decisions

Based on current results, the best next directions are:

### A. Use sparse / 20% as the main experimental setting

Dense 80/20 is not showing meaningful improvement. Sparse data is where initialization should matter more.

### B. Add candidate-wise keep probability

The most important missing module is:

```text
score_i / keep_prob_i / top-K pruning
```

This would move the method from:

```text
adjust all Gaussians slightly
```

to:

```text
learn which Gaussian candidates should matter
```

### C. Split Gaussian decoders by physical parameter

Instead of one shared trunk:

```text
shared -> xyz_head / att_head / scale_head
```

use:

```text
geometry decoder -> xyz
support decoder -> scale
power decoder -> attenuation
directional decoder -> FLE features
```

### D. Reconsider persistent conditioning

If spectrum/TX/RX conditioning is important, it may need to remain active during part of main training instead of being averaged into static initialization.

### E. Treat Haar as optional, not core

Current Haar triplane did not beat direct triplane or baseline in dense 30K. For the paper, wavelet should be used only if it helps in sparse/data-efficiency settings or in a clear subband-error analysis.

## 19. One-Paragraph Summary for Paper Notes

TGS-RF V2 is a warm-up based RF-aware Gaussian initializer for GSRF. It keeps the standard GSRF renderer and optimization pipeline, but before main training it encodes each training RF spectrum with angular coordinate channels and TX/RX Fourier embeddings, builds point-level features through cross-attention between Gaussian anchors and spectrum tokens, queries a geometry-aware triplane field at candidate Gaussian locations, and decodes bounded residuals for Gaussian center, attenuation, and scale. The predicted parameters are rendered with the original RF Gaussian renderer during warm-up, trained using the GSRF reconstruction loss plus residual regularization, then averaged across several training views and materialized back into `GaussianModel`. The current dense V2 implementation is architecturally much richer than the earlier triplane gate, but final dense 30K inference does not exceed the baseline, suggesting that the next version should focus on sparse-data settings, candidate-wise point selection, and more disentangled Gaussian parameter decoders.
