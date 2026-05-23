# TGS-RF V2 模块设计说明

本文档详细说明当前 TGS-RF V2 做了哪些模块、每个模块的输入输出、它如何接入 GSRF、当前实验到底启用了哪些能力，以及目前实验结果说明了什么。

当前远端仓库：

```text
/home/wys/GSRF
```

当前分支：

```text
wavelet-triplane-research
```

核心文件：

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

## 1. V2 的目标

V2 的目标是把方法从早期的：

```text
triplane / wavelet 只调 attenuation
```

推进到：

```text
利用 RF spectrum、TX/RX 几何、candidate anchors、triplane 特征，
共同预测多个 Gaussian 初始化参数。
```

也就是说，V2 不再只是一个 gate，而是一个完整的 RF-aware Gaussian initializer。

它参考的思想主要来自：

```text
NeRF2: RF 几何、TX/RX、路径长度、相位 proxy
MVSplat / pixelSplat: candidate-wise Gaussian 初始化，不直接自由回归中心
TGS / Hybrid Triplane-Gaussian: point features + triplane features + Gaussian decoder
TriNeRFLet: wavelet-domain triplane，可选 coarse-to-fine
```

当前 V2 的总体数据流是：

```text
训练样本 spectrum + TX/RX
        |
        v
RF spectrum encoder
        |
        v
spectrum tokens / global token / local feature map
        |
        v
candidate Gaussian anchors
        |
        v
RF geometry encoder + point cross-attention decoder
        |
        v
geometry-aware triplane query
        |
        v
Gaussian decoder
        |
        v
delta_xyz / attenuation_delta / scale_delta
        |
        v
临时 override 参数调用 render_rfid
        |
        v
warm-up loss 反向传播
        |
        v
warm-up 结束后 materialize 回 GaussianModel
        |
        v
进入原始 GSRF 30K training
```

## 2. V2 和旧 triplane gate 的区别

旧方法主要是：

```text
Gaussian xyz -> triplane query -> MLP -> attenuation correction
```

V2 做的是：

```text
spectrum encoder
RF geometry encoder
point transformer / cross-attention
geometry-aware triplane
Gaussian multi-parameter decoder
warm-up materialization
```

所以 V2 的代码复杂度和表达能力明显高于旧 gate。

旧 gate 更像：

```text
调整每个 Gaussian 的贡献强弱
```

V2 更像：

```text
根据 RF 观测和几何条件，重新初始化 Gaussian 的位置、强度和空间支持范围
```

不过当前 V2 还没有实现 candidate-wise keep probability / top-K pruning，所以它还没有真正做到“学会选点”。

## 3. V2 在 GSRF 中的接入位置

入口在 `train_rfid.py`。

当配置中：

```yaml
use_tgsrf_initializer: true
```

训练开始时会调用：

```python
run_tgsrf_init_warmup(
    scene,
    gaussians,
    model_para_args,
    optimization_para_args,
    pipeline_para_args,
    render,
)
```

这一步发生在：

```python
gaussians.training_setup(...)
```

之前。

这意味着：

```text
V2 是初始化模块，不是替代 renderer 的模块。
```

warm-up 阶段，V2 会临时生成一组 Gaussian 参数，并通过 `render_rfid` 的 override 参数渲染：

```python
override_xyz
override_attenuation
override_scaling
override_features
override_rotation
```

warm-up 结束后，V2 把预测出来的参数写回：

```python
gaussians.apply_initial_xyz_attenuation(...)
```

然后进入原始 GSRF training。

inference 阶段不会再运行 V2 网络，只加载已经保存的 Gaussian checkpoint。

## 4. V2 的两种模式

V2 支持：

```yaml
tgsrf_mode: "refine"    # 当前 dense 30K 实验使用
tgsrf_mode: "generate"  # 已实现，但不是当前稳定主实验
```

### 4.1 refine mode

当前 dense 30K V2 实验使用：

```yaml
tgsrf_mode: "refine"
```

在 refine mode 中：

```text
seed_xyz = 原始 GSRF 生成的 Gaussian 点
num_points = 原始 Gaussian 数量
```

也就是说，当前 V2 没有抛弃 baseline 点云，而是在 baseline 点云上预测局部 residual。

优点：

```text
稳定
公平对比 baseline
不会因为新点云初始化太差导致训练崩掉
```

缺点：

```text
仍然受原始 GSRF 点云布局限制
如果 baseline dense 已经很强，V2 的提升空间有限
```

### 4.2 generate mode

generate mode 中，V2 不直接使用 baseline Gaussian points，而是通过 `build_rf_ray_seeds` 生成非网格 anchors。

它混合三类点：

```text
1. angular spectrum 高能方向上的 ray samples
2. TX-RX path 附近的 jitter samples
3. Sobol/random 体采样点
```

这个设计更接近 MVSplat / pixelSplat 的 candidate initialization 思路，但当前实验中 generate mode 不够稳定，所以 dense 30K V2 主实验没有用它。

## 5. 模块一：RFSpectrumEncoder

文件：

```text
scene/tgsrf/rf_spectrum_encoder.py
```

类：

```python
RFSpectrumEncoder
```

作用：

```text
把 RFID spatial spectrum 编码成 token，用于条件化点特征和 Gaussian decoder。
```

输入：

```text
spectrum: [90, 360]
TX: [3]
RX: [3]
```

它不会只把 spectrum 当灰度图，而是额外拼接角度位置编码：

```text
spectrum
sin(azimuth)
cos(azimuth)
sin(elevation)
cos(elevation)
```

所以 CNN 输入通道数是 5。

网络结构：

```text
Conv2d(5 -> hidden_dim/2)
SiLU
Conv2d(hidden_dim/2 -> hidden_dim)
SiLU
Conv2d(hidden_dim -> hidden_dim, kernel=patch_stride, stride=patch_stride)
SiLU
```

当前默认：

```yaml
tgsrf_hidden_dim: 128
tgsrf_spectrum_patch_stride: 8
```

对于 RFID spectrum：

```text
90 x 360 -> 约 11 x 45 tokens
```

同时，TX/RX 会被 Fourier 编码：

```text
concat(TX, RX) -> fourier_encode -> MLP -> txrx_emb
```

然后把 `txrx_emb` 加到每个 spectrum token 上。

输出：

```text
tokens:      [1, N_tokens, hidden_dim]
global:      [1, hidden_dim]
feature_map: [1, hidden_dim, H_patch, W_patch]
```

三类输出的作用：

```text
tokens: 给 point decoder 做 cross-attention
global: 给 Gaussian decoder 作为全局 RF 条件
feature_map: 根据 Gaussian 方向查 local angular feature
```

重要限制：

当前 warm-up 直接使用训练样本的 ground-truth spectrum 作为条件输入。warm-up 结束后只保留静态 Gaussian 参数，inference 不再运行 spectrum encoder。因此当前 V2 不是 test-time feed-forward model，而是一个 training-time initializer。

## 6. 模块二：RFGeometryEncoder

文件：

```text
scene/tgsrf/rf_geometry_encoder.py
```

类：

```python
RFGeometryEncoder
```

作用：

```text
为每个 candidate Gaussian 构造 RF propagation-aware geometry feature。
```

输入：

```text
xyz: [N, 3]
TX:  [3]
RX:  [3]
```

原始几何特征包括：

```text
scaled xyz
point -> RX 的单位方向
point -> TX 的单位方向
RX -> point 的单位方向
TX -> point 的单位方向
d_rx / scene_scale
d_tx / scene_scale
(d_rx + d_tx) / scene_scale
sin(phase)
cos(phase)
```

其中：

```text
phase = 2*pi*(d_rx + d_tx) / wavelength
wavelength = c / frequency
frequency = 915 MHz
```

然后做 Fourier encoding：

```text
raw_features -> fourier_encode(raw_features, tgsrf_geometry_freqs)
```

再经过 MLP：

```text
Linear -> SiLU -> Linear -> LayerNorm
```

输出：

```text
geometry_feature: [N, hidden_dim]
```

这个模块是 V2 中最明确的 RF 物理几何部分。它让网络知道每个 Gaussian 和 TX/RX 的路径长度、方向和相位 proxy。

## 7. 模块三：RFPointDecoder

文件：

```text
scene/tgsrf/rf_point_decoder.py
```

类：

```python
RFPointDecoder
CrossAttentionBlock
```

作用：

```text
让 candidate Gaussian anchors 通过 cross-attention 读取 spectrum tokens，形成 point-level RF feature。
```

每个点的输入包括：

```text
normalized xyz
learnable point latent
RF geometry feature
local spectrum feature
```

local spectrum feature 来自：

```python
point_to_local_spectrum(feature_map, xyz, origin)
```

它先计算：

```text
direction = normalize(xyz - origin)
```

再把方向映射到 azimuth/elevation grid，从 spectrum feature map 中 bilinear sample。

当前默认：

```yaml
tgsrf_ray_origin: "tx"
```

所以 local spectrum feature 是从 TX 指向 Gaussian 的角度方向上查到的。

point decoder 内部：

```text
point query -> MultiheadAttention(query, spectrum_tokens)
            -> FFN
            -> point_features
```

输出：

```text
point_features: [N, hidden_dim]
point_delta:    [N, 3]
point_xyz:      seed_xyz + point_delta
```

point offset 是 bounded residual：

```text
point_delta = tanh(delta_head(point_features)) * tgsrf_point_offset_radius
```

但当前 dense V2 实验设置：

```bash
--tgsrf_point_refine_start 999999
```

所以：

```text
point_delta 被关闭
point_xyz = seed_xyz
```

也就是说，当前 dense V2 中 point decoder 主要作为 feature extractor，而不是移动点云的模块。

## 8. 模块四：RFTriplaneField

文件：

```text
scene/tgsrf/rf_triplane_field.py
```

类：

```python
RFTriplaneField
```

作用：

```text
提供三平面空间先验，让每个 Gaussian 根据 xyz 查询局部空间特征。
```

空间范围来自当前 Gaussian 点云的 3D bounding box：

```python
xyz_min, xyz_max = scene_bounds_from_gaussians(gaussians, tgsrf_bbox_pad_scale)
```

当前默认：

```yaml
tgsrf_bbox_pad_scale: 0.02
```

表示在点云包围盒每个轴向两侧扩展 2% span，避免边界点查询时过于贴边。

查询流程：

```text
xyz -> normalize to [-1, 1]^3
sample P_xy with (x, y)
sample P_xz with (x, z)
sample P_yz with (y, z)
concat(f_xy, f_xz, f_yz)
```

当前默认：

```yaml
tgsrf_triplane_resolution: 128
tgsrf_triplane_channels: 32
```

所以输出 triplane feature 维度是：

```text
32 * 3 = 96
```

### 8.1 direct triplane

当：

```yaml
tgsrf_triplane_type: "direct"
```

直接优化三张 feature plane：

```text
planes: [3, C, R, R]
```

### 8.2 Haar wavelet triplane

当：

```yaml
tgsrf_triplane_type: "haar"
```

不直接优化完整 plane，而是优化：

```text
LL coarse coefficients
LH / HL / HH high-frequency coefficients
```

再通过 inverse Haar wavelet transform 重建完整 planes。

当前默认：

```yaml
tgsrf_wavelet_levels: 2
tgsrf_wavelet_high_init: 0.0
```

所以高频系数初始为 0。

如果启用：

```yaml
tgsrf_wavelet_l1: 0.001
```

会对高频 coefficients 加 L1 稀疏约束。

如果启用：

```yaml
tgsrf_wavelet_c2f: true
```

warm-up 过程中会 coarse-to-fine 逐步打开 high-frequency levels。

### 8.3 bior44 注意事项

代码里接受：

```text
haar
bior44
bior4.4
```

但当前 `bior44` 仍然走稳定 Haar IWT 实现。代码注释也写了：

```text
V1 uses stable Haar reconstruction for both Haar and bior44 switches.
```

所以目前不能在论文中声称已经完整实现 bior4.4 filter-bank wavelet triplane。

## 9. 模块五：RFGeometryAwareTriplaneEncoder

文件：

```text
scene/tgsrf/rf_geometry_aware_triplane.py
```

类：

```python
RFGeometryAwareTriplaneEncoder
```

作用：

```text
把 point decoder 得到的 point_features scatter 回三张 plane，
生成一个和当前 RF 条件相关的 extra triplane。
```

流程：

```text
point_features -> MLP -> feat [N, C]
xyz_norm -> 对应 xy/xz/yz plane index
index_add scatter 到三个 plane
extra_planes = scale * stacked_planes
```

然后 query triplane 时：

```text
planes = learned_planes + extra_planes
```

因此 V2 的 triplane 不是完全静态的。它由两部分组成：

```text
learned triplane: 全局/场景空间先验
geometry-aware extra triplane: 当前 spectrum + TX/RX 条件下的点特征投影
```

这个模块对应 TGS 图里“Geometry-aware Encoding / Hybrid Triplane-Gaussian”的思想。

限制：

scatter 用的是 round 后的离散 index，坐标方向不可微。当前 dense V2 关闭 point offset，所以这个问题暂时不严重。

## 10. 模块六：RFGaussianDecoder

文件：

```text
scene/tgsrf/rf_gaussian_decoder.py
```

类：

```python
RFGaussianDecoder
```

作用：

```text
把 point feature、triplane feature、geometry feature、local/global spectrum feature 融合，
预测 Gaussian 参数 residual。
```

输入拼接：

```text
point_features
triplane_features
geometry_features
local_spectrum_features
global_spectrum_token
```

然后进入 shared MLP：

```text
Linear -> SiLU -> Linear -> SiLU
```

最后三个 head：

```text
xyz_head
att_head
scale_head
```

输出：

```text
delta_xyz
attenuation_delta
log_scale_delta
```

所有 residual 都是 bounded：

```text
delta_xyz = tanh(xyz_head(h)) * tgsrf_gaussian_offset_radius
attenuation_delta = tanh(att_head(h)) * tgsrf_att_radius
log_scale_delta = tanh(scale_head(h)) * tgsrf_scale_radius
```

当前 dense V2 使用：

```text
tgsrf_gaussian_offset_radius = 0.05
tgsrf_att_radius = 0.05
tgsrf_scale_radius = 0.10
```

最终参数：

```text
final_xyz = point_xyz + delta_xyz
attenuation_raw = base_attenuation_raw + attenuation_delta
scaling_raw = base_scaling_raw + log_scale_delta
```

然后：

```text
attenuation = sigmoid(attenuation_raw)
scale = exp(scaling_raw)
```

并且做 clamp：

```text
attenuation_raw: [-8, 2]
scaling_raw: [-6, 0]
final_xyz: 限制在 bbox 内
```

这个模块比旧 gate 更强，因为它不仅改 attenuation，也改 xyz 和 scale。

但它还有一个明显不足：

```text
不同物理量仍然共享一个 MLP trunk，只是输出 head 不同。
```

这可能不够合理。后续 V3 可以考虑：

```text
geometry decoder -> xyz
support decoder -> scale
power decoder -> attenuation
directional decoder -> FLE features
```

## 11. 当前 V2 实际改变了哪些 Gaussian 参数

当前 V2 可以改变：

```text
Gaussian xyz center
Gaussian attenuation
Gaussian scale
```

当前 V2 没有真正改变：

```text
rotation
FLE feature
candidate keep probability
top-K pruning
densification policy
renderer formulation
```

### 11.1 xyz

当前 dense V2：

```text
point offset disabled
Gaussian decoder offset enabled
```

也就是说：

```text
xyz 的变化来自 Gaussian decoder 的 delta_xyz
```

不是 point decoder 的 point_delta。

### 11.2 attenuation

V2 在 raw/logit 空间预测 residual：

```text
attenuation_raw_new = attenuation_raw_base + attenuation_delta
```

再过 sigmoid：

```text
attenuation = sigmoid(attenuation_raw_new)
```

### 11.3 scale

V2 在 log-scale 空间预测 residual：

```text
scaling_raw_new = scaling_raw_base + log_scale_delta
```

再过 exp：

```text
scale = exp(scaling_raw_new)
```

### 11.4 rotation

refine mode 中 rotation 直接复制 baseline：

```text
rotation_raw = gaussians._rotation
```

当前没有 rotation residual head。

### 11.5 FLE features

refine mode 中默认：

```yaml
tgsrf_refine_features: false
```

所以 warm-up 阶段 FLE features 不训练。materialize 后，后续标准 GSRF training 会继续训练 FLE features。

## 12. Warm-up loss

warm-up loss 位于：

```text
scene/tgsrf/initializer.py
```

重建损失沿用 GSRF：

```text
L_recon =
    (1 - lambda_dssim - lambda_dfourier) * L1
    + lambda_dssim * SSIM_loss
    + lambda_dfourier * Fourier_loss
```

再加 V2 regularization：

```text
L =
    L_recon
    + lambda_point * ||point_delta||^2
    + lambda_gauss * ||gaussian_delta||^2
    + lambda_att * ||attenuation_delta||^2
    + lambda_scale * ||log_scale_delta||^2
    + lambda_wavelet * wavelet_l1
```

当前配置：

```yaml
tgsrf_point_offset_l2: 0.02
tgsrf_gaussian_offset_l2: 0.02
tgsrf_att_l2: 0.02
tgsrf_scale_l2: 0.01
tgsrf_wavelet_l1: 0.0       # direct
tgsrf_wavelet_l1: 0.001     # Haar 实验
tgsrf_grad_clip: 1.0
```

如果 loss 非有限：

```text
skip this warm-up step
```

warm-up 后会把：

```text
base_attenuation_raw clamp 到 [-8, 2]
base_scaling_raw clamp 到 [-6, 0]
FLE features 中的 nan/inf 清零
```

## 13. Materialization 机制

warm-up 结束后，V2 不是直接用最后一个 viewpoint 的预测结果，而是：

```text
从 train views 中选 tgsrf_finalize_views 个样本
对每个样本预测一组 Gaussian 参数
对 xyz / attenuation_raw / scaling_raw 取平均
写回 GaussianModel
```

当前：

```yaml
tgsrf_finalize_views: 16
```

优点：

```text
得到一个稳定的静态 Gaussian initialization
不需要 inference 时再运行 V2 网络
保持 GSRF 原始训练流程
```

缺点：

```text
view-conditioned 的 spectrum/TX/RX 信息被平均掉
```

这可能是当前 V2 dense 提升不明显的重要原因。

## 14. 当前 dense 30K 实验启用的具体配置

### 14.1 V2 direct dense 30K

命令入口：

```bash
GPU=0 V2_WARMUP=2000 ./run_rfid_triplane_experiments.sh tgsrf_v2_direct_dense30k
```

关键配置：

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

含义：

```text
使用 baseline Gaussian 点
不启用 point-level anchor movement
warm-up 前 1000 step 只训练特征/attenuation/scale 等可用分支
warm-up 后 1000 step 打开 Gaussian xyz residual
使用 direct triplane
```

### 14.2 V2 Haar dense 30K

命令入口：

```bash
GPU=1 V2_WARMUP=2000 ./run_rfid_triplane_experiments.sh tgsrf_v2_haar_dense30k
```

额外配置：

```text
tgsrf_triplane_type = haar
tgsrf_wavelet_l1 = 0.001
tgsrf_wavelet_c2f = true
```

含义：

```text
triplane 参数化改成 Haar wavelet coefficients
高频 coefficients 加 L1 稀疏约束
warm-up 中逐步打开高频 levels
```

## 15. 当前 dense 30K 结果

正式结果必须看 `inference_rfid.py` 的 summary，不能看 `training_report` 的 5-sample 随机评估。

| Method | Iter | PSNR | MSE | SSIM |
|---|---:|---:|---:|---:|
| baseline dense | 30K | 22.0571 | 0.009428 | 0.8165 |
| old triplane gate dense | 30K | 21.9720 | 0.009467 | 0.8148 |
| old bior44 dense | 30K | 21.9392 | 0.009644 | 0.8144 |
| V2 direct dense | 30K | 21.9376 | 0.009458 | 0.8128 |
| V2 Haar dense | 30K | 21.9296 | 0.009542 | 0.8128 |
| V2 direct dense | 20K | 21.3880 | 0.010404 | 0.7985 |
| V2 Haar dense | 20K | 21.4003 | 0.010498 | 0.7976 |

结论：

```text
V2 dense 没有超过 dense baseline。
Haar 版本也没有超过 direct 版本。
dense 80/20 目前不适合作为主故事。
```

训练日志中曾经出现 20K `25+dB`，但原因是：

```text
utils/train_utils.py 的 training_report 只随机抽 5 个 test samples。
```

正式 inference 是 1225 个 test samples 的全集均值。

## 16. V2 已经实现的研究点

当前 V2 已经实现：

```text
1. NeRF2-style RF geometry features
2. spectrum encoder
3. candidate/anchor point representation
4. point-to-spectrum local angular feature lookup
5. point cross-attention decoder
6. TGS-style geometry-aware triplane
7. direct triplane / Haar wavelet triplane switch
8. wavelet high-frequency L1
9. wavelet coarse-to-fine activation
10. multi-parameter Gaussian decoder
11. warm-up then materialize into GaussianModel
12. standard GSRF training/inference compatibility
```

所以 V2 不是小改动，已经是一个完整 initializer 框架。

## 17. V2 还没有实现的关键点

当前还没做：

```text
1. candidate-wise keep probability
2. top-K pruning
3. cost-volume-style RF candidate score
4. 单独的 xyz / scale / attenuation / FLE decoder
5. rotation residual
6. FLE residual
7. persistent conditioning during main training
8. test-time feed-forward Gaussian prediction
9. 真正的 bior4.4 filter bank
10. final spectrum wavelet subband error analysis
```

其中最关键的是：

```text
V2 还没有学会选点，只是在微调所有已有点。
```

这也是它和 MVSplat / pixelSplat 最重要的差距。

## 18. 当前 V2 的主要问题

### 18.1 dense baseline 太强

dense 80/20 下，baseline 已经达到约 22.06 dB。30K 标准优化会把很多初始化差异抹平。

### 18.2 条件信息被 materialization 平均掉

V2 warm-up 是 view-conditioned：

```text
spectrum + TX/RX -> Gaussian 参数
```

但最终变成：

```text
16 个 train views 的预测均值 -> 一套静态 Gaussian
```

所以 spectrum encoder 和 TX/RX conditioning 的能力没有完整保留下来。

### 18.3 没有 candidate selection

当前每个 baseline Gaussian 都保留，只是做 residual。

这和 MVSplat / pixelSplat 的关键思想还有差距：

```text
candidate score
probability
keep probability
top-K
opacity as probability
```

### 18.4 物理参数 decoder 仍然耦合

当前是：

```text
shared trunk -> xyz_head / att_head / scale_head
```

但 RF 中这些量物理意义不同：

```text
xyz: 空间几何位置
scale: Gaussian 支持范围
attenuation: RF 强度/贡献
FLE: 方向复响应
rotation: 各向异性空间方向
```

后续应该拆成 parameter-specific decoders。

## 19. 下一步建议

### 19.1 不再把 dense 80/20 当主故事

当前 dense 结果证明：

```text
dense sample 上 V2 没有稳定超过 baseline。
```

接下来应该重点看：

```text
20% train
sparse220
更少观测下的 data efficiency
```

### 19.2 加 candidate-wise keep probability

最值得做的下一步：

```text
Gaussian candidate feature -> keep_logit
keep_prob = sigmoid(keep_logit)
```

用法可以分两级：

```text
safe: keep_prob 初始化 attenuation / opacity
strong: top-K pruning 或 threshold pruning
```

这样方法才从：

```text
调参数
```

升级到：

```text
学会选点
```

### 19.3 拆分 Gaussian decoder

建议改成：

```text
GeometryDecoder: xyz residual
SupportDecoder: scale residual
PowerDecoder: attenuation residual
DirectionalDecoder: FLE residual
```

这样更容易和论文中的物理解释对应。

### 19.4 重新考虑 persistent modulation

如果 spectrum/TX/RX conditioning 是核心，那么 warm-up 后完全丢掉 V2 网络可能太弱。

可以考虑：

```text
warm-up materialize + 前 K iterations persistent conditional residual
```

但这会增加训练复杂度，需要谨慎做消融。

## 20. 一段可用于论文/汇报的总结

当前 TGS-RF V2 是一个 warm-up 型 RF-aware Gaussian initializer。它在原始 GSRF 优化之前，使用训练样本的 RF spatial spectrum、TX/RX 几何和 candidate Gaussian anchors 构造条件特征；通过 spectrum encoder 提取角度域 token，通过 RF geometry encoder 显式编码路径长度、方向和相位 proxy，通过 point decoder 让 Gaussian anchors cross-attend 到 spectrum tokens，再结合 learned/wavelet triplane 和 geometry-aware extra triplane 查询空间特征，最后由 Gaussian decoder 预测 bounded xyz、attenuation 和 scale residual。warm-up 阶段这些参数通过 `render_rfid` override 进入原始 RF Gaussian renderer，并用 GSRF 的 L1/SSIM/Fourier loss 反向传播；warm-up 结束后，多个 train views 的预测结果被平均并写回 `GaussianModel`，随后继续标准 GSRF training。当前 V2 已经实现了比旧 triplane gate 更完整的多模块初始化框架，但 dense 30K 结果没有超过 baseline，说明下一版应重点转向 sparse/data-efficiency 场景，并补上 candidate-wise keep probability、top-K pruning 和参数解耦 decoder。
