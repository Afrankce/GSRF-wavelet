# `scene/triplane_initializer.py` 详细说明与实验思路

更新时间：2026-05-22  
远端文件：`/home/wys/GSRF/scene/triplane_initializer.py`  
相关配置：`/home/wys/GSRF/arguments/configs/rfid/exp_triplane.yaml`

这份文档专门解释 `scene/triplane_initializer.py` 中每个类、函数、关键超参数，以及它们和目前已经完成实验之间的关系。核心目的是把代码逻辑、实验现象和下一步论文叙事连起来。

---

## 1. 这个文件解决的问题

原始 GSRF 的训练流程是：先根据初始点云创建一批 Gaussian，然后主训练直接优化 Gaussian 的位置、attenuation、FLE features、scale、rotation 等参数。

`triplane_initializer.py` 加入了一个训练前 warmup 阶段：

```text
初始 Gaussian
  -> TriPlane / Wavelet-TriPlane warmup
  -> 预测 Gaussian 初始位置偏移和 attenuation delta
  -> 将预测结果 materialize 到 Gaussian
  -> 进入正常 GSRF 主训练
```

它不是主训练里的持续模块，而是一个初始化模块。warmup 结束后，TriPlane 本身不再参与后续训练，只有它产生的 `xyz` 和 `attenuation` 初始化被写入 Gaussian。

在 `train_rfid.py` 中调用位置是：

```python
if not checkpoint:
    run_triplane_init_warmup(...)
```

因此它只在从头训练时生效，从 checkpoint resume 时不会再跑初始化。

---

## 2. 总体数据流

### 2.1 输入

主要输入来自三个地方：

1. `scene`
   - 提供 train spectrums。
   - 每个 viewpoint 有 `spectrum`、`T_tx`、`T_rx`。

2. `gaussians`
   - 提供当前 Gaussian 坐标 `gaussians.get_xyz`。
   - 提供 raw attenuation `gaussians._attenuation`。
   - 提供写入函数 `apply_initial_xyz_attenuation(...)`。

3. `model_args / optim_args / pipe_args`
   - 从 yaml 和命令行合并而来。
   - 控制 TriPlane 分辨率、warmup 步数、decoder、regularization、wavelet 等。

### 2.2 Warmup 阶段

每个 warmup step：

```text
随机取一个 train viewpoint
  -> 用 Gaussian 坐标查询 TriPlane feature
  -> 拼接 RF geometry feature
  -> decoder 输出 4 维预测
       前 3 维：xyz delta
       第 4 维：attenuation delta
  -> 用 override_xyz / override_attenuation 渲染 spectrum
  -> 和 GT spectrum 计算 L1 + SSIM + Fourier loss
  -> 加 offset / attenuation / wavelet regularization
  -> 只优化 initializer，默认不直接优化 Gaussian 主参数
```

### 2.3 Finalize 阶段

warmup 结束后：

```text
把 wavelet high-frequency levels 全部激活
  -> 对多个 TX view 做预测平均
  -> 得到 final_xyz 和 final_attenuation_raw
  -> gaussians.apply_initial_xyz_attenuation(...)
```

之后进入正常 GSRF 训练。

---

## 3. 文件中的类和函数总览

当前文件主要包含：

```text
TriPlaneField
HaarWaveletTriPlaneField
FourierGeometryEncoding
RFMLPDecoder
FiLMResidualBlock
RFFiLMFourierDecoder
RFTriplaneInitializer
_scene_context
_make_warmup_optimizer
_finalize_with_tx_average
run_triplane_init_warmup
```

可以把它们分成四层：

| 层级 | 组件 | 作用 |
|---|---|---|
| Spatial field | `TriPlaneField`, `HaarWaveletTriPlaneField` | 给 3D Gaussian 中心提供 plane feature |
| Geometry encoder | `FourierGeometryEncoding` | 编码 RF 几何关系 |
| Decoder | `RFMLPDecoder`, `RFFiLMFourierDecoder`, `FiLMResidualBlock` | 从 plane feature + geometry feature 预测初始化 delta |
| Warmup driver | `RFTriplaneInitializer`, `_scene_context`, `_make_warmup_optimizer`, `_finalize_with_tx_average`, `run_triplane_init_warmup` | 组织 warmup 训练并写回 Gaussian |

---

## 4. `TriPlaneField`

代码位置：`scene/triplane_initializer.py:10`

### 4.1 作用

`TriPlaneField` 是 direct triplane 表示。它维护三张 learnable feature plane：

```text
P_xy
P_xz
P_yz
```

对于每个 Gaussian center `(x, y, z)`：

1. 在 `P_xy` 上用 `(x, y)` 采样。
2. 在 `P_xz` 上用 `(x, z)` 采样。
3. 在 `P_yz` 上用 `(y, z)` 采样。
4. 三个 feature 拼接成一个长度为 `3 * channels` 的特征。

### 4.2 `__init__(xyz_min, xyz_max, channels=16, resolution=64)`

初始化内容：

```python
self.channels = int(channels)
self.resolution = int(resolution)
self.xyz_min = xyz_min
self.xyz_max = xyz_max
self.planes = nn.Parameter(
    0.01 * torch.randn(3, channels, resolution, resolution)
)
```

含义：

| 参数 | 含义 | 实验影响 |
|---|---|---|
| `xyz_min` | Gaussian bbox 最小坐标 | 决定物理空间如何映射到 plane |
| `xyz_max` | Gaussian bbox 最大坐标 | 同上 |
| `channels` | 每张 plane 的 feature channel 数 | 增大后 decoder 输入更宽，容量更强，但更容易过拟合 warmup |
| `resolution` | 每张 plane 的空间分辨率 | `64` 是当前较稳定值，`128` 实验变差 |

当前 direct gate 最好的一组是：

```bash
--triplane_resolution 64
--triplane_channels 16
```

### 4.3 `normalize_xyz(xyz)`

代码逻辑：

```python
denom = (xyz_max - xyz_min).clamp(min=1e-6)
xyz01 = (xyz - xyz_min) / denom
return xyz01.clamp(0.0, 1.0) * 2.0 - 1.0
```

作用：

```text
真实世界坐标 -> [0, 1] -> [-1, 1]
```

因为 PyTorch `grid_sample` 需要 `[-1, 1]` 坐标。

非常关键的一点：

```python
xyz01.clamp(0.0, 1.0)
```

如果某个坐标落在 bbox 外，会被夹到边界。这意味着：

1. bbox 太紧时，边缘点或 TX/RX 可能被压到 plane 边界。
2. 很多 bbox 外坐标会共享边界 feature。
3. 这可能伤害 RF geometry 表达。

这就是后来加入：

```bash
--triplane_bbox_pad_scale
```

的原因。

### 4.4 `_sample_plane(plane, coords)`

作用：在一张 2D plane 上双线性采样 feature。

关键设置：

```python
F.grid_sample(
    plane.unsqueeze(0),
    grid,
    mode="bilinear",
    padding_mode="border",
    align_corners=True,
)
```

含义：

| 设置 | 含义 | 影响 |
|---|---|---|
| `mode="bilinear"` | 双线性插值 | feature 随坐标连续变化 |
| `padding_mode="border"` | 超出边界时用边界值 | 和 `clamp` 一起导致边界 feature 复用 |
| `align_corners=True` | `-1` 和 `1` 对齐最外侧像素中心 | 让 bbox 两端严格映射到 plane 边缘 |

实验相关：

`bbox_pad_scale=0.05` 就是为了观察 plane 覆盖范围稍微放大后，边界复用问题是否缓解。结果：

```text
res64 direct gate: 21.9598
res64 pad005:      21.9347
```

说明这个因素有影响，但不是当前主突破口。

### 4.5 `forward(xyz)`

流程：

```text
xyz -> normalize_xyz
    -> xy / xz / yz
    -> sample P_xy / P_xz / P_yz
    -> concat
```

输出：

```python
return torch.cat([f_xy, f_xz, f_yz], dim=-1), xyz_norm
```

其中：

| 输出 | shape | 用途 |
|---|---|---|
| `tri_feat` | `[N, 3 * channels]` | 输入 decoder |
| `xyz_norm` | `[N, 3]` | 输入 RF geometry feature |

---

## 5. `HaarWaveletTriPlaneField`

代码位置：`scene/triplane_initializer.py:58`

### 5.1 作用

这是参考 TriNeRFLet 思路加入的 wavelet triplane。

它不是直接优化完整 feature plane，而是优化：

```text
LL 低频系数
LH/HL/HH 高频系数
```

然后通过 Haar inverse wavelet transform 重建三张 plane。

RF 直觉：

| wavelet 子带 | RF 含义 |
|---|---|
| `LL` | 大尺度 path loss / smooth propagation |
| `LH/HL/HH` | 局部 multipath / small-scale fading |

### 5.2 `__init__(..., levels=2, high_init=0.0)`

关键逻辑：

```python
coarse_resolution = resolution // (2 ** levels)
self.ll = nn.Parameter(...)
self.highs = nn.ParameterList()
```

例如：

```text
resolution = 64
levels = 2
coarse_resolution = 16
```

表示：

```text
LL: 16x16
level 0 high: 16x16
level 1 high: 32x32
IWT 后恢复到 64x64
```

约束：

```python
if resolution % (2 ** levels) != 0:
    raise ValueError
```

所以 `resolution` 必须能被 `2^levels` 整除。

### 5.3 `set_active_levels(active_levels=None)`

作用：控制 coarse-to-fine 训练中激活几个高频层。

例如 `levels=2`：

| `active_levels` | 实际使用 |
|---:|---|
| 0 | 只用 LL，高频全置零 |
| 1 | 用 level 0 高频 |
| 2 | 用全部高频 |

实验背景：

最开始 c2f schedule 有问题，高频层激活太晚，Haar 10K 很差。修复后 10K 回升，但 30K 仍然不如 baseline。

### 5.4 `_haar_iwt_step(ll, high)`

作用：单层 Haar inverse wavelet transform。

输入：

```text
ll:   低频
high: 高频，其中 high[:, :, 0] = LH, high[:, :, 1] = HL, high[:, :, 2] = HH
```

输出分辨率扩大 2 倍。

公式：

```python
out[..., 0::2, 0::2] = 0.5 * (ll + lh + hl + hh)
out[..., 0::2, 1::2] = 0.5 * (ll - lh + hl - hh)
out[..., 1::2, 0::2] = 0.5 * (ll + lh - hl - hh)
out[..., 1::2, 1::2] = 0.5 * (ll - lh - hl + hh)
```

这里实现的是 Haar IWT，把低频和三个方向的高频重建为空间 plane。

### 5.5 `materialize_planes()`

作用：从 wavelet coefficients 生成实际可采样的三张 plane。

逻辑：

```text
planes = LL
for each high level:
    if level active:
        use high
    else:
        use zeros_like(high)
    planes = IWT(planes, high)
```

所以 c2f 并不是冻结参数，而是在 forward 时把未激活的 high band 当 0。

### 5.6 `wavelet_l1_loss()`

作用：对所有 high bands 做 L1 稀疏约束。

```python
return mean(abs(high))
```

对应超参数：

```bash
--triplane_wavelet_l1
```

实验结果：

```text
triplane_gate_haar_plain_default_10k:  19.6325
triplane_gate_haar_c2ffix_default_10k: 19.5650
```

当前 Haar + L1 + c2f 没有带来收益，说明 Haar 参数化本身还不足以构成强 RF prior。

### 5.7 `forward(xyz)`

和 direct triplane 一样，只是采样前先：

```python
planes = self.materialize_planes()
```

然后再从 `xy/xz/yz` 三个方向采样。

---

## 6. `FourierGeometryEncoding`

代码位置：`scene/triplane_initializer.py:148`

### 6.1 作用

把低维 RF geometry feature 做 Fourier positional encoding。

输入是 geometry feature，例如：

```text
xyz_norm, tx_norm, rx_norm, distance, direction
```

输出包含：

```text
x
sin(pi * x * 2^k)
cos(pi * x * 2^k)
```

### 6.2 `__init__(num_frequencies=4, include_input=True)`

关键参数：

| 参数 | 含义 |
|---|---|
| `num_frequencies` | Fourier 频率数量 |
| `include_input` | 是否保留原始输入 |

如果 `num_frequencies=4`，频率是：

```text
1, 2, 4, 8
```

当前较好实验使用：

```bash
--triplane_fourier_freqs 4
```

### 6.3 `out_multiplier`

输出维度倍数：

```python
(1 if include_input else 0) + 2 * num_frequencies
```

如果 `include_input=True` 且 `num_frequencies=4`：

```text
out_multiplier = 1 + 2 * 4 = 9
```

对于 18 维 geometry feature，编码后维度是：

```text
18 * 9 = 162
```

### 6.4 `forward(x)`

输出：

```python
torch.cat([x, sin_features, cos_features], dim=-1)
```

实验意义：

`film_fourier` 明显强于普通 `mlp`：

```text
triplane_gate_default_10k:     19.9350
triplane_gate_mlp_default_10k: 19.4781
```

说明 RF geometry 中的距离、方向、归一化位置确实需要 Fourier-style encoding。

---

## 7. `RFMLPDecoder`

代码位置：`scene/triplane_initializer.py:176`

### 7.1 作用

这是最简单的 decoder，用于 ablation。

输入：

```text
tri_feat + geom_feat
```

输出：

```text
4 维预测
```

其中：

```text
pred[:, :3] -> xyz delta
pred[:, 3]  -> attenuation delta
```

### 7.2 网络结构

```text
Linear(tri_dim + geom_dim, hidden_dim)
ReLU
Linear(hidden_dim, hidden_dim)
ReLU
Linear(hidden_dim, 4)
```

### 7.3 `reset_output()`

最后一层权重和 bias 初始化为 0：

```python
nn.init.zeros_(last.weight)
nn.init.zeros_(last.bias)
```

意义：

```text
warmup 一开始输出 delta=0
不会一开始就破坏原始 Gaussian 初始化
```

### 7.4 实验结论

MLP decoder 的 10K 结果：

```text
triplane_gate_mlp_default_10k: 19.4781
```

明显低于 Fourier-FiLM：

```text
triplane_gate_default_10k: 19.9350
```

所以后续主线不建议继续使用 `mlp`，只保留作 ablation。

---

## 8. `FiLMResidualBlock`

代码位置：`scene/triplane_initializer.py:199`

### 8.1 作用

这是 Fourier-FiLM decoder 里的残差调制模块。它用 RF geometry condition 去调制 hidden feature。

### 8.2 网络结构

```python
self.linear = nn.Linear(hidden_dim, hidden_dim)
self.norm = nn.LayerNorm(hidden_dim)
self.film = nn.Linear(cond_dim, 2 * hidden_dim)
self.act = nn.SiLU(inplace=True)
```

### 8.3 `forward(h, cond)`

流程：

```python
gamma, beta = self.film(cond).chunk(2, dim=-1)
x = self.norm(self.linear(h))
x = x * (1.0 + 0.1 * tanh(gamma)) + 0.1 * tanh(beta)
return h + SiLU(x)
```

解释：

1. `gamma` 是 multiplicative modulation。
2. `beta` 是 additive modulation。
3. `0.1 * tanh(...)` 限制调制幅度，避免 warmup 太激进。
4. 残差连接保证 decoder 比较稳。

实验意义：

FiLM block 让 TX/RX geometry 可以动态调制预测，比普通 concat MLP 更适合 RF 场景。

---

## 9. `RFFiLMFourierDecoder`

代码位置：`scene/triplane_initializer.py:216`

### 9.1 作用

这是当前最推荐的 decoder。

它做三件事：

1. 对 geometry feature 做 Fourier encoding。
2. 把 `tri_feat + encoded geometry` 投影到 hidden。
3. 用原始 geometry feature 作为 condition，通过 FiLM residual blocks 调制 hidden。

### 9.2 `__init__(tri_dim, geom_dim, hidden_dim, num_layers=3, num_frequencies=4)`

关键结构：

```python
self.fourier = FourierGeometryEncoding(...)
self.input_proj = Linear(tri_dim + geom_encoded_dim, hidden_dim)
self.cond_proj = MLP(geom_dim -> hidden_dim)
self.blocks = ModuleList([FiLMResidualBlock(...)] * num_layers)
self.out = Linear(hidden_dim, 4)
```

超参数：

| 参数 | 对应 CLI | 当前推荐值 | 含义 |
|---|---|---:|---|
| `hidden_dim` | `--triplane_hidden_dim` | 128 | decoder hidden 宽度 |
| `num_layers` | `--triplane_film_layers` | 3 | FiLM residual blocks 数量 |
| `num_frequencies` | `--triplane_fourier_freqs` | 4 | Fourier 频率数量 |

### 9.3 `reset_output()`

同样把输出层初始化为 0，使初始预测为 no-op。

### 9.4 `forward(tri_feat, geom_feat)`

流程：

```text
geom_feat -> FourierGeometryEncoding
geom_feat -> cond_proj -> cond
concat(tri_feat, geom_encoded) -> input_proj -> h
h, cond -> FiLMResidualBlock * L
h -> out -> pred[4]
```

### 9.5 实验结论

这一组是目前 10K 最好的设置：

```bash
--triplane_output_mode gating_only
--triplane_decoder_type film_fourier
--triplane_hidden_dim 128
--triplane_fourier_freqs 4
--triplane_film_layers 3
```

结果：

```text
triplane_gate_default_10k: 19.9350
baseline_default_10k:      19.6692
```

说明 Fourier-FiLM + gating-only 对 early convergence 有帮助。

---

## 10. `RFTriplaneInitializer`

代码位置：`scene/triplane_initializer.py:253`

这是整个初始化模型的主体。

### 10.1 `__init__(...)`

输入参数：

```python
xyz_min, xyz_max,
channels,
resolution,
hidden_dim,
offset_radius,
attenuation_radius,
decoder_type,
fourier_frequencies,
film_layers,
output_mode,
field_type,
wavelet_levels,
wavelet_high_init
```

它负责：

1. 根据 `field_type` 创建 direct triplane 或 Haar wavelet triplane。
2. 根据 `decoder_type` 创建 MLP 或 Fourier-FiLM decoder。
3. 根据 `output_mode` 决定是否允许 xyz offset。

### 10.2 `field_type`

代码：

```python
if field_type in {"direct", "triplane"}:
    self.triplane = TriPlaneField(...)
elif field_type in {"haar", "wavelet_haar"}:
    self.triplane = HaarWaveletTriPlaneField(...)
```

CLI：

```bash
--triplane_field_type direct
--triplane_field_type haar
```

实验结论：

| field type | 代表实验 | 结论 |
|---|---|---|
| `direct` | `triplane_gate_default_10k`, `triplane_gate_default_30k` | 10K 有收益，30K 不超过 baseline |
| `haar` | `triplane_gate_haar_default_30k`, `triplane_gate_haar_c2ffix_default_30k` | 当前没有超过 direct/baseline |

### 10.3 `decoder_type`

代码支持：

```text
mlp
film_fourier
mlp_no_triplane
film_fourier_no_triplane
```

其中 `_no_triplane` 是通过这一句实现的：

```python
self.use_plane_features = not self.decoder_type.endswith("_no_triplane")
base_decoder_type = self.decoder_type.removesuffix("_no_triplane")
```

也就是说：

```bash
--triplane_decoder_type film_fourier_no_triplane
```

会保留 geometry feature，但不给 decoder 输入 triplane plane feature。

这组实验的作用是判断：

```text
收益到底来自 plane feature，还是来自 geometry-conditioned gating 本身？
```

结果：

```text
triplane_gate_default_30k:          21.9598
triplane_gate_notriplane_default_30k: 21.9593
```

结论：

```text
30K 最终几乎一样，说明当前 plane feature 贡献很弱。
```

### 10.4 `output_mode`

支持：

```text
offset_att
gating_only
```

代码：

```python
if output_mode == "gating_only":
    delta_xyz = torch.zeros_like(base_xyz)
else:
    delta_xyz = offset_radius * tanh(pred[:, :3])

attenuation_delta = attenuation_radius * tanh(pred[:, 3:4])
```

含义：

| output mode | 作用 |
|---|---|
| `offset_att` | 同时预测 Gaussian 位置偏移和 attenuation delta |
| `gating_only` | 不移动 Gaussian，只预测 attenuation delta |

实验结论：

早期 naive `offset_att` 容易破坏几何结构：

```text
baseline_default_7k:  19.0910
triplane_default_7k:  18.4649
```

改成 `gating_only` 后，10K 有明显改善：

```text
baseline_default_10k:      19.6692
triplane_gate_default_10k: 19.9350
```

所以当前主线应该以 `gating_only` 为中心，而不是继续让 initializer 移动 Gaussian。

### 10.5 `_geometry_features(xyz, xyz_norm, tx_context, rx_context)`

这是 RF 相关信息的核心。

输出维度固定为 18：

```text
xyz_norm: 3
tx_norm:  3
rx_norm:  3
d_tx:     1
d_rx:     1
d_total:  1
dir_tx:   3
dir_rx:   3
总计:     18
```

具体含义：

| feature | 维度 | RF 含义 |
|---|---:|---|
| `xyz_norm` | 3 | Gaussian 在归一化场景中的位置 |
| `tx_norm` | 3 | 发射端位置 |
| `rx_norm` | 3 | 接收端位置 |
| `d_tx / scene_scale` | 1 | TX 到 Gaussian 的距离 |
| `d_rx / scene_scale` | 1 | Gaussian 到 RX 的距离 |
| `d_total / scene_scale` | 1 | 路径总长度近似 |
| `dir_tx` | 3 | TX -> Gaussian 的方向 |
| `dir_rx` | 3 | Gaussian -> RX 的方向 |

这套 feature 体现的是：

```text
RF field 不只是空间坐标函数，还依赖 TX/RX 几何关系。
```

这也是为什么 `film_fourier_no_triplane` 可以做到和 direct triplane 接近：geometry feature 本身已经很强。

### 10.6 `forward(base_xyz, base_attenuation_raw, tx_context, rx_context)`

流程：

```text
base_xyz
  -> triplane sampling 或 empty tri_feat
  -> geometry feature
  -> decoder
  -> bounded delta_xyz / attenuation_delta
  -> output initialized xyz and raw attenuation
```

边界约束：

```python
delta_xyz = offset_radius * tanh(...)
attenuation_delta = attenuation_radius * tanh(...)
```

意义：

1. `tanh` 把输出限制在 `[-1, 1]`。
2. `offset_radius` 控制最大位置移动距离。
3. `attenuation_radius` 控制 raw attenuation 最大修改幅度。

当前主实验中：

```bash
--triplane_output_mode gating_only
--triplane_offset_radius_scale 0.0
--triplane_att_radius 0.05
```

所以实际只改 attenuation，最多改：

```text
raw attenuation delta in [-0.05, 0.05]
```

### 10.7 `set_wavelet_active_levels(active_levels=None)`

包装函数，只在 Haar field 有效。用于 c2f。

### 10.8 `wavelet_l1_loss()`

包装函数，只在 Haar field 有实际 L1 loss。direct triplane 下返回 0。

---

## 11. `_scene_context(scene, device, dtype)`

代码位置：`scene/triplane_initializer.py:380`

作用：

```python
train_views = scene.getTrainSpectrums()
tx_stack = torch.stack([view.T_tx for view in train_views], dim=0)
rx_context = train_views[0].T_rx
tx_context = tx_stack.mean(dim=0)
```

返回：

```text
tx_context = 所有训练 TX 的平均位置
rx_context = 第一个训练 view 的 RX 位置
```

注意：

warmup 每一步如果 `triplane_use_view_tx=True`，会使用当前 viewpoint 的 `T_tx`，而不是平均 TX。

平均 TX 主要用于：

1. `triplane_use_view_tx=False` 时作为固定 context。
2. finalize 时作为 fallback。

---

## 12. `_make_warmup_optimizer(init_model, gaussians, model_args, optim_args)`

代码位置：`scene/triplane_initializer.py:388`

默认只优化 initializer：

```python
params = [{"params": init_model.parameters(), "lr": triplane_lr}]
```

如果打开：

```bash
--triplane_update_gaussian_attrs
```

则还会在 warmup 中一起优化 Gaussian 的：

```text
features_dc
features_rest
attenuation
scaling
rotation
```

目前配置默认：

```yaml
triplane_update_gaussian_attrs: false
```

实验上建议保持 false，因为 warmup 阶段如果直接动主 Gaussian 参数，容易让对照不干净，也更难判断收益来自哪里。

优化器：

```python
torch.optim.Adam(params, lr=0.0, eps=1e-15)
```

虽然全局 `lr=0.0`，但 param group 里设置了具体 lr，所以实际使用 param group lr。

---

## 13. `_finalize_with_tx_average(...)`

代码位置：`scene/triplane_initializer.py:408`

### 13.1 作用

warmup 训练时每一步随机用一个 viewpoint 的 TX。最终 materialize 时，如果只用一个 TX，可能会偏向某个 view。

所以这里对多个 TX 做平均：

```text
sample several train TX
  -> predict delta for each TX
  -> average delta_xyz and attenuation_delta
```

### 13.2 相关参数

```bash
--triplane_use_view_tx
--triplane_finalize_tx_samples
```

逻辑：

```python
if not use_view_tx or n_samples <= 1 or len(train_views) <= 1:
    return init_model(...)
```

否则：

```python
sample_ids = torch.linspace(0, len(train_views)-1, steps=n_samples)
```

默认推荐：

```bash
--triplane_use_view_tx true
--triplane_finalize_tx_samples 32
```

### 13.3 实验意义

这个设计让 initializer 学到的 attenuation delta 不完全绑定单个 TX，而是更接近平均 RF geometry prior。

但从 no-triplane 实验看，geometry feature 本身已经很强，这部分平均可能贡献了不少，而不是 plane feature 本身。

---

## 14. `run_triplane_init_warmup(...)`

代码位置：`scene/triplane_initializer.py:439`

这是整个文件最重要的入口函数。

### 14.1 Early return

```python
if not use_triplane_init:
    return
if warmup_iters <= 0:
    return
```

对应参数：

```bash
--use_triplane_init
--triplane_warmup_iters
```

如果要关闭 triplane warmup：

```bash
--triplane_warmup_iters 0
```

### 14.2 bbox 初始化

```python
base_xyz = gaussians.get_xyz.detach()
xyz_min = base_xyz.min(dim=0).values
xyz_max = base_xyz.max(dim=0).values
```

默认 plane 物理范围刚好覆盖初始 Gaussian bbox。

新增 pad：

```python
bbox_pad = (xyz_max - xyz_min) * triplane_bbox_pad_scale
xyz_min -= bbox_pad
xyz_max += bbox_pad
```

实验：

```text
res64 direct gate: 21.9598
res64 pad005:      21.9347
```

结论：

```text
pad=0.05 没有提升，但说明 plane physical extent 是一个可以报告的尺度敏感性因素。
```

### 14.3 offset radius

代码：

```python
offset_radius = (3.0e8 / frequency) * voxel_size_scale * triplane_offset_radius_scale
```

物理含义：

```text
最大 offset = wavelength * voxel_size_scale * scale
```

RFID 频率：

```text
frequency = 915 MHz
wavelength = 3e8 / 915e6 ≈ 0.328 m
voxel_size_scale = 1.846
wavelength * voxel_size_scale ≈ 0.606 m
```

所以如果：

```bash
--triplane_offset_radius_scale 0.05
```

最大 offset 约：

```text
0.606 * 0.05 ≈ 0.030 m
```

当前 gating-only 主实验使用：

```bash
--triplane_offset_radius_scale 0.0
```

因为不想让 initializer 移动 Gaussian。

### 14.4 初始化 `RFTriplaneInitializer`

传入：

```python
channels
resolution
hidden_dim
offset_radius
attenuation_radius
decoder_type
fourier_frequencies
film_layers
output_mode
field_type
wavelet_levels
wavelet_high_init
```

对应几乎所有 `triplane_*` 超参数。

### 14.5 Loss 权重

从 `optim_args` 读：

```python
lambda_ssim = optim_args.lambda_dssim
lambda_fourier = optim_args.lambda_dfourier
```

从 `model_args` 读：

```python
lambda_offset = triplane_offset_l2
lambda_attenuation = triplane_att_l2
lambda_wavelet = triplane_wavelet_l1
```

重建 loss：

```python
recon_loss =
    (1 - lambda_ssim - lambda_fourier) * L1
    + lambda_ssim * SSIM_loss
    + lambda_fourier * Fourier_loss
```

总 loss：

```python
loss =
    recon_loss
    + lambda_offset * offset_reg
    + lambda_attenuation * attenuation_reg
    + lambda_wavelet * wavelet_reg
```

### 14.6 c2f schedule

代码：

```python
active_levels = min(
    wavelet_levels,
    ((step - 1) * (wavelet_levels + 1)) // max(1, warmup_iters),
)
init_model.set_wavelet_active_levels(active_levels)
```

对于：

```text
wavelet_levels = 2
warmup_iters = 300
```

大致是：

| step 范围 | active levels |
|---|---:|
| 1-100 | 0 |
| 101-200 | 1 |
| 201-300 | 2 |

含义：

```text
先只用 LL，之后逐渐加入高频。
```

实验结果：

修复 c2f 后 Haar 10K 不再极差，但最终 30K 仍然不如 baseline。

### 14.7 Warmup loop

每一步：

```python
viewpoint = random train view
base_xyz = gaussians.get_xyz.detach()
base_attenuation = gaussians._attenuation.detach()
tx_step = viewpoint.T_tx if use_view_tx else tx_context
xyz_init, attenuation_raw_init = init_model(...)
render_fn(... override_xyz=xyz_init, override_attenuation=sigmoid(attenuation_raw_init))
```

关键点：

1. 默认不直接优化 Gaussian 主参数。
2. 用 render override 测试“如果这样初始化，会不会让当前 viewpoint spectrum 更接近 GT”。
3. warmup 学到的是初始化 delta，而不是长期参与训练的 neural field。

### 14.8 Regularization

```python
offset_reg = normalized delta_xyz L2
attenuation_reg = attenuation_delta^2
wavelet_reg = high-band L1
```

当前 gating-only：

```text
delta_xyz = 0
offset_reg = 0
```

所以主要是：

```text
recon_loss + triplane_att_l2 * attenuation_reg
```

### 14.9 Logging

每 20% warmup 打印一次：

```text
loss
recon
offset_reg
att_reg
wavelet_reg
active_wavelet_levels
mean_offset
```

实验中用这些日志确认：

1. Haar c2f 是否真的激活高频层。
2. gating-only 的 mean_offset 是否为 0。
3. attenuation delta 是否被限制在合理范围内。

### 14.10 Materialize

warmup 结束：

```python
init_model.set_wavelet_active_levels()
final_xyz, final_attenuation_raw = _finalize_with_tx_average(...)
gaussians.apply_initial_xyz_attenuation(...)
```

打印：

```text
mean_offset
max_offset
mean_att_delta
max_att_delta
```

当前 gating-only 典型日志：

```text
mean_offset=0.0000m
max_offset=0.0000m
mean_att_delta≈0.01-0.02
max_att_delta=0.0500
```

这说明 gating-only 确实没有改位置，只在 raw attenuation 上做小范围门控。

---

## 15. 超参数总表

### 15.1 开关类

| 参数 | 默认/常用值 | 含义 | 实验建议 |
|---|---:|---|---|
| `use_triplane_init` | `true` | 是否启用 warmup initializer | baseline 用 exp1 不启用；triplane 实验启用 |
| `triplane_warmup_iters` | `300` 或 `500` | warmup 步数 | gating-only 主实验用 300 |
| `triplane_reset_seed_after_warmup` | `true` | warmup 后重置随机种子 | 建议 true，保证主训练随机性可比 |

### 15.2 Plane 表示类

| 参数 | 当前推荐 | 含义 | 实验结论 |
|---|---:|---|---|
| `triplane_field_type` | `direct` | `direct` 或 `haar` | 当前 direct 更稳 |
| `triplane_resolution` | `64` | plane 分辨率 | 128 更差，不建议 256 |
| `triplane_channels` | `16` | 每张 plane channel 数 | 暂未深入扫，当前保持 16 |
| `triplane_bbox_pad_scale` | `0.0` | plane bbox 外扩比例 | 0.05 未提升 |

### 15.3 Wavelet 类

| 参数 | 当前值 | 含义 | 实验结论 |
|---|---:|---|---|
| `triplane_wavelet_levels` | `2` | Haar 分解层数 | levels=2 可跑，但未提升 |
| `triplane_wavelet_high_init` | `0.0` | 高频系数初始化 | 0 表示高频初始关闭 |
| `triplane_wavelet_l1` | `0.001` 或 `0.0` | 高频 L1 稀疏约束 | 当前没有收益 |
| `triplane_wavelet_c2f` | `true/false` | coarse-to-fine 激活高频 | 修复后仍不够好 |

### 15.4 Decoder 类

| 参数 | 当前推荐 | 含义 | 实验结论 |
|---|---:|---|---|
| `triplane_decoder_type` | `film_fourier` | decoder 类型 | 明显优于 `mlp` |
| `triplane_hidden_dim` | `128` | decoder hidden width | 当前主实验用 128 |
| `triplane_fourier_freqs` | `4` | Fourier 频率数 | 当前主实验用 4 |
| `triplane_film_layers` | `3` | FiLM residual block 数 | 当前主实验用 3 |

特殊 decoder：

```bash
--triplane_decoder_type film_fourier_no_triplane
```

含义：

```text
只用 RF geometry feature，不用 plane feature。
```

它是判断 plane feature 是否真正有贡献的关键对照。

### 15.5 输出和约束类

| 参数 | 当前推荐 | 含义 | 实验结论 |
|---|---:|---|---|
| `triplane_output_mode` | `gating_only` | 只改 attenuation，不改 xyz | 10K 最有收益 |
| `triplane_offset_radius_scale` | `0.0` | 最大位置偏移半径比例 | 当前不建议移动位置 |
| `triplane_offset_l2` | `0.0` | offset 正则 | gating-only 下无实际作用 |
| `triplane_att_radius` | `0.05` | raw attenuation delta 最大幅度 | 当前主实验值 |
| `triplane_att_l2` | `0.02` | attenuation delta L2 正则 | 当前主实验值 |

### 15.6 TX/RX 上下文类

| 参数 | 当前推荐 | 含义 |
|---|---:|---|
| `triplane_use_view_tx` | `true` | warmup 每步使用当前 viewpoint 的 TX |
| `triplane_finalize_tx_samples` | `32` | materialize 时平均多少个 TX |

### 15.7 Optimizer 类

| 参数 | 当前推荐 | 含义 |
|---|---:|---|
| `triplane_lr` | `0.001` | initializer Adam 学习率 |
| `triplane_update_gaussian_attrs` | `false` | warmup 是否同时更新 Gaussian 主属性 |

目前建议：

```text
保持 triplane_update_gaussian_attrs=false。
```

否则实验对照不干净，难以判断收益来自 initializer 还是提前训练了 Gaussian 参数。

---

## 16. 已完成实验如何对应代码假设

### 16.1 Naive triplane 为什么失败

实验：

```text
baseline_default_7k: 19.0910
triplane_default_7k: 18.4649
```

对应代码：

```bash
--triplane_output_mode offset_att
```

解释：

`offset_att` 同时改位置和 attenuation。RF Gaussian 的位置结构很敏感，warmup 只看短期 reconstruction，可能把点往错误方向推，主训练反而从更差的位置开始。

代码上直接体现为：

```python
delta_xyz = offset_radius * tanh(pred[:, :3])
```

### 16.2 为什么 gating-only 是更合理方向

实验：

```text
baseline_default_10k:      19.6692
triplane_gate_default_10k: 19.9350
```

对应代码：

```python
if output_mode == "gating_only":
    delta_xyz = torch.zeros_like(base_xyz)
```

解释：

gating-only 不改变 Gaussian 空间结构，只给 attenuation 一个小范围 raw delta：

```python
attenuation_delta = triplane_att_radius * tanh(pred[:, 3:4])
```

这更像 RF attenuation prior，而不是直接扰动几何。

### 16.3 为什么 plane feature 的贡献目前不强

实验：

```text
triplane_gate_default_30k:            21.9598
triplane_gate_notriplane_default_30k: 21.9593
```

对应代码：

```python
self.use_plane_features = not decoder_type.endswith("_no_triplane")
```

解释：

`film_fourier_no_triplane` 不用 plane feature，只用 RF geometry feature，但 30K 结果几乎一样。说明当前 direct plane 并没有学到比 geometry feature 更多的稳定初始化信息。

### 16.4 为什么 Fourier-FiLM decoder 有必要

实验：

```text
triplane_gate_default_10k:     19.9350
triplane_gate_mlp_default_10k: 19.4781
```

对应代码：

```text
RFMLPDecoder vs RFFiLMFourierDecoder
```

解释：

RF 传播依赖距离、方向、TX/RX 条件。普通 MLP concat 不够强，Fourier + FiLM 更适合表达这类几何依赖。

### 16.5 为什么 Haar wavelet 当前没有成为主方法

实验：

```text
triplane_gate_haar_default_30k:        21.9527
triplane_gate_haar_c2ffix_default_30k: 21.8159
baseline_default_30k:                  22.0436
```

对应代码：

```text
HaarWaveletTriPlaneField
wavelet_l1_loss
wavelet_c2f
```

解释：

Haar 的 LL/HF 分解逻辑是合理的，但当前实现仍然存在几个问题：

1. Haar 基太简单，可能不适合平滑 RF field。
2. warmup 只优化初始化，不是完整 field representation training。
3. high-band L1 可能压掉有用 multipath 细节。
4. 30K 主训练足够强，baseline 最终能追上甚至超过。

因此 Haar 可以作为 negative / exploratory ablation，不适合作为当前主结果。

### 16.6 为什么提高 resolution 不对

实验：

```text
res64 direct gate:  21.9598
res128 direct gate: 21.8020
```

对应代码：

```python
self.planes = nn.Parameter(0.01 * torch.randn(3, C, R, R))
```

解释：

提高 `R` 增加 plane 自由度，但 warmup 数据和步数有限。更高 resolution 更容易拟合局部噪声，不能稳定转化为后续 30K 收益。

### 16.7 bbox pad 的意义

实验：

```text
res64 direct gate: 21.9598
res64 pad005:      21.9347
```

对应代码：

```python
xyz_min -= bbox_pad
xyz_max += bbox_pad
```

解释：

`bbox_pad_scale=0.05` 让 plane 覆盖范围稍微变大，减少边界 clamp。但结果没有超过 direct gate，说明 bbox 不是主要瓶颈。

---

## 17. 当前推荐的“标准 direct gate”设置

如果需要复现当前最有价值的 direct gate：

```bash
--triplane_output_mode gating_only
--triplane_field_type direct
--triplane_resolution 64
--triplane_bbox_pad_scale 0.0
--triplane_decoder_type film_fourier
--triplane_hidden_dim 128
--triplane_fourier_freqs 4
--triplane_film_layers 3
--triplane_warmup_iters 300
--triplane_offset_radius_scale 0.0
--triplane_offset_l2 0.0
--triplane_att_radius 0.05
--triplane_att_l2 0.02
--triplane_finalize_tx_samples 32
```

对应结果：

```text
10K: 19.9350
30K: 21.9598
```

---

## 18. 实验思路应该如何整理成论文叙事

当前实验结果不适合写成：

```text
我们提出 TriPlane 初始化，并显著超过 GSRF。
```

因为 30K 最终结果没有超过 baseline。

更合理的论文叙事是：

```text
我们系统研究了 triplane-based RF Gaussian initialization。
实验发现，naive triplane 和高分辨率 triplane 并不能自然提升 RF Gaussian reconstruction。
真正有效的是更受约束的 attenuation gating，它改善 early convergence。
进一步的 no-triplane / Haar / resolution / bbox ablation 表明，RF 场景需要比 naive spatial plane 更明确的 RF-aware multiscale prior。
```

也就是说，当前工作可以从“单一方法增量”转为“机制分析 + RF-aware prior 设计”。

---

## 19. 下一步实验建议

### 19.1 短期不建议继续

不建议优先继续：

```text
resolution=256
bbox_pad_scale=0.10
更多 Haar L1/c2f 小调参
```

原因：

```text
res128 已经更差，pad005 没有提升，Haar 当前没有明显希望。
```

### 19.2 更有价值的下一步

建议补三类实验或分析。

第一类：RF-aware subband error

```text
对 predicted / GT spectrum 做 2D wavelet transform。
报告 LL-MSE 和 HF-MSE。
```

目的：

```text
证明方法是否改善 multipath/high-frequency，而不是只看整体 PSNR。
```

第二类：sparse measurement

```text
ratio_train=0.2
sparse220
```

目的：

```text
如果 gating 或 multiscale prior 在稀疏观测下更有效，论文故事会更强。
```

第三类：更合理 wavelet basis

```text
bior4.4
sym4
db4
```

目的：

```text
Haar 过于块状，可能不适合 RF spectrum 的平滑传播结构。
```

---

## 20. 总结

`scene/triplane_initializer.py` 的核心贡献是提供了一个训练前 RF-aware Gaussian initializer：

```text
TriPlane / Wavelet-TriPlane + RF geometry decoder
  -> bounded xyz / attenuation delta
  -> render-supervised warmup
  -> materialize to Gaussian
```

但目前实验表明：

1. 直接移动 Gaussian 位置不稳。
2. `gating_only` 是最合理的安全形式。
3. `film_fourier` decoder 明显优于普通 MLP。
4. direct plane feature 在 30K 最终贡献很弱。
5. Haar wavelet 当前没有超过 direct/baseline。
6. 提高 plane resolution 和 bbox pad 都不是突破口。

当前最应该保留的正向发现是：

```text
attenuation-only RF geometry gating improves early convergence.
```

当前最应该作为机制发现写出来的是：

```text
naive triplane capacity is insufficient for RF Gaussian initialization.
```
