# TriPlane-FiLM 与 Gating-only 的代码思路详细对比

## 0. 先回答最重要的问题：有没有都用到 TriPlane？

**有。两者都用到了 TriPlane。**

更准确地说，现在代码里不是两个完全不同的网络，而是同一个初始化框架 `RFTriplaneInitializer` 的两种输出模式：

```text
共同部分：
Gaussian center
-> TriPlaneField 查询三平面特征
-> 拼接 TX/RX 几何特征
-> FiLM-Fourier decoder
-> 输出 pred

不同部分：
TriPlane-FiLM offset+att: 使用 pred 的前 3 维预测 delta_xyz，使用第 4 维预测 delta_attenuation
Gating-only: 强制 delta_xyz = 0，只使用第 4 维预测 delta_attenuation
```

所以 `gating-only` 不是“没有 triplane”，也不是“普通 MLP gate”。它仍然是：

```text
TriPlane feature + TX/RX geometry -> RF-aware Gaussian attenuation gate
```

名称上的区别需要特别注意：

```text
film_fourier 是 decoder 类型
gating_only 是 output mode
```

当前实验里的 gating-only 实际上仍然使用：

```bash
--triplane_decoder_type film_fourier
--triplane_output_mode gating_only
```

因此它更完整的名字应该是：

```text
TriPlane-FiLM gating-only initialization
```

而不是简单的“非 triplane gate”。

## 1. 代码位置

主要网络：

```text
/home/wys/GSRF/scene/triplane_initializer.py
```

训练接入：

```text
/home/wys/GSRF/train_rfid.py
```

配置：

```text
/home/wys/GSRF/arguments/configs/rfid/exp_triplane.yaml
```

实验脚本：

```text
/home/wys/GSRF/run_rfid_triplane_experiments.sh
```

最核心的类：

```python
TriPlaneField
FourierGeometryEncoding
FiLMResidualBlock
RFFiLMFourierDecoder
RFTriplaneInitializer
```

其中 `TriPlaneField` 是空间特征场，`RFFiLMFourierDecoder` 是 TX/RX 条件解码器，`RFTriplaneInitializer` 把它们组合起来。

## 2. 整体 warmup 流程

triplane 初始化发生在 GSRF 正式训练之前。

入口在 `train_rfid.py`：

```python
run_triplane_init_warmup(scene, gaussians, model_para_args, optim_para_args, pipe_para_args, render)
```

完整流程可以理解为：

```text
1. GSRF 先生成一批初始 Gaussians
2. 取出每个 Gaussian 的 base_xyz 和 base_attenuation_raw
3. 创建一个 learnable TriPlaneField
4. 每个 warmup step 随机取一个训练 TX view
5. 对所有 Gaussian 查询 triplane feature
6. 拼接 TX/RX 几何特征
7. decoder 输出 delta_xyz / delta_attenuation
8. 用 override_xyz 和 override_attenuation 临时渲染
9. 用 RF spectrum loss 反传，只训练 triplane initializer
10. warmup 结束后，把最终预测的初始化结果 materialize 到 Gaussian
11. 再开始正常 GSRF 训练
```

注意第 8 步非常关键：warmup 阶段不是直接改 renderer，而是通过：

```python
render_fn(
    viewpoint,
    gaussians,
    pipe_args,
    override_xyz=xyz_init.contiguous(),
    override_attenuation=torch.sigmoid(attenuation_raw_init),
)
```

临时把初始化网络预测出来的 Gaussian 位置和 attenuation 送进去渲染。

warmup 完成后才真正写入：

```python
gaussians.apply_initial_xyz_attenuation(final_xyz, final_attenuation_raw)
```

所以当前方法属于：

```text
initialization-time method
```

不是：

```text
render-time dynamic modulation
```

这点写论文时要说清楚。

## 3. TriPlaneField 到底做了什么？

代码：

```python
class TriPlaneField(nn.Module):
    def __init__(self, xyz_min, xyz_max, channels=16, resolution=64):
        self.planes = nn.Parameter(
            0.01 * torch.randn(3, self.channels, self.resolution, self.resolution)
        )
```

这里创建的是三张二维特征平面：

```text
planes[0] -> XY plane
planes[1] -> XZ plane
planes[2] -> YZ plane
```

如果 `channels=16`、`resolution=64`，那么参数形状是：

```text
3 x 16 x 64 x 64
```

每个 Gaussian center `xyz` 会先归一化到 `[-1, 1]`：

```python
xyz_norm = self.normalize_xyz(xyz)
```

然后分别投影到三个平面：

```python
xy = xyz_norm[:, [0, 1]]
xz = xyz_norm[:, [0, 2]]
yz = xyz_norm[:, [1, 2]]
```

再用 `grid_sample` 从三张平面上取特征：

```python
f_xy = self._sample_plane(self.planes[0], xy)
f_xz = self._sample_plane(self.planes[1], xz)
f_yz = self._sample_plane(self.planes[2], yz)
```

最后拼接：

```python
tri_feat = torch.cat([f_xy, f_xz, f_yz], dim=-1)
```

如果每张平面 16 维，那么：

```text
tri_feat dimension = 16 x 3 = 48
```

直观理解：

```text
TriPlane 不是直接存一个完整 3D voxel grid。
它用三张 2D feature planes 近似表示 3D 空间中的可学习先验。
```

为什么适合这里？

因为 GSRF 的初始 Gaussian 是分布在 3D 空间里的。我们希望网络知道：

```text
这个 Gaussian 位于空间中的什么区域？
这个区域在 RF 场景里更可能有用还是无用？
```

TriPlane 给每个 Gaussian 一个可学习的空间上下文特征。

## 4. RF geometry feature 是什么？

代码在：

```python
RFTriplaneInitializer._geometry_features(...)
```

构造出的几何特征维度是 18：

```text
xyz_norm: 3
tx_norm: 3
rx_norm: 3
d_tx: 1
d_rx: 1
d_total: 1
dir_tx: 3
dir_rx: 3
total: 18
```

具体含义：

```text
xyz_norm:
  当前 Gaussian 的归一化坐标

tx_norm:
  当前 TX 的归一化坐标

rx_norm:
  RX 的归一化坐标

d_tx:
  TX 到 Gaussian 的距离

d_rx:
  Gaussian 到 RX 的距离

d_total:
  TX -> Gaussian -> RX 的总距离

dir_tx:
  TX 指向 Gaussian 的方向向量

dir_rx:
  Gaussian 指向 RX 的方向向量
```

对应 RF 物理意义：

```text
d_tx / d_rx / d_total:
  近似反映 path length、path loss、传播延迟、相位变化相关因素

dir_tx / dir_rx:
  近似反映传播方向、角度相关的多径结构

TX/RX 坐标:
  让网络知道当前测量条件，而不是学习一个完全静态的场
```

所以当前网络输入不是只有 Gaussian 位置，而是：

```text
Gaussian spatial prior + TX/RX propagation geometry
```

这比普通 3DGS 初始化更贴近 RF 场景。

## 5. FiLM-Fourier decoder 是什么？

当前主要用的是：

```python
class RFFiLMFourierDecoder(nn.Module)
```

它做两件事。

第一，对 18 维几何特征做 Fourier encoding：

```python
self.fourier = FourierGeometryEncoding(num_frequencies=4, include_input=True)
```

`num_frequencies=4` 时，每个输入维度会变成：

```text
原始输入 1 份
sin 编码 4 份
cos 编码 4 份
总共 1 + 4 + 4 = 9 份
```

所以 18 维几何特征会变成：

```text
18 x 9 = 162
```

再和 48 维 triplane feature 拼起来：

```text
decoder input = tri_feat 48 + Fourier geometry 162 = 210
```

第二，用 FiLM residual blocks 做条件调制。

代码：

```python
gamma, beta = self.film(cond).chunk(2, dim=-1)
x = self.norm(self.linear(h))
x = x * (1.0 + 0.1 * torch.tanh(gamma)) + 0.1 * torch.tanh(beta)
return h + self.act(x)
```

FiLM 的作用是：让 TX/RX 几何条件去调制 hidden feature。

直观说：

```text
同一个 Gaussian，在不同 TX/RX 条件下，它的重要性应该不同。
FiLM 就是把 TX/RX 条件注入到网络中，让 decoder 不是纯静态地看空间位置。
```

不过当前最终 materialization 仍然是初始化时写入一次，所以它不是完全动态的 render-time 条件网络。

## 6. RFTriplaneInitializer 的共同 forward

两种方法共用下面这段逻辑：

```python
tri_feat, xyz_norm = self.triplane(base_xyz)
geom_feat = self._geometry_features(base_xyz, xyz_norm, tx_context, rx_context)

pred = self.head(tri_feat, geom_feat)
```

这里：

```text
self.triplane(...) -> 一定会被调用
self.head(...)     -> 当前实验中是 RFFiLMFourierDecoder
pred               -> 4 维输出
```

4 维输出的默认解释是：

```text
pred[:, 0] -> x direction offset
pred[:, 1] -> y direction offset
pred[:, 2] -> z direction offset
pred[:, 3] -> attenuation raw delta
```

也就是：

```text
pred = [dx_raw, dy_raw, dz_raw, datt_raw]
```

真正的分歧发生在 `output_mode`：

```python
if self.output_mode == "gating_only":
    delta_xyz = torch.zeros_like(base_xyz)
else:
    delta_xyz = self.offset_radius * torch.tanh(pred[:, :3])

attenuation_delta = self.attenuation_radius * torch.tanh(pred[:, 3:4])
attenuation_raw = base_attenuation_raw + attenuation_delta
```

所以可以画成：

```text
                              pred[0:3] -> delta_xyz
TriPlane + Geometry + FiLM -> 
                              pred[3]   -> delta_attenuation
```

TriPlane-FiLM 使用两条输出。

Gating-only 只使用 attenuation 这条输出，位置输出被强制置零。

## 7. TriPlane-FiLM offset+att 版本

实验名：

```text
triplane_film_default_10k
```

脚本：

```bash
GPU=0 ./run_rfid_triplane_experiments.sh triplane_film10k
```

关键参数：

```bash
--triplane_decoder_type film_fourier
--triplane_warmup_iters 300
--triplane_offset_radius_scale 0.05
--triplane_offset_l2 0.05
--triplane_att_radius 0.1
--triplane_att_l2 0.01
--triplane_finalize_tx_samples 32
```

它没有显式传：

```bash
--triplane_output_mode offset_att
```

因为配置文件默认就是：

```yaml
triplane_output_mode: "offset_att"
```

因此它的输出是：

```text
delta_xyz = offset_radius * tanh(pred[:, :3])
delta_att = att_radius * tanh(pred[:, 3:4])
```

数学形式：

```text
x_i' = x_i + r_x * tanh(f_xyz(T_i, G_i, tx, rx))

a_i' = a_i + r_a * tanh(f_att(T_i, G_i, tx, rx))
```

其中：

```text
x_i:
  第 i 个 Gaussian 的中心

a_i:
  第 i 个 Gaussian 的 raw attenuation

T_i:
  第 i 个 Gaussian 查询到的 triplane feature

G_i:
  第 i 个 Gaussian 对应的 TX/RX geometry feature

r_x:
  offset_radius

r_a:
  attenuation_radius
```

它的思想是：

```text
让 triplane 网络学习“这个 Gaussian 应该移动到哪里，以及应该多强”。
```

优点：

```text
表达能力强。
如果初始 Gaussian 的空间位置确实不准，它理论上可以修正位置。
```

问题：

```text
RF 场景对位置非常敏感。
小的中心移动可能改变相位、多径叠加和局部衰落结构。
```

所以在 RF 中，`delta_xyz` 不一定是好事。它可能破坏原本还可以的 Gaussian 几何结构。

10k 实验结果：

```text
baseline:
  PSNR 19.6692
  MSE  0.014311
  SSIM 0.7464

TriPlane-FiLM offset+att:
  PSNR 19.5572
  MSE  0.014452
  SSIM 0.7455
```

结论：

```text
允许移动 Gaussian center 后，平均结果反而变差。
```

这不是说 triplane 没用，而是说：

```text
triplane 用来直接预测 position offset 风险较高。
```

## 8. Gating-only 版本

实验名：

```text
triplane_gate_default_10k
```

脚本：

```bash
GPU=0 ./run_rfid_triplane_experiments.sh triplane_gate10k
```

关键参数：

```bash
--triplane_output_mode gating_only
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

注意这两行：

```bash
--triplane_output_mode gating_only
--triplane_decoder_type film_fourier
```

这说明：

```text
gating-only 仍然使用 FiLM-Fourier decoder。
gating-only 仍然使用 TriPlaneField。
```

它只是把位置偏移关闭：

```python
if self.output_mode == "gating_only":
    delta_xyz = torch.zeros_like(base_xyz)
```

数学形式：

```text
x_i' = x_i

a_i' = a_i + r_a * tanh(f_att(T_i, G_i, tx, rx))
```

所以 gating-only 的思想是：

```text
不要改变 Gaussian 在空间中的位置。
只根据 triplane spatial prior 和 TX/RX geometry 判断这个 Gaussian 应该增强还是减弱。
```

这里的 gate 不是显式写成 `0 到 1` 的概率，而是 raw attenuation 的一个有界增量：

```text
delta_att in [-att_radius, +att_radius]
```

然后渲染时会使用：

```python
override_attenuation = torch.sigmoid(attenuation_raw_init)
```

所以它实际改变的是：

```text
Gaussian 的有效 attenuation / contribution strength
```

为什么叫 gating？

因为它不改变 Gaussian 位置，只改变 Gaussian 的贡献强弱：

```text
增强有用 Gaussian
抑制不可靠 Gaussian
```

这相当于给已有 Gaussian initialization 加了一个 RF-aware contribution prior。

## 9. 为什么 gating-only 更适合当前 GSRF？

你之前一直强调：

```text
initialization 是核心。
RF 中 TX 很小的位置变化也可能导致多径结构大变化。
```

这正好解释了实验现象。

GSRF 的原始 Gaussian 位置虽然不完美，但它已经来自一个几何初始化流程。直接让神经网络在 warmup 阶段移动中心，会出现两个风险：

```text
1. warmup 数据有限，网络可能学到局部过拟合的位置偏移。
2. RF 多径对位置敏感，小位移可能改变相位关系，导致泛化变差。
```

而 gating-only 不移动位置，它更像是在问：

```text
在当前 TX/RX 条件和空间先验下，这个 Gaussian 值不值得被信任？
```

这种问题比“它应该移动到哪里”更容易学，也更稳定。

因此当前实验支持的主张是：

```text
RF-aware initialization should first modulate Gaussian contribution,
instead of directly relocating Gaussian centers.
```

中文论文表述可以是：

```text
由于 RF 场景中的小尺度多径衰落对空间位置高度敏感，直接预测 Gaussian 中心偏移容易破坏已有几何初始化。
因此我们采用更保守的 TriPlane-guided attenuation gating，在固定 Gaussian 中心的前提下，
根据 TX/RX 几何关系选择性增强或抑制 Gaussian 的贡献。
```

## 10. warmup loss 如何训练这个 gate？

warmup 时用的是和 GSRF 训练类似的 RF spectrum reconstruction loss：

```python
ll1 = l1_loss(spectrum, gt_spectrum)
ssim_loss = 1.0 - ssim(...)
lf = fourier_loss(spectrum, gt_spectrum)

recon_loss = (1.0 - lambda_ssim - lambda_fourier) * ll1 \
           + lambda_ssim * ssim_loss \
           + lambda_fourier * lf
```

然后加正则：

```python
offset_reg = (delta_xyz / offset_radius).pow(2).sum(dim=-1).mean()
attenuation_reg = attenuation_delta.pow(2).mean()

loss = recon_loss + lambda_offset * offset_reg + lambda_attenuation * attenuation_reg
```

对 gating-only 来说：

```text
delta_xyz = 0
offset_reg = 0
```

所以实际优化目标主要是：

```text
RF spectrum reconstruction loss + attenuation_delta regularization
```

这让 gate 不能随便把 attenuation 改得太夸张。

当前 gating-only 的约束是：

```bash
--triplane_att_radius 0.05
--triplane_att_l2 0.02
```

最终 warmup 诊断：

```text
mean_offset = 0.0000m
max_offset  = 0.0000m
mean_att_delta = 0.0159
max_att_delta  = 0.0500
```

这说明它确实没有移动 Gaussian，只做了较小幅度的 attenuation 修正。

## 11. 最终 materialization 如何处理 TX？

warmup 中每一步可以使用当前 viewpoint 的 TX：

```python
tx_step = viewpoint.T_tx if use_view_tx else tx_context
```

但是最终初始化要写成一个静态 Gaussian，所以代码用多个 TX 样本平均：

```python
final_xyz, final_attenuation_raw, final_delta, final_attenuation_delta =
    _finalize_with_tx_average(...)
```

关键逻辑：

```python
for idx in sample_ids:
    _, _, delta, attenuation_delta = init_model(
        base_xyz,
        base_attenuation_raw,
        train_views[idx].T_tx,
        rx_context,
    )
    delta_sum += delta
    attenuation_delta_sum += attenuation_delta

final_delta = delta_sum / n_samples
final_attenuation_delta = attenuation_delta_sum / n_samples
```

也就是说：

```text
当前方法不是为每一个 TX 动态存一个不同 Gaussian。
它是在多个训练 TX 条件下估计一个平均初始化 prior。
```

这也是为什么它仍属于 initialization 方法。

后续如果要更强，可以做 render-time TX-conditioned modulation，但那会更难，因为需要在每次 render 时动态调整 Gaussian contribution。

## 12. 实验结果对比

10k default split：

| 方法 | 是否用 TriPlane | 是否移动 Gaussian | 是否调 attenuation | PSNR ↑ | MSE ↓ | SSIM ↑ |
|---|---|---:|---:|---:|---:|---:|
| baseline | 否 | 否 | 原始训练 | 19.6692 | 0.014311 | 0.7464 |
| TriPlane-FiLM offset+att | 是 | 是 | 是 | 19.5572 | 0.014452 | 0.7455 |
| TriPlane-FiLM gating-only | 是 | 否 | 是 | 19.9350 | 0.013468 | 0.7552 |

提升：

```text
Gating-only vs baseline:
PSNR +0.2657 dB
MSE  -0.000843
SSIM +0.0087
```

逐样本统计：

```text
Gating-only vs baseline:
n = 1225
mean delta PSNR   = +0.2657 dB
median delta PSNR = +0.2371 dB
improved samples  = 704
worse samples     = 521
```

这说明 gating-only 不是只靠少数极端样本拉高均值。它在超过一半测试样本上都有提升。

## 13. 方法命名建议

不建议论文里叫：

```text
TriPlane initialization
```

这个太泛，而且容易被问：

```text
你到底是预测位置，还是预测 attenuation？
```

建议叫：

```text
TriPlane-guided RF-aware Gaussian Gating
```

或者：

```text
TriPlane-guided Attenuation Gating for RF Gaussian Initialization
```

中文理解：

```text
基于 TriPlane 的 RF 感知 Gaussian 贡献门控初始化
```

核心卖点：

```text
不是让网络重新放置 Gaussian，
而是让网络学习哪些 Gaussian 在 RF 传播条件下更可信。
```

## 14. 可以写进论文的贡献表述

可以这样组织贡献：

```text
1. We identify that directly optimizing Gaussian center offsets during RF initialization is unstable,
   because small spatial perturbations may cause large changes in multipath fading.

2. We propose a TriPlane-guided RF-aware Gaussian gating module,
   which queries learnable triplane features at Gaussian centers and combines them with TX/RX geometry.

3. Instead of relocating Gaussians, the proposed gating-only design modulates Gaussian attenuation,
   providing a conservative but effective initialization prior for RF field reconstruction.
```

中文版本：

```text
1. 我们发现，在 RF 场重建中直接预测 Gaussian 中心偏移并不稳定，
   因为小尺度空间扰动可能引起明显的多径衰落变化。

2. 我们提出基于 TriPlane 的 RF 感知 Gaussian 门控初始化模块，
   在 Gaussian 中心查询三平面空间特征，并融合 TX/RX 几何传播特征。

3. 与直接移动 Gaussian 不同，我们只对 Gaussian attenuation 进行有界调制，
   从而在保持几何初始化稳定性的同时，提高 RF spectrum 重建质量。
```

## 15. 你答辩时可以怎么解释

如果老师问：

```text
gating-only 还算 triplane 吗？
```

回答：

```text
算。gating-only 仍然使用 TriPlaneField 在 Gaussian center 查询三平面特征，
也仍然使用 TX/RX-conditioned FiLM-Fourier decoder。
它和 TriPlane-FiLM offset 版本的区别只在输出约束：
我们不使用网络预测的位置偏移，而是只使用 attenuation correction。
因此 gating-only 是 TriPlane-guided attenuation gating，而不是去掉 triplane。
```

如果老师问：

```text
为什么不移动 Gaussian？
```

回答：

```text
RF 场景的多径衰落对空间位置非常敏感。
实验中我们发现同时预测 delta_xyz 和 delta_attenuation 会使结果低于 baseline。
这说明直接移动 Gaussian 容易破坏已有几何初始化。
因此我们采用更保守的策略：固定 Gaussian center，只学习 attenuation gate。
实验中该策略将 PSNR 从 19.6692 提升到 19.9350。
```

如果老师问：

```text
那 triplane 的作用是什么？
```

回答：

```text
TriPlane 提供了一个可学习的空间先验。
它不是直接输出 RF 场，而是在每个 Gaussian center 上提供局部空间上下文。
decoder 再结合 TX/RX 距离和方向信息，判断该 Gaussian 对 RF 重建应当增强还是抑制。
```

## 16. 当前结论

当前实验不支持：

```text
TriPlane 直接预测 Gaussian center offset 是有效的。
```

当前实验支持：

```text
TriPlane-guided attenuation gating 是有效的。
```

所以现在最合理的论文主线应该从：

```text
Wavelet/TriPlane improves Gaussian position initialization
```

转成：

```text
TriPlane-guided RF-aware gating improves Gaussian contribution initialization
```

这条线更稳，更符合实验结果，也更容易解释为什么 RF 场景不同于普通视觉 3DGS。
