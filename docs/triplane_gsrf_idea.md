# Triplane-Guided RF-Aware Gaussian Initialization for GSRF

本文档记录当前 `triplane` 方案的研究思路、论文动机、网络设计、代码对应关系和实验计划。它不是代码说明书，而是给 PRICAI/PRACAI short paper 使用的“方法主线文档”。

## 1. 一句话贡献

我们提出一种 **Triplane-guided RF-aware Gaussian initialization** 方法，在 GSRF 正式优化之前，利用 TX/RX 几何关系和可学习的三平面空间先验，预测每个 Gaussian 的位置修正量和 RF 衰减属性修正量，从而缓解 GSRF 原始初始化缺乏 RF 传播先验的问题。

更直接地说：

```text
原始 GSRF:
    随机/均匀初始化 Gaussian
    -> 直接依赖后续优化把 Gaussian 调整到合适位置

我们的思路:
    先用 Triplane + TX/RX 几何学习一个 RF-aware initialization prior
    -> 修正 Gaussian 初始位置和衰减属性
    -> 再进入原始 GSRF 优化流程
```

## 2. 为什么 GSRF 需要改 initialization

3DGS 在视觉任务中通常可以依赖 SfM 点云初始化。SfM 点云本身已经包含了相机多视角几何信息，所以初始 Gaussian 往往和真实场景结构有一定对应关系。

但 RF 场景不同。GSRF 面对的是 RF spatial spectrum / channel field reconstruction，没有天然的视觉点云先验。原始 Gaussian 初始化更接近一种 RF-blind 的空间撒点方式。

这会带来一个关键问题：

```text
RF 场景中，TX/RX 位置的小变化可能导致信道响应的大变化。
如果 Gaussian 初始位置和 RF 传播结构偏离太多，后续优化不一定容易修正回来。
```

尤其是在室内或复杂环境中，多径、遮挡、反射、绕射会让信道呈现强烈的空间非平滑性。初始化如果只看三维空间坐标，而不看 TX/RX 传播几何，就会缺少对 RF 物理结构的感知。

因此，本工作的核心不是“为了稀疏采样而稀疏采样”，而是：

```text
GSRF 缺少 RF-aware Gaussian initialization。
```

Sparse measurement 可以作为补充鲁棒性实验，但不应该作为主故事。

## 3. Triplane 的作用

Triplane 的作用是提供一个紧凑的、可学习的 3D 空间先验。

完整 3D voxel grid 的参数量比较大，而普通 MLP 又不容易显式保存局部空间结构。Triplane 用三个二维特征平面近似表示三维空间：

```text
P_xy, P_xz, P_yz
```

对于一个 Gaussian 中心：

```text
x_i = (x, y, z)
```

分别从三个平面采样：

```text
f_xy = sample(P_xy, x, y)
f_xz = sample(P_xz, x, z)
f_yz = sample(P_yz, y, z)
```

然后拼接：

```text
f_tri(x_i) = concat(f_xy, f_xz, f_yz)
```

这个特征可以理解为：

```text
这个空间位置在 RF 场景中可能对应什么传播区域？
它更像 LoS path 附近？
更像反射/多径敏感区域？
还是低贡献区域？
```

所以 Triplane 不是最终 renderer，也不是替代 GSRF，而是在 initialization 阶段给 Gaussian 一个更好的空间先验。

## 4. 网络输入

当前方法对每个 Gaussian 构造一个特征向量。

### 4.1 Gaussian 位置特征

原始 Gaussian 中心：

```text
x_i = (x, y, z)
```

归一化到场景 bounding box：

```text
x_i_norm in [-1, 1]^3
```

作用：告诉网络当前 Gaussian 在场景中的相对位置。

### 4.2 TX/RX 位置特征

TX 和 RX 坐标也归一化：

```text
tx_norm
rx_norm
```

作用：让网络知道这个 Gaussian 是在什么发射和接收配置下被判断的。

### 4.3 距离特征

对于每个 Gaussian：

```text
d_tx    = ||x_i - TX||
d_rx    = ||RX - x_i||
d_total = d_tx + d_rx
```

作用：

```text
d_tx    表示 Gaussian 离发射端多远
d_rx    表示 Gaussian 离接收端多远
d_total 表示它是否处于可能的传播路径附近
```

这对应 RF 中的大尺度传播趋势，例如 path loss。

### 4.4 方向特征

```text
dir_tx = normalize(x_i - TX)
dir_rx = normalize(RX - x_i)
```

作用：让网络感知传播方向。多径不是只由距离决定，反射、绕射和遮挡都和方向有关。

### 4.5 Triplane 空间特征

```text
f_tri(x_i) = concat(f_xy, f_xz, f_yz)
```

作用：提供可学习的空间 prior，表达哪些位置更可能产生有效 RF 贡献。

### 4.6 完整输入

当前实现的输入为：

```text
feature_i =
[
    f_tri(x_i),
    x_i_norm,
    tx_norm,
    rx_norm,
    d_tx,
    d_rx,
    d_total,
    dir_tx,
    dir_rx
]
```

当前配置下：

```text
triplane feature: 16 * 3 = 48 dims
geometry feature: 18 dims
total input: 66 dims
```

## 5. 网络输出

MLP 输出 4 个数：

```text
pred_i = [a_x, a_y, a_z, a_att]
```

前三个用于位置修正：

```text
Delta x_i = offset_radius * tanh([a_x, a_y, a_z])
```

新的 Gaussian 中心：

```text
x_i' = x_i + Delta x_i
```

第四个用于 attenuation 修正：

```text
attenuation_raw_i' = attenuation_raw_i + a_att
attenuation_i' = sigmoid(attenuation_raw_i')
```

所以两个核心输出是：

```text
Delta xyz:
    调整 Gaussian 的空间位置

Delta attenuation:
    调整 Gaussian 的 RF 贡献强弱
```

这里使用 residual 形式，而不是直接预测最终值，是为了让方法更稳定。初始化开始时，最后一层 MLP 被置零，因此：

```text
Delta xyz = 0
Delta attenuation = 0
```

也就是说，模型一开始严格等价于原始 GSRF 初始化，然后通过 warm-up 逐步学习修正。

## 6. 训练方式

这个 initializer 没有显式标签。我们不知道每个 Gaussian 真实应该移动多少，也不知道它真实 attenuation 应该是多少。

它通过 RF rendering loss 间接学习：

```text
1. 取原始 Gaussian 参数
2. Triplane initializer 预测 x_i' 和 attenuation_i'
3. 使用修正后的 Gaussian 调用 GSRF render_rfid
4. 得到预测 spectrum
5. 和真实 spectrum 计算 loss
6. 反向传播更新 Triplane 和 MLP
```

当前 warm-up loss 和 GSRF 原始训练保持一致：

```text
Loss = L1 + SSIM loss + Fourier loss
```

warm-up 完成后，预测出的 `x_i'` 和 `attenuation_i'` 会被写回 Gaussian，然后进入原始 GSRF training。

方法流程：

```text
Base Gaussian initialization
        |
        v
Triplane + TX/RX geometry initializer
        |
        v
RF-aware Gaussian initialization
        |
        v
Original GSRF optimization
        |
        v
RF spatial spectrum reconstruction
```

## 7. 和 GSRF 的关系

本方法不是重写 GSRF。

保留内容：

```text
GSRF renderer
GSRF main optimization
GSRF loss
GSRF dataset split
GSRF inference protocol
```

修改内容：

```text
Gaussian initialization stage
```

论文表述时要强调：

```text
We do not replace the RF Gaussian renderer.
Instead, we introduce an RF-aware initialization prior before the standard GSRF optimization.
```

这样贡献更稳，也更适合 short paper。

## 8. 当前代码对应关系

### 8.1 核心方法文件

```text
/home/wys/GSRF/scene/triplane_initializer.py
```

包含：

```text
TriPlaneField
RFTriplaneInitializer
run_triplane_init_warmup
```

### 8.2 训练入口

```text
/home/wys/GSRF/train_rfid.py
```

在 Scene 和 Gaussian 创建后、正式 training_setup 前调用：

```text
run_triplane_init_warmup(...)
```

### 8.3 Renderer 修改

```text
/home/wys/GSRF/gaussian_renderer/__init__.py
```

给 `render_rfid` 增加：

```text
override_xyz
override_attenuation
```

这样 warm-up 阶段可以临时用修正后的 Gaussian 渲染，而不破坏原始 GSRF 训练流程。

### 8.4 Gaussian 写回

```text
/home/wys/GSRF/scene/gaussian_model.py
```

增加：

```text
apply_initial_xyz_attenuation(...)
```

用于 warm-up 结束后把预测结果真正写回 Gaussian 参数。

### 8.5 配置文件

```text
/home/wys/GSRF/arguments/configs/rfid/exp_triplane.yaml
```

核心配置：

```yaml
use_triplane_init: true
triplane_warmup_iters: 500
triplane_resolution: 64
triplane_channels: 16
triplane_hidden_dim: 64
triplane_lr: 0.001
triplane_offset_radius_scale: 0.5
```

### 8.6 实验脚本

```text
/home/wys/GSRF/run_rfid_triplane_experiments.sh
```

主实验：

```bash
GPU=0 ./run_rfid_triplane_experiments.sh triplane_default7k
```

## 9. 实验主线

主实验必须围绕“初始化改进”展开，而不是围绕 sparse measurement 展开。

### 9.1 Main comparison

默认 80/20 split：

```text
GSRF baseline
Triplane-Guided GSRF
```

指标：

```text
PSNR
MSE
SSIM
```

如果 Triplane 方法有效，应该看到：

```text
higher PSNR
lower MSE
higher SSIM
```

同时需要看收敛曲线：

```text
same iteration 下是否更好
early-stage 是否更快
```

### 9.2 Initialization quality analysis

建议分析：

```text
mean/max Gaussian offset
训练前后 spectrum 可视化
error map
收敛曲线
```

这部分是证明“不是单纯堆网络”，而是 initialization 真的影响了后续优化。

### 9.3 Ablation studies

#### Warm-up iteration

```text
100
500
1000
```

目的：验证 initializer 需要多少预热，不是越长越好。

#### Triplane resolution

```text
32
64
128
```

目的：验证空间先验表达能力。

预期：

```text
32 可能表达不足
64 可能较稳
128 可能更强但容易过拟合或训练慢
```

#### Offset radius

```text
0.25
0.5
1.0
```

目的：验证 Gaussian 位置修正尺度。

预期：

```text
太小: 修正不够
太大: 初始化可能不稳定
中间值: 更稳
```

### 9.4 Sparse measurement as secondary experiment

Sparse measurement 不作为主贡献背景，只作为补充实验：

```text
当训练观测变少时，RF-aware initialization 是否更稳？
```

这个实验可以放在 robustness / data efficiency 部分。

## 10. 论文贡献写法

可以写成三点：

```text
1. We identify the RF-blind initialization limitation in GSRF, where Gaussian parameters are initialized without explicitly considering TX/RX propagation geometry.

2. We propose a triplane-guided RF-aware Gaussian initializer that combines compact spatial features with TX/RX distance and direction cues to predict Gaussian center and attenuation residuals.

3. Experiments on the RFID spatial spectrum synthesis task show that the proposed initialization improves reconstruction quality and convergence over the original GSRF baseline.
```

中文理解：

```text
1. 指出 GSRF 初始化不感知 RF 传播几何。
2. 提出 Triplane + TX/RX geometry 的 RF-aware Gaussian 初始化。
3. 在 NeRF2/RFID spatial spectrum synthesis 上验证效果。
```

## 11. 当前方法的不足

当前版本是 V1，优点是稳、容易跑实验，但还有不足：

```text
1. 当前主要预测 Delta xyz 和 Delta attenuation，还没有预测 scale/rotation。
2. 当前 TX context 使用训练集 TX 均值，不是每个 TX 独立条件化。
3. 当前没有接入 spectrum/CSI encoder。
4. Triplane 只用于 initialization，未进入完整 rendering representation。
```

这些不足可以转化为 future work，也可以作为后续增强方向。

## 12. 下一步增强方向

优先级从高到低：

```text
1. 先跑通 baseline vs triplane，确认主结果是否提升。
2. 如果提升明显，做 warmup/resolution/offset ablation。
3. 如果提升不明显，增强 TX conditioning，用当前 viewpoint 的 TX 或多 TX aggregation。
4. 进一步预测 scale residual。
5. 最后再考虑加入 wavelet/spectrum encoder。
```

短期最重要的不是把方法做复杂，而是先拿到一组可以支撑 short paper 的主结果。

## 13. 适合放在论文方法图里的结构

```text
              TX coordinate
                   |
              RX coordinate
                   |
Base Gaussian --> Geometry Features -----
      |                                  |
      v                                  v
  Triplane Query ------------------> Feature Fusion
                                         |
                                         v
                                      MLP Head
                                         |
                         -------------------------------
                         |                             |
                      Delta xyz                Delta attenuation
                         |                             |
                         v                             v
                  Refined Gaussian Initialization
                                         |
                                         v
                              Original GSRF Optimization
                                         |
                                         v
                              RF Spectrum Reconstruction
```

## 14. 当前投稿定位

这个工作适合放在：

```text
Machine Learning & Models
    - Neural Networks & Deep Learning
    - Generative AI

Vision & Perception
    - Computer Vision

Intelligent Systems & Applications
    - Internet of Things
```

如果只能选最贴切的方向，建议优先：

```text
Neural Networks & Deep Learning
```

因为本文核心是用 neural representation / triplane prior 改进 RF field reconstruction。

