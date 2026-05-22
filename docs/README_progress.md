# gsrf_wavelet_research: 研究计划与进度速览 (2026-05-22)

## 1. 计划主线（一句话）

目标不是“dense 上 PSNR 微涨”，而是：

> 用 triplane + TX/RX 传播几何 +（可选）wavelet 多尺度证据，做 candidate-wise 的 RF-aware Gaussian 初始化，从而在 sparse/泛化场景里更稳、更省测量/更快收敛。

## 2. 当前已验证的关键现象

- `gating-only`（固定 xyz，只学 attenuation 修正）比 `offset+att` 稳定，且在 default split 的 10k 上超过 baseline。
- 只做 “Haar wavelet triplane 参数化（wavelet coeff -> IWT -> planes）”目前收益不稳定；修 c2f 调度可以止损，但仍不如 direct gating。
- 20% train / 20k full inference 结果也很弱：baseline `20.4743`，direct gate `20.4528`，bior4.4 gate `20.5052`。bior4.4 只比 baseline 高约 `+0.031 dB`，不足以支撑论文主贡献。

这两点把“下一步应该卷哪里”指向得很明确：不要继续把主要赌注压在 plane 参数化，而要回到 candidate-wise 初始化器的核心配方（keep_prob / 观测 conditioning / 指标与机制证明）。

20%/20k 的结果进一步说明：仅靠 triplane/wavelet gate 初始化 attenuation，哪怕换成 bior4.4，也没有把 sparse 场景拉开差距。

## 3. 还没做但最值得优先做的点

详见 `untried_high_value_ideas.md`，优先级从高到低：

- candidate-wise `keep_prob` / top-K / budget（把方法升级为“会选点、会分配 capacity”）
- 加入 RF 观测 encoder（spectrum/CSI -> global code conditioning）
- wavelet 用在“观测侧指标/损失/训练调度”，先别执着于 wavelet-triplane 表示
- bounded offset 的小半径 sweep（和 keep_prob 联动），而不是一上来就大位移

## 4. 你接下来跑 sparse 的推荐顺序

- 20% train / 20k：baseline vs gating-only（先确认 sparse 下是否真的更有意义）
- 20% train / 20k：gating-only + keep_prob（如果这个有效，论文贡献立刻变硬）
- 20% train / 20k：gating-only + spectrum/CSI conditioning（验证“观测证据”的必要性）
- 增加 DWT subband 指标（LL/HF error）作为机制解释与图表

## 5. 关键参考文档入口

- 总计划与定位：`research_plan_pricai_short.md`
- 方法分类与实验矩阵：`methodology_and_experiment_plan.md`
- triplane-first 版本计划：`triplane_first_methodology_and_experiment_plan.md`
- 当前代码与结果最贴近解释：`triplane_film_vs_gating_only.md`
