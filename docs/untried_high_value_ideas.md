# GSRF Wavelet/Triplane: 计划回顾与未尝试高价值点 (2026-05-22)

这份文档回答两个问题：

1. `gsrf_wavelet_research` 里“原研究计划”到底想做什么（主线是什么）。
2. 对照目前已经做过的实验/实现，还有哪些“参考价值很大但还没真正尝试”的点，且值得优先做。

## 1. 原研究计划主线 (从文档抽取)

来自：

- `research_plan_pricai_short.md`
- `methodology_and_experiment_plan.md`
- `triplane_first_methodology_and_experiment_plan.md`

计划的“最小可投稿闭环”不是“换一种 triplane 参数化”，而是：

- 发现问题：GSRF 的初始化是 RF-blind（均匀/立方体撒点，不感知 TX/RX 传播几何与多尺度结构）。
- 方法形态：candidate-wise 的初始化器（从已有 candidate center 出发），输出 `bounded offset + keep_prob/opacity`，并能接入 RF 观测（CSI/spectrum）的多尺度证据。
- wavelet 的角色：不是单纯加一个 loss，而是提供 coarse-to-fine 的 RF 证据（LL 表示大尺度趋势，HF 表示小尺度 multipath/fading），用于 conditioning / regularization / 训练调度。
- 实验主线：在 full/dense 与 sparse 采样下都要说明 “初始化效率/稳定性/泛化” 或 “measurement efficiency”，而不仅是 dense 上 PSNR 提一点点。

## 2. 目前已经做过/基本确认过的事实

从 `triplane_film_vs_gating_only.md` 的对照与现有实验观察，已经比较明确的结论是：

- `TriPlane + geometry + FiLM-Fourier decoder` 的初始化 warmup 是可行的。
- “同时预测 `delta_xyz + delta_attenuation`”在当前 RFID 任务上比 baseline 更不稳，甚至可能更差。
- “gating-only（固定 xyz，只预测 attenuation 修正）”在 80/20 默认 split 的 10k 上能超过 baseline（并且比 offset+att 更稳）。
- “Haar wavelet triplane（对 plane 用 wavelet 系数参数化 + IWT）”在 10k 上并没有成为主方法；修 c2f 调度能把它从很差救回，但仍未超过 direct gate。

这意味着：如果后续还沿着 “只改 plane 参数化” 去卷，很容易陷入增量；要回到计划主线里更“像论文贡献”的部分。

## 3. 还没尝试但很高价值的方向 (按优先级)

下面这些都在计划文档里出现过，但从目前实验链路来看尚未系统验证或根本没实现。

### P0. candidate-wise `keep_prob` / top-K pruning (把方法从“调 attenuation”升级为“学会选点”)

为什么值钱：

- 你们当前 gating-only 其实是在“所有 Gaussians 都保留”的前提下做软调制。
- 在 sparse 场景，关键往往不是把每个点调准，而是“更快、更稳地把 capacity 放对地方”，`keep_prob` 和 pruning/预算正好对齐这个贡献点。
- 这也是计划里最像 pixelSplat/MVSplat 的“稳定配方”：candidate -> score -> keep/move（move 可以先不做）。

最小实现建议（不改 renderer）：

- warmup 仍然用 `override_xyz` 固定；
- 新增一个 `keep_logit` 输出，用它生成 `opacity_init` 或 `attenuation_scale` 的 gate；
- 加一个很轻的 budget 正则（例如 `L_budget = mean(sigmoid(keep_logit))` 或 target-K）；
- inference/训练日志里报告 “有效高权重点比例” 或 “最终可视化的 keep 分布”。

最小对照实验（建议先在 20% train 上做）：

- baseline
- gating-only（已有）
- gating-only + keep_prob（新）
- gating-only + keep_prob + budget（新）

### P0. 加入 RF 观测的 encoder（Spectrum/CSI -> conditioning code），而不是只靠 geometry

为什么值钱：

- 现在 triplane initializer 主要靠 geometry + learnable spatial prior；它不看“观测长什么样”。
- 计划主线强调：RF 观测提供了 multi-scale 证据，尤其 sparse 时更关键。

最小实现建议：

- 不需要上来就做复杂的 WTConv/MWCNN。
- 先做一个非常小的 encoder：`spectrum (或 CSI) -> global code z`，接到 decoder 输入（concat 或 FiLM）。
- 如果是 complex CSI：做 `ComplexFusion1x1`（Re/Im 先 1x1 融合）再进 encoder。

最小对照实验：

- geometry-only gating
- geometry + spectrum-code gating

如果这个都没收益，再谈 wavelet 化就更有根据。

### P0. wavelet 用在“观测侧的损失/指标”而非 “plane 参数化”

为什么值钱：

- 你们尝试的 Haar-triplane 是 “representation side wavelet”，工程复杂度高且目前收益不稳定。
- 计划里另一条更稳的线是 “supervision side wavelet”：用 DWT subband error 报告 HF 改善，或者用 coarse-to-fine HF gate 做训练稳定性。

最小落地建议：

- 不改变模型结构，先把评估指标补齐：对预测 spectrum 和 GT 做 2D DWT，报告 `LL-MSE` 与 `HF-MSE`（LH/HL/HH 的和）。
- 如果 HF 明显改善，可以作为主要图表之一；如果 HF 没改善，也能解释为什么 wavelet-triplane 方向不 work。

### P1. bounded offset “只在很小半径”里做（并配合 keep_prob），而不是直接学大位移

为什么值钱：

- 现在 offset+att 更差，很可能来自“RF 对位置太敏感”导致局部陷阱。
- 计划里提到 `offset_radius * tanh(delta)` 这种 bounded offset，但需要把半径 sweep 做完，并与 keep_prob 联动（先选点，再小修）。

建议的最小 sweep：

- `offset_radius ∈ {0.25, 0.5, 1.0} * voxel`（以你们现有 bbox/plane 分辨率为基准）
- `offset` 只在 warmup 前 N steps 打开，后续 freeze 或者只让一部分点 move（由 keep_prob 决定）。

### P1. 预测更多 Gaussian 属性（scale / rotation / RF coeff 初始化）

为什么值钱：

- 计划里明确写了：初始化器不仅能调 `xyz`，还可以预测 `scale/opacity/coeff`，这更像“学到了 RF prior”。
- 但这项实现风险更高，建议放在 P1：先把 keep_prob + 观测 conditioning 跑通。

### P2. wavelet-guided densification / sampling（从 WaveNeRF 的“HF cue 导向”借鉴）

为什么值钱：

- 这是把 “wavelet” 从表面装饰变成“优化策略”：HF 残差大就多分配 Gaussians/迭代预算。
- 但要改 densify/prune 逻辑，属于更大工程量，建议在前面 P0/P1 收益不足时再做。

## 4. 我建议你们下一阶段的优先实验（符合你说的 sparse 方向）

因为你已经判断 dense 上提升空间不大，下一阶段建议直接转到 sparse（例如 20% train），但要避免一次塞太多变量。

建议顺序：

1. 20% train / 20k：baseline vs gating-only（确认 sparse 下 gating-only 是否更稳/更好）
2. 20% train / 20k：gating-only + keep_prob（看是否显著改善稳定性与 final）
3. 20% train / 20k：gating-only + spectrum/CSI global code conditioning（看“观测侧证据”是否带来收益）
4. 加 DWT subband 指标（LL/HF error）作为论文图表与诊断

## 4.1 可直接复制的运行指令模板（含选 GPU）

下面指令默认你在远端仓库根目录（例如 `/home/wys/GSRF`），并且脚本/路径与当前版本一致。

### A) 用现成脚本跑（推荐，用于 default 0.8 或 sparse220）

选显卡的方法是设置环境变量 `GPU`（脚本内部会转成 `CUDA_VISIBLE_DEVICES`）：

```bash
cd /home/wys/GSRF
conda activate gsrf

# GPU=0 代表用 0 号卡
GPU=0 ./run_rfid_triplane_experiments.sh triplane_gate_haar30k

# 并行跑三个实验时，换 GPU 即可
GPU=0 ./run_rfid_triplane_experiments.sh triplane_gate10k &
GPU=1 ./run_rfid_triplane_experiments.sh triplane_gate_mlp10k &
GPU=2 ./run_rfid_triplane_experiments.sh triplane_gate_haar10k &
wait
```

日志目录固定是：

- `logs/rfid_experiment_runs`

### B) 20% train / 20k：不改脚本的做法（直接跑 train_rfid.py + inference_rfid.py）

`run_rfid_triplane_experiments.sh` 目前内置的是 `DEFAULT_RATIO=0.8` 和 `sparse220`，没有显式提供 0.2 的 case。
因此最稳的方式是直接调用训练与推论脚本：

```bash
cd /home/wys/GSRF
conda activate gsrf
export CUDA_VISIBLE_DEVICES=0

# 20% split 的索引文件名建议单独起，避免和 0.8 混淆
RATIO=0.2
SEED=8371
TRAIN_IDX="train_index_ratio${RATIO}_seed${SEED}.txt"
TEST_IDX="test_index_ratio${RATIO}_seed${SEED}.txt"

# 1) baseline 20k
python train_rfid.py \
  --config arguments/configs/rfid/exp1.yaml \
  --exp_name baseline_ratio02_20k \
  --iterations 20000 \
  --ratio_train ${RATIO} \
  --train_index_path ${TRAIN_IDX} \
  --test_index_path ${TEST_IDX}

python inference_rfid.py \
  --config arguments/configs/rfid/exp1.yaml \
  --exp_name baseline_ratio02_20k \
  --iterations 20000 \
  --ratio_train ${RATIO} \
  --train_index_path ${TRAIN_IDX} \
  --test_index_path ${TEST_IDX}

# 2) gating-only 20k（direct triplane）
python train_rfid.py \
  --config arguments/configs/rfid/exp_triplane.yaml \
  --exp_name triplane_gate_ratio02_20k \
  --iterations 20000 \
  --ratio_train ${RATIO} \
  --train_index_path ${TRAIN_IDX} \
  --test_index_path ${TEST_IDX} \
  --triplane_output_mode gating_only \
  --triplane_field_type direct \
  --triplane_decoder_type film_fourier \
  --triplane_hidden_dim 128 \
  --triplane_fourier_freqs 4 \
  --triplane_film_layers 3 \
  --triplane_warmup_iters 300 \
  --triplane_offset_radius_scale 0.0 \
  --triplane_offset_l2 0.0 \
  --triplane_att_radius 0.05 \
  --triplane_att_l2 0.02 \
  --triplane_finalize_tx_samples 32

python inference_rfid.py \
  --config arguments/configs/rfid/exp_triplane.yaml \
  --exp_name triplane_gate_ratio02_20k \
  --iterations 20000 \
  --ratio_train ${RATIO} \
  --train_index_path ${TRAIN_IDX} \
  --test_index_path ${TEST_IDX}
```

说明：

- `--ratio_train/--train_index_path/--test_index_path` 三个参数要配套用，保证 train/infer 走同一个 split。
- 如果你担心 index 文件已存在但对应比例不对：直接删掉这两个 index 文件再重跑一次（确保重新生成）。

## 5. 你问“原来的研究计划在哪里”

主计划与方法路线都在这些文件里：

- `research_plan_pricai_short.md`
- `methodology_and_experiment_plan.md`
- `triplane_first_methodology_and_experiment_plan.md`
- `triplane_film_vs_gating_only.md`（包含目前最贴近代码与结果的解释）
