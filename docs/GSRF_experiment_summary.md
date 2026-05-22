# GSRF TriPlane / Wavelet 实验整理

更新时间：2026-05-22 10:28 CST  
远端工程目录：`/home/wys/GSRF`

本文档整理当前已经完成和正在运行的 RFID / GSRF 实验，包括实验目录、实验步骤、主要参数、结果表、阶段性结论和下一步建议。

---

## 1. 实验目标

当前研究问题是：在 GSRF 的 RFID spectrum reconstruction 中，是否可以通过 TriPlane 或 Wavelet-TriPlane 初始化，为 Gaussian attenuation / RF field 提供更好的空间先验。

最初想法是：

1. 原始 GSRF 初始化可能较弱。
2. TriPlane 可以作为空间 feature field，辅助初始化 Gaussian。
3. RF 场景中传播具有多尺度结构：大尺度 path loss / smooth propagation，小尺度 multipath / fading。
4. 因此参考 TriNeRFLet，尝试用 Haar wavelet triplane，把 plane 参数化为低频 LL 与高频 LH/HL/HH。
5. 同时尝试 gating-only，即只通过 triplane 预测 attenuation delta，不直接移动 Gaussian 位置。

当前实验已经说明：

1. `gating_only + direct triplane` 在 10K 有一定收益。
2. 但 30K 充分训练后，baseline 仍然最好。
3. 当前 Haar wavelet triplane 没有超过 direct gate / baseline。
4. 提高 direct triplane 分辨率到 128 反而更差。
5. 新增的 bbox pad 0.05 接近 direct gate，但没有超过。

---

## 2. 远端目录结构

工程根目录：

```bash
/home/wys/GSRF
```

训练与推理日志目录：

```bash
/home/wys/GSRF/logs/rfid_experiment_runs/
```

每个实验的 checkpoint、inference 结果目录：

```bash
/home/wys/GSRF/logs/rfid/<EXP_NAME>/
```

典型目录结构：

```bash
logs/rfid/<EXP_NAME>/chkpnt7000.pth
logs/rfid/<EXP_NAME>/chkpnt10000.pth
logs/rfid/<EXP_NAME>/chkpnt20000.pth
logs/rfid/<EXP_NAME>/chkpnt30000.pth
logs/rfid/<EXP_NAME>/inference_iter30000/summary.json
logs/rfid_experiment_runs/<EXP_NAME>_train.log
logs/rfid_experiment_runs/<EXP_NAME>_infer_30000.log
```

核心代码与配置文件：

```bash
/home/wys/GSRF/scene/triplane_initializer.py
/home/wys/GSRF/arguments/configs/rfid/exp_triplane.yaml
/home/wys/GSRF/run_rfid_triplane_experiments.sh
```

---

## 3. 当前代码改动

### 3.1 Haar Wavelet TriPlane

文件：

```bash
/home/wys/GSRF/scene/triplane_initializer.py
```

已加入 `HaarWaveletTriPlaneField`，核心思想是：

1. 不直接优化 feature plane。
2. 优化 Haar wavelet coefficients。
3. 通过 inverse wavelet transform 重建三张 plane：`P_xy`、`P_xz`、`P_yz`。
4. 低频 `LL` 表示粗尺度结构。
5. 高频 `LH/HL/HH` 表示局部细节。

相关参数：

```bash
--triplane_field_type haar
--triplane_wavelet_levels 2
--triplane_wavelet_l1 0.001
--triplane_wavelet_c2f
```

### 3.2 c2f 修复

之前的 coarse-to-fine active level 调度有问题：对于 `levels=2, warmup=300`，高频层基本到最后才激活，导致 Haar 训练早期很差。

已修复为：

```python
active_levels = min(
    wavelet_levels,
    ((step - 1) * (wavelet_levels + 1)) // max(1, warmup_iters),
)
```

修复后 10K Haar 不再崩，但 30K 仍没有超过 baseline / direct gate。

### 3.3 bbox pad 参数

新增参数：

```bash
--triplane_bbox_pad_scale
```

配置位置：

```bash
/home/wys/GSRF/arguments/configs/rfid/exp_triplane.yaml
```

默认值：

```yaml
triplane_bbox_pad_scale: 0.0
```

代码位置：

```bash
/home/wys/GSRF/scene/triplane_initializer.py
```

逻辑：

```python
base_xyz = gaussians.get_xyz.detach()
xyz_min = base_xyz.min(dim=0).values
xyz_max = base_xyz.max(dim=0).values
bbox_pad_scale = float(getattr(model_args, "triplane_bbox_pad_scale", 0.0))
if bbox_pad_scale > 0.0:
    bbox_pad = (xyz_max - xyz_min) * bbox_pad_scale
    xyz_min = xyz_min - bbox_pad
    xyz_max = xyz_max + bbox_pad
```

含义：

1. `triplane_resolution` 控制 plane 有多少格子。
2. `triplane_bbox_pad_scale` 控制 plane 覆盖的物理空间范围。
3. `pad=0.05` 表示在 Gaussian bbox 每个方向两端各扩 5%。

---

## 4. 通用实验设置

默认数据 split：

```bash
ratio_train=0.8
seed=8371
train_index_path=train_index_default_seed8371.txt
test_index_path=test_index_default_seed8371.txt
```

测试样本数量：

```text
num_test_samples = 1225
```

默认配置：

```bash
arguments/configs/rfid/exp1.yaml
arguments/configs/rfid/exp_triplane.yaml
```

注意：当前 `exp_triplane.yaml` 中默认写着：

```yaml
triplane_resolution: 128
```

因此复现实验中最好的 direct gate 时，必须显式加：

```bash
--triplane_resolution 64
```

否则会默认跑成 128。

---

## 5. 实验命令模板

### 5.1 查看正在运行的训练/推理

```bash
cd /home/wys/GSRF
ps -ef | grep -E 'train_rfid.py|inference_rfid.py' | grep -v grep
```

### 5.2 查看训练日志关键行

```bash
cd /home/wys/GSRF
grep -n 'Evaluating test\|Evaluating train\|Saving Checkpoint\|Training complete\|Traceback\|Error' \
  logs/rfid_experiment_runs/<EXP_NAME>_train.log | tail -n 80
```

### 5.3 查看 inference summary

```bash
cd /home/wys/GSRF
cat logs/rfid/<EXP_NAME>/inference_iter30000/summary.json
```

### 5.4 手动跑 inference

```bash
cd /home/wys/GSRF
conda activate gsrf

EXP=<EXP_NAME>

PYTHONUNBUFFERED=1 python inference_rfid.py \
  --config arguments/configs/rfid/exp_triplane.yaml \
  --exp_name "$EXP" \
  --iterations 30000 \
  --ratio_train 0.8 \
  --train_index_path train_index_default_seed8371.txt \
  --test_index_path test_index_default_seed8371.txt \
  2>&1 | tee logs/rfid_experiment_runs/${EXP}_infer_30000.log

cat logs/rfid/${EXP}/inference_iter30000/summary.json
```

---

## 6. 已完成实验结果总表

所有数值来自：

```bash
logs/rfid/<EXP_NAME>/inference_iter*/summary.json
```

| 实验名 | Iter | PSNR | MSE | SSIM | 结果目录 |
|---|---:|---:|---:|---:|---|
| `baseline_default_7k` | 7000 | 19.0910 | 0.015880 | 0.7255 | `logs/rfid/baseline_default_7k` |
| `triplane_default_7k` | 7000 | 18.4649 | 0.017646 | 0.7024 | `logs/rfid/triplane_default_7k` |
| `triplane_safe_default_7k` | 7000 | 19.0179 | 0.016029 | 0.7220 | `logs/rfid/triplane_safe_default_7k` |
| `triplane_film_default_7k` | 7000 | 18.9813 | 0.016227 | 0.7214 | `logs/rfid/triplane_film_default_7k` |
| `baseline_default_10k` | 10000 | 19.6692 | 0.014311 | 0.7464 | `logs/rfid/baseline_default_10k` |
| `triplane_film_default_10k` | 10000 | 19.5572 | 0.014452 | 0.7455 | `logs/rfid/triplane_film_default_10k` |
| `triplane_gate_default_10k` | 10000 | **19.9350** | **0.013468** | **0.7552** | `logs/rfid/triplane_gate_default_10k` |
| `triplane_gate_mlp_default_10k` | 10000 | 19.4781 | 0.014565 | 0.7413 | `logs/rfid/triplane_gate_mlp_default_10k` |
| `triplane_gate_notriplane_default_10k` | 10000 | 19.8043 | 0.013900 | 0.7526 | `logs/rfid/triplane_gate_notriplane_default_10k` |
| `triplane_gate_haar_c2ffix_default_10k` | 10000 | 19.5650 | 0.014463 | 0.7469 | `logs/rfid/triplane_gate_haar_c2ffix_default_10k` |
| `triplane_gate_haar_plain_default_10k` | 10000 | 19.6325 | 0.014202 | 0.7480 | `logs/rfid/triplane_gate_haar_plain_default_10k` |
| `baseline_default_30k` | 30000 | **22.0436** | **0.009435** | **0.8166** | `logs/rfid/baseline_default_30k` |
| `triplane_gate_default_30k` | 30000 | 21.9598 | 0.009612 | 0.8141 | `logs/rfid/triplane_gate_default_30k` |
| `triplane_gate_notriplane_default_30k` | 30000 | 21.9593 | 0.009588 | 0.8153 | `logs/rfid/triplane_gate_notriplane_default_30k` |
| `triplane_gate_haar_default_30k` | 30000 | 21.9527 | 0.009550 | 0.8140 | `logs/rfid/triplane_gate_haar_default_30k` |
| `triplane_gate_haar_c2ffix_default_30k` | 30000 | 21.8159 | 0.009795 | 0.8117 | `logs/rfid/triplane_gate_haar_c2ffix_default_30k` |
| `triplane_gate_res128_default_30k` | 30000 | 21.8020 | 0.009772 | 0.8104 | `logs/rfid/triplane_gate_res128_default_30k` |
| `triplane_gate_res64_pad005_default_30k` | 30000 | 21.9347 | 0.009616 | 0.8147 | `logs/rfid/triplane_gate_res64_pad005_default_30k` |

---

## 7. 最新 bbox pad 实验

### 7.1 `triplane_gate_res64_pad005_default_30k`

目的：测试 plane 物理范围是否影响 direct gate。  
设置：`resolution=64`，`bbox_pad_scale=0.05`。

训练命令核心参数：

```bash
--triplane_output_mode gating_only
--triplane_field_type direct
--triplane_resolution 64
--triplane_bbox_pad_scale 0.05
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

训练日志位置：

```bash
logs/rfid_experiment_runs/triplane_gate_res64_pad005_default_30k_train.log
```

训练过程结果：

| Iter | Train/Test | L1 | PSNR |
|---:|---|---:|---:|
| 7000 | test | 0.115124 | 17.3744 |
| 7000 | train | 0.065843 | 22.0514 |
| 10000 | test | 0.101611 | 18.2083 |
| 10000 | train | 0.092784 | 18.9349 |
| 20000 | test | 0.084210 | 19.6708 |
| 20000 | train | 0.051494 | 24.7905 |
| 30000 | test | 0.080405 | 21.0362 |
| 30000 | train | 0.051968 | 23.9945 |

正式 inference summary：

```text
PSNR = 21.9347
MSE  = 0.009616
SSIM = 0.8147
```

对比：

```text
triplane_gate_default_30k:              21.9598
triplane_gate_res64_pad005_default_30k: 21.9347
baseline_default_30k:                   22.0436
```

结论：

```text
pad005 比训练过程 report 看起来好很多，但仍略低于 res64 direct gate。
bbox_pad_scale=0.05 可以作为尺度敏感性 ablation，但不适合作为主结果。
```

---

## 8. 分阶段实验整理

### 8.1 Stage A：7K 初步验证

目的：测试最原始的 triplane 初始化是否能在短训练下改善 baseline。

实验：

```bash
baseline_default_7k
triplane_default_7k
triplane_safe_default_7k
triplane_film_default_7k
```

结论：

1. `triplane_default_7k` 明显差于 baseline。
2. `safe` 和 `film` 可以缓解，但没有超过 baseline。
3. 说明直接加入 triplane 初始化并不自然有效，需要约束输出方式。

### 8.2 Stage B：10K gating-only

目的：只让 triplane 预测 attenuation delta，不直接移动 Gaussian 位置，降低破坏几何结构的风险。

关键设置：

```bash
--triplane_output_mode gating_only
--triplane_field_type direct
--triplane_resolution 64
--triplane_decoder_type film_fourier
--triplane_hidden_dim 128
--triplane_warmup_iters 300
--triplane_offset_radius_scale 0.0
--triplane_offset_l2 0.0
--triplane_att_radius 0.05
--triplane_att_l2 0.02
```

结果：

```text
baseline_default_10k:      19.6692
triplane_gate_default_10k: 19.9350
```

结论：

1. `gating_only + film_fourier + direct triplane` 在 10K 有收益。
2. 这是目前最正向的结果。
3. 但这个收益没有延续到 30K 最终结果。

### 8.3 Stage C：decoder ablation

实验：

```bash
triplane_gate_default_10k
triplane_gate_mlp_default_10k
```

结果：

```text
film_fourier: 19.9350
mlp:          19.4781
```

结论：

```text
Fourier-FiLM decoder 明显优于普通 MLP decoder。
```

### 8.4 Stage D：no-triplane gating 对照

实验：

```bash
triplane_gate_notriplane_default_10k
triplane_gate_notriplane_default_30k
```

结果：

```text
10K no-triplane gating: 19.8043
30K no-triplane gating: 21.9593
30K direct gate:        21.9598
```

结论：

1. 10K 时 direct triplane 比 no-triplane gating 高约 0.13 dB。
2. 30K 时二者几乎完全一样。
3. 这说明当前 direct triplane 的最终贡献很弱。
4. 也说明目前收益可能主要来自 gating-only / training schedule，而不是 plane 表示本身。

### 8.5 Stage E：Haar Wavelet TriPlane

实验：

```bash
triplane_gate_haar_default_30k
triplane_gate_haar_c2ffix_default_10k
triplane_gate_haar_c2ffix_default_30k
triplane_gate_haar_plain_default_10k
```

结果：

```text
triplane_gate_haar_default_30k:        21.9527
triplane_gate_haar_c2ffix_default_30k: 21.8159
triplane_gate_haar_plain_default_10k:  19.6325
```

结论：

1. Haar 没有超过 direct gate。
2. Haar 也没有超过 baseline。
3. c2f 修复能改善早期崩坏，但最终 30K 仍较差。
4. 当前 Haar wavelet 参数化不能作为主结果。

### 8.6 Stage F：direct triplane 分辨率

实验：

```bash
triplane_gate_default_30k      # resolution=64
triplane_gate_res128_default_30k
```

结果：

```text
res64 direct gate:  21.9598
res128 direct gate: 21.8020
```

结论：

1. 提高 plane resolution 没有帮助。
2. `128` 比 `64` 更差。
3. 说明问题不是 spatial grid 不够细。
4. 更高分辨率可能增加自由度，导致 warmup 学到更噪的 attenuation prior。
5. 不建议继续跑 `256`。

### 8.7 Stage G：bbox pad / plane physical extent

实验：

```bash
triplane_gate_res64_pad005_default_30k
```

目的：

```text
测试 plane 的物理覆盖范围是否太紧，导致坐标 clamp 到边界，影响 TX/RX 或边缘点采样。
```

正式 inference 结果：

```text
30K inference PSNR = 21.9347
MSE = 0.009616
SSIM = 0.8147
```

结论：

```text
pad=0.05 接近 direct gate，但没有超过 direct gate 和 baseline。
可以证明 plane physical extent 会影响结果，但不是当前突破口。
```

---

## 9. 关键结论

### 9.1 当前最强结论

```text
gating-only 在 10K 有早期收益，但当前 triplane 表示没有带来稳定的 30K 最终提升。
```

证据：

```text
baseline_default_10k:      19.6692
triplane_gate_default_10k: 19.9350
```

但：

```text
baseline_default_30k:      22.0436
triplane_gate_default_30k: 21.9598
```

### 9.2 direct triplane 当前不是足够强的主方法

证据：

```text
triplane_gate_default_30k:          21.9598
triplane_gate_notriplane_default_30k: 21.9593
```

这两个几乎一样，说明 plane 的 contribution 很弱。

### 9.3 Haar wavelet 当前没有成功

证据：

```text
triplane_gate_haar_default_30k:        21.9527
triplane_gate_haar_c2ffix_default_30k: 21.8159
baseline_default_30k:                  22.0436
```

### 9.4 提高 resolution 是负方向

证据：

```text
res64 direct gate:  21.9598
res128 direct gate: 21.8020
```

### 9.5 bbox pad 0.05 没有带来提升

证据：

```text
res64 direct gate: 21.9598
res64 pad005:      21.9347
baseline:          22.0436
```

---

## 10. 非常重要的比较注意事项

不要直接比较：

```text
某个 30K 训练日志中的 10K evaluation
```

和：

```text
单独训练 10K 得到的 inference summary
```

原因：

```python
args.densify_until_iter = args.iterations // 2
args.position_lr_max_steps = args.iterations
```

也就是说：

1. `--iterations 10000` 和 `--iterations 30000` 会改变 densification schedule。
2. position learning rate schedule 也不同。
3. 所以 10K-only run 与 30K run 的中间 10K 点不是严格同一个实验。

正式比较应优先使用：

```bash
logs/rfid/<EXP_NAME>/inference_iterXXXX/summary.json
```

---

## 11. 当前实验复现命令

### 11.1 复现最好的 10K direct gate

```bash
cd /home/wys/GSRF
conda activate gsrf

GPU=0 ./run_rfid_triplane_experiments.sh triplane_gate10k
```

### 11.2 复现 30K baseline 与 direct gate

baseline：

```bash
cd /home/wys/GSRF
conda activate gsrf
GPU=0 ./run_rfid_triplane_experiments.sh final_default30k
```

注意：这个脚本里的 `final_default30k` 是 baseline + 原始 triplane default，不一定等于 gating-only direct gate。

手动跑 gating-only direct gate：

```bash
cd /home/wys/GSRF
conda activate gsrf
mkdir -p logs/rfid_experiment_runs

export CUDA_VISIBLE_DEVICES=0
EXP=triplane_gate_default_30k

PYTHONUNBUFFERED=1 python train_rfid.py \
  --config arguments/configs/rfid/exp_triplane.yaml \
  --exp_name "$EXP" \
  --iterations 30000 \
  --ratio_train 0.8 \
  --train_index_path train_index_default_seed8371.txt \
  --test_index_path test_index_default_seed8371.txt \
  --triplane_output_mode gating_only \
  --triplane_field_type direct \
  --triplane_resolution 64 \
  --triplane_decoder_type film_fourier \
  --triplane_hidden_dim 128 \
  --triplane_fourier_freqs 4 \
  --triplane_film_layers 3 \
  --triplane_warmup_iters 300 \
  --triplane_offset_radius_scale 0.0 \
  --triplane_offset_l2 0.0 \
  --triplane_att_radius 0.05 \
  --triplane_att_l2 0.02 \
  --triplane_finalize_tx_samples 32 \
  2>&1 | tee logs/rfid_experiment_runs/${EXP}_train.log

PYTHONUNBUFFERED=1 python inference_rfid.py \
  --config arguments/configs/rfid/exp_triplane.yaml \
  --exp_name "$EXP" \
  --iterations 30000 \
  --ratio_train 0.8 \
  --train_index_path train_index_default_seed8371.txt \
  --test_index_path test_index_default_seed8371.txt \
  2>&1 | tee logs/rfid_experiment_runs/${EXP}_infer_30000.log
```

### 11.3 复现 Haar c2f 30K

```bash
cd /home/wys/GSRF
conda activate gsrf
GPU=0 ./run_rfid_triplane_experiments.sh triplane_gate_haar30k
```

### 11.4 复现 res128 direct gate

```bash
cd /home/wys/GSRF
conda activate gsrf
mkdir -p logs/rfid_experiment_runs

export CUDA_VISIBLE_DEVICES=0
EXP=triplane_gate_res128_default_30k

PYTHONUNBUFFERED=1 python train_rfid.py \
  --config arguments/configs/rfid/exp_triplane.yaml \
  --exp_name "$EXP" \
  --iterations 30000 \
  --ratio_train 0.8 \
  --train_index_path train_index_default_seed8371.txt \
  --test_index_path test_index_default_seed8371.txt \
  --triplane_output_mode gating_only \
  --triplane_field_type direct \
  --triplane_resolution 128 \
  --triplane_decoder_type film_fourier \
  --triplane_hidden_dim 128 \
  --triplane_fourier_freqs 4 \
  --triplane_film_layers 3 \
  --triplane_warmup_iters 300 \
  --triplane_offset_radius_scale 0.0 \
  --triplane_offset_l2 0.0 \
  --triplane_att_radius 0.05 \
  --triplane_att_l2 0.02 \
  --triplane_finalize_tx_samples 32 \
  2>&1 | tee logs/rfid_experiment_runs/${EXP}_train.log
```

### 11.5 复现 bbox pad 0.05

```bash
cd /home/wys/GSRF
conda activate gsrf
mkdir -p logs/rfid_experiment_runs

export CUDA_VISIBLE_DEVICES=0
EXP=triplane_gate_res64_pad005_default_30k

PYTHONUNBUFFERED=1 python train_rfid.py \
  --config arguments/configs/rfid/exp_triplane.yaml \
  --exp_name "$EXP" \
  --iterations 30000 \
  --ratio_train 0.8 \
  --train_index_path train_index_default_seed8371.txt \
  --test_index_path test_index_default_seed8371.txt \
  --triplane_output_mode gating_only \
  --triplane_field_type direct \
  --triplane_resolution 64 \
  --triplane_bbox_pad_scale 0.05 \
  --triplane_decoder_type film_fourier \
  --triplane_hidden_dim 128 \
  --triplane_fourier_freqs 4 \
  --triplane_film_layers 3 \
  --triplane_warmup_iters 300 \
  --triplane_offset_radius_scale 0.0 \
  --triplane_offset_l2 0.0 \
  --triplane_att_radius 0.05 \
  --triplane_att_l2 0.02 \
  --triplane_finalize_tx_samples 32 \
  2>&1 | tee logs/rfid_experiment_runs/${EXP}_train.log
```

---

## 12. 下一步建议

### 12.1 不建议继续跑的方向

不建议继续跑：

```bash
--triplane_resolution 256
```

原因：

```text
res128 已经比 res64 更差，res256 大概率只会更慢、更噪。
```

也不建议把 `bbox_pad_scale=0.10` 作为优先方向。`pad005` 的正式 inference 虽然接近 direct gate，但没有超过，因此继续扫 bbox pad 的收益预期不高。

### 12.2 论文层面更有价值的方向

当前结果反而可以支持一个更清楚的研究叙事：

```text
Naive/high-resolution triplane is insufficient for RF Gaussian fields.
The useful signal is not simply more spatial capacity, but how RF-aware priors are injected.
```

也就是说，不要把论文写成：

```text
We add triplane initialization to GSRF.
```

应该写成：

```text
We investigate triplane-based RF Gaussian initialization and find that naive spatial feature planes are insufficient. 
We then study more constrained gating and multiscale regularization as RF-aware priors.
```

如果要继续冲 CCF C short paper，建议补三类实验：

1. Wavelet subband error analysis  
   比较 predicted / GT spectrum 的 LL error 和 HF error，看方法是否真的改善高频 multipath。

2. Sparse measurement setting  
   在 20% 或 sparse220 训练样本上比较 baseline / gating / Haar / no-triplane。

3. 更合理的小波基  
   Haar 太粗，可以尝试 bior4.4 / sym4 / db4，但要谨慎控制实验规模。

### 12.3 最小可投稿实验包

如果时间紧，建议保留：

1. Full setting 30K 主表：
   `baseline_default_30k`、`triplane_gate_default_30k`、`triplane_gate_notriplane_default_30k`、`triplane_gate_haar_default_30k`。

2. Early convergence 表：
   `baseline_default_10k`、`triplane_gate_default_10k`、`triplane_gate_notriplane_default_10k`。

3. Negative ablation：
   `res128`、`Haar c2f`、`bbox pad`，说明 naive capacity / naive wavelet 并不自动有效。

4. RF-aware analysis：
   wavelet subband error 或 sparse measurement。

---

## 13. 一句话总结

当前实验结论是：

```text
gating-only 确实带来 10K early convergence improvement，但当前 direct triplane / Haar wavelet / high-resolution / bbox pad 都没有形成 30K 最终优势。
```

所以下一步不应该继续盲目加 plane capacity，而应该转向：

```text
RF-aware analysis + sparse setting + 更合理的 multiscale prior
```

这样论文故事会比单纯调 triplane 更稳。
