# Router Mismatch Sequence-RS Experiment Plan

## 2026-07-02 实验总记录

新增实验汇总文档：

```text
/inspire/hdd/project/qianghuaxuexi/hujiarui-25046/verl-sequence/low_precision/experiment_analysis/experiment_summary.md
```

内容包含当前 FP8 rollout / Megatron GRPO / router mismatch / sequence RS / TIS / overlap_fraction / debug 实验的环境版本、公共设置、实验矩阵、主要现象、解释假设、后续建议和论文资料。

## 2026-07-03 threshold0.25 + TIS-C2 resume200 dump 分析

新增分析文档：

```text
/inspire/hdd/project/qianghuaxuexi/hujiarui-25046/verl-sequence/low_precision/experiment_analysis/threshold0.25_TIS_resume200_analysis.md
```

对应聚合 JSON：

```text
/inspire/hdd/project/qianghuaxuexi/hujiarui-25046/verl-sequence/low_precision/experiment_analysis/threshold0.25_TIS_resume200_dump_analysis.json
```

主要结论：`seq_mismatch > 0.25` 在 step201-244 阶段过滤比例约 7.0%，但 rejected response 更短且正确率更高，prompt group 层面更偏 easy group，不是过滤 hard/wrong response；极端 token 没有显示出明确的后续 mismatch 级联影响。

## 2026-07-06 threshold0.25 + TIS-C2 resume200 dump 全量重算

覆盖更新：

```text
/inspire/hdd/project/qianghuaxuexi/hujiarui-25046/verl-sequence/low_precision/experiment_analysis/threshold0.25_TIS_resume200_analysis.md
/inspire/hdd/project/qianghuaxuexi/hujiarui-25046/verl-sequence/low_precision/experiment_analysis/threshold0.25_TIS_resume200_dump_analysis.json
```

本次基于 `router_analysis_dump_resume200` 当前全量 164 个 dump 文件重算，覆盖 step201-364，共 335872 条 response。主要结论：`threshold0.25 + TIS-C2` 后期 step301-364 仍有 length/correctness bias，rejected response 仍更短、accuracy/reward 更高；bias 相比早期减弱，但没有反转为优先过滤错误或困难 response。

## 2026-07-05 overlap 0.045RS vs 0.04RS 对比分析

新增分析文档：

```text
/inspire/hdd/project/qianghuaxuexi/hujiarui-25046/verl-sequence/low_precision/experiment_analysis/overlap_0.045_vs_0.04_comparison.md
```

对应聚合 JSON：

```text
/inspire/hdd/project/qianghuaxuexi/hujiarui-25046/verl-sequence/low_precision/experiment_analysis/overlap_threshold0.045RS_20260701_202845_analysis.json
/inspire/hdd/project/qianghuaxuexi/hujiarui-25046/verl-sequence/low_precision/experiment_analysis/overlap_threshold0.04RS_20260703_020122_analysis.json
```

主要结论：`0.04RS` 比 `0.045RS` 前期过滤更强，但没有阻止 train reward 上涨伴随 response length 下降的短答案化趋势；`0.04RS` 后期过滤比例暴涨并进入长输出/高截断/低 reward 状态。两组极端 token 均不是主要原因，val 先崩更像策略长度/格式/泛化偏移，train reward 是滞后信号。

## 实验目标

当前实验面向 Qwen3-30B-A3B MoE 的 **FP8 rollout + BF16/Megatron training** 场景，核心问题是缓解 rollout 端推理路由和训练端 old log prob forward 路由不一致带来的训练不稳定。

目标是设计一个基于 router mismatch 的 sequence-level rejection sampling：

1. 记录 rollout 端和训练端 MoE router 选择的一致性。
2. 将 token/layer 级 router mismatch 聚合成 response 级 `seq_mismatch`。
3. 当某条 response 的 `seq_mismatch` 超过阈值时，mask 掉整条 response，使其不参与后续 loss/advantage 更新。
4. 在尽量不改变主训练逻辑的前提下，控制额外记录和过滤逻辑的时间开销。

当前主脚本：

```bash
/inspire/hdd/project/qianghuaxuexi/hujiarui-25046/verl-sequence/low_precision/run_grpo_qwen3_moe_30b_megatron_fp8_rollout_baseline_local.sh
```

## 当前实验配置

主要配置：

- 算法：`algorithm.adv_estimator=grpo`
- 模型：`/inspire/hdd/project/qianghuaxuexi/public/models/Qwen3-30B-A3B`
- 训练引擎：Megatron
- Rollout 引擎：vLLM
- Rollout 量化：FP8
- TIS：关闭
- 原始 rollout correction RS：关闭
- Router mismatch metrics：开启
- Router mismatch sequence-RS：开启
- 训练前验证：开启

当前关键参数：

```bash
router.enable_mismatch_metrics=True
router.enable_mismatch_rs=True
router.mismatch_rs_threshold=0.3
router.mismatch_alignment_warmup_steps=1
```

含义：

- 第 1 个 step 在 `alignment=0/-1/1` 中选择最优对齐。
- 第 2 个 step 起固定使用第一步选出的 alignment。
- 对每条 response 计算 `seq_mismatch`。
- 若 `seq_mismatch > 0.3`，则过滤该 response。

## Router Mismatch 定义

当前专家匹配规则：

1. 对某个 token 的某一层，比较 rollout top8 experts 和 train top8 experts。
2. top8 expert id **不考虑顺序**，只要 expert id 集合一致，该层算 match。
3. 如果该层 top8 expert id 集合不一致，该层算 mismatch。

当前 sequence mismatch 定义：

```text
layer_mismatch[b,t,l] = 1[top8_set_rollout[b,t,l] != top8_set_train[b,t,l]]

token_mismatch[b,t] = mean_l(layer_mismatch[b,t,l])

seq_mismatch[b] =
    mean_{t where response_mask[b,t] = 1}(token_mismatch[b,t])
```

其中 `response_mask[b,t] = 1` 表示有效 response token，不包括 padding 或已经被其他逻辑 mask 的 token。

当前过滤逻辑：

```text
if seq_mismatch[b] > router.mismatch_rs_threshold:
    response_mask[b, :] = 0
```

也就是说，过滤单位是整条 response，不是单个 token。

## 当前记录指标

主要关注：

- `router/rollout_vs_train/response_token_match_rate`
- `router/rollout_vs_train/seq_mismatch_mean`
- `router/rollout_vs_train/seq_mismatch_max`
- `router/rollout_vs_train/seq_valid_token_count_mean`
- `router/rollout_vs_train/rs_threshold`
- `router/rollout_vs_train/rs_rejected_fraction`
- `router/rollout_vs_train/rs_rejected_count`
- `router/rollout_vs_train/rs_kept_count`
- `router/rollout_vs_train/selected_alignment`
- `router/rollout_vs_train/alignment_frozen`

用于定位开销的 timing：

- `timing_s/old_log_prob`
- `timing_s/old_log_prob_forward`
- `timing_s/router_mismatch_metrics`
- `timing_s/router_mismatch_rs`

解释：

- `old_log_prob_forward` 是 actor 端重新 forward 计算 old log prob，并返回 actor routed experts。
- `router_mismatch_metrics` 是 rollout/train routed experts 比较和 `seq_mismatch` 计算。
- `router_mismatch_rs` 是根据 `seq_mismatch` 修改 `response_mask` 的过滤过程。

## 已完成改进

### 1. 从 ordered top8 match 改为 unordered top8 set match

原因：

MoE top8 expert 的顺序不应影响“专家集合是否一致”的判断。只要两端选择的 expert id 集合相同，即使顺序不同，也应认为这一层 match。

当前做法：

- 对 top8 expert id 做排序。
- 排序后逐位比较。
- 完全一致则该层 match。

### 2. 引入 sequence-level mismatch 和 RS

原因：

原始 `response_token_match_rate` 过于严格，它要求某个 token 的所有 MoE 层全部一致才算 token match。对于 48 层 MoE，这个指标会非常低，不适合作为直接过滤阈值。

当前做法：

- 先计算每个 token 有多少比例的 MoE 层 mismatch。
- 再对 response 内所有有效 token 取平均，得到 `seq_mismatch`。
- 用 `seq_mismatch` 判断是否过滤整条 response。

### 3. 关闭 TIS

原因：

当前实验重点是验证 router mismatch sequence-RS 对 FP8 rollout / BF16 train 不一致的缓解作用。为避免 TIS 和 router RS 同时作用导致归因困难，当前脚本关闭 TIS。

当前配置：

```bash
algorithm.rollout_correction.rollout_is=null
algorithm.rollout_correction.rollout_rs=null
```

### 4. Alignment warmup 从 3 步改为 1 步

原因：

实际观察中 `selected_alignment` 基本一直为 `1`。重复计算 `0/-1/1` 三种候选会增加不必要开销。

当前做法：

- 第 1 步计算三个 alignment。
- 第 1 步后冻结最优 alignment。
- 后续 step 只使用冻结的 alignment。

### 5. 阈值从 0.8 改为 0.3

原因：

观察到 `seq_mismatch_max` 大约只有 `0.35`，阈值 `0.8` 基本不会触发过滤。

当前阈值：

```bash
router.mismatch_rs_threshold=0.3
```

后续需要重点观察：

- `rs_rejected_fraction`
- `rs_rejected_count`
- `seq_mismatch_mean`
- `seq_mismatch_max`
- 训练 reward / validation accuracy 是否受影响

### 6. 拆分 old log prob timing

原因：

原始 `timing_s/old_log_prob` 同时包含 actor old log prob forward、router mismatch metrics、router RS mask 等多个过程，无法判断瓶颈来源。

当前拆分：

- `timing_s/old_log_prob_forward`
- `timing_s/router_mismatch_metrics`
- `timing_s/router_mismatch_rs`

当前观察：

- `old_log_prob_forward` 大约 210s。
- `router_mismatch_metrics` 在第 3 步后仍大约 700s。

结论：

当前主要瓶颈不是 actor forward，而是 router mismatch metrics 的全量大张量比较。

### 7. 预排序 top8 并复用

原因：

unordered top8 set match 需要忽略 expert 顺序。直接每次比较都排序会重复开销。

当前做法：

- 对 rollout top8 和 train top8 先 canonicalize 一次。
- 后续 alignment、token match、sequence mismatch 复用排序结果。

## 当前主要瓶颈

`timing_s/router_mismatch_metrics` 约 700s，说明当前全量 mismatch 计算过重。

主要原因：

1. routed experts 张量维度非常大：

   ```text
   batch_response x response_len x num_moe_layers x topk
   ```

   当前大致为：

   ```text
   2048 responses x 最长 16k tokens x 48 layers x top8
   ```

2. 当前对所有有效 response token、所有 MoE 层、top8 expert id 做全量比较。
3. unordered top8 set match 需要排序或等价的 set 比较。
4. 如果 routed experts 已经在 CPU 上，CPU 上做全量 sort/compare 会非常慢。
5. 每层指标记录和 `.item()` 同步也会贡献额外开销，但不是最主要瓶颈。

## 后续优化计划

### Step 1: 只对 response token 做采样估计 seq_mismatch

目标：

降低 `router_mismatch_metrics` 的主计算开销，同时保留 sequence-level RS 逻辑。

计划：

```text
每条 response 最多采样 K 个有效 response token
只在这些 token 上比较 rollout/train router mismatch
用采样 token 的平均 mismatch 估计 seq_mismatch
若 seq_mismatch > threshold，则过滤整条 response
```

建议默认：

```bash
router.mismatch_sample_tokens=1024
```

备选：

- `512`：更快，但估计噪声更大。
- `1024`：当前推荐，速度和稳定性折中。
- `2048`：更稳，但开销更高。
- `0` 或 `null`：全量计算，和当前一致。

采样方式建议：

- 使用 deterministic evenly-spaced sampling。
- 不使用随机采样，避免每步结果不可复现。
- 对有效 response token 均匀覆盖，避免只采样开头或结尾。

风险：

- `seq_mismatch` 从全量精确值变成采样估计值。
- 阈值附近的 response 可能因采样误差出现过滤抖动。
- 需要观察 `rs_rejected_fraction` 是否稳定。

### Step 2: 保持 routed experts 为 compact dtype

目标：

减少内存和数据搬运。

当前问题：

expert id 本身通常不需要 int64。若 expert id 小于 256，`uint8` 足够表达。但当前 mismatch 计算中会转为 int64。

计划：

- 避免在 mismatch 计算中强制转 int64。
- 保持原始 dtype，例如 `uint8`。
- 只在必要时转换。

当前状态：

- 已实现。
- mismatch 计算不再强制把 routed experts 转为 int64。
- 若 routed experts 已经是 `uint8/int16/int32/int64` 等整数 dtype，会保留该 dtype。
- 只有异常的非整数 dtype 才会兜底转为 int64。

风险：

- 需要确认排序和比较逻辑对 `uint8` 的支持。
- 需要确认所有 expert id 都在 dtype 可表示范围内。

### Step 3: 关闭非必要 layer-level 指标

目标：

减少每层统计和 GPU/CPU 同步开销。

计划：

增加开关：

```bash
router.mismatch_log_layer_metrics=False
```

默认只保留：

- global response token match rate
- seq mismatch mean/max
- RS rejected/kept 指标
- alignment 指标

风险：

- 无法在 W&B 中逐层观察 routing match rate。
- 对过滤逻辑无影响。

### Step 4: 删除 prompt-level rejected 统计

目标：

减少 `_apply_router_mismatch_rs` 中的 CPU `tolist()` 聚合。

计划：

只保留 response-level 过滤统计：

- `rs_rejected_fraction`
- `rs_rejected_count`
- `rs_kept_count`

风险：

- 无法直接看到一个 prompt 下 n 条 response 是否全部被过滤。
- 对训练逻辑无影响。

### Step 5: 视情况关闭 rollout log probs 诊断

当前脚本：

```bash
actor_rollout_ref.rollout.calculate_log_probs=True
```

作用：

- 产生 rollout log probs。
- 用于 `rollout_corr/*` 和 `training/rollout_probs_diff_*` 诊断。

当前 TIS/原始 rollout RS 已关闭，因此 rollout log probs 不参与权重修正。

可选优化：

```bash
actor_rollout_ref.rollout.calculate_log_probs=False
```

风险：

- 少掉 rollout/train probability 差异相关诊断。
- 对 router mismatch sequence-RS 本身不应有直接影响。

### Step 6: 如果仍然过慢，再考虑 layer sampling

目标：

减少 `num_moe_layers` 维度的计算量。

可选方案：

- 每隔 2 层取 1 层。
- 只取中后层。
- 取固定若干代表层。

风险：

- 不同层 router mismatch 分布不同，layer sampling 改变指标语义更明显。
- 建议在 token sampling 不足以降开销时再考虑。

## 当前不建议优先改的项

### 不建议先改 batch size 或 n

原因：

这些会直接改变 GRPO 的训练 batch 和组内 response 数，影响实验语义。

### 不建议先改 topk

原因：

当前模型是 top8 routing。只比较 top1/top2 会把指标语义从“完整 dispatch 一致性”改成“主专家一致性”，不适合当前目标。

### 不建议强制全量 GPU 计算作为第一选择

原因：

如果 routed experts 已经在 CPU，强制搬回 GPU 可能导致：

- 大量 CPU/GPU 拷贝
- 训练显存峰值上升
- OOM 风险
- 与 actor forward 竞争 GPU 资源

更稳妥的是先减少张量规模，再考虑 GPU 计算。

## 已完成改动记录

### 2026-06-20: 新增 top 10% router mismatch 过滤脚本

新增脚本：

```bash
/inspire/hdd/project/qianghuaxuexi/hujiarui-25046/verl-sequence/low_precision/run_grpo_qwen3_moe_30b_megatron_fp8_rollout_top10rs_local.sh
```

该脚本从当前 Megatron FP8 rollout baseline 脚本复制而来，并改为使用 top-fraction router mismatch RS：

```bash
router.mismatch_rs_mode=top_fraction
router.mismatch_rs_fraction=0.1
```

含义：

- 每个 step 在有效 response 中按 `seq_mismatch` 从高到低排序。
- 过滤 mismatch 最高的约 10% response。
- 过滤单位仍然是整条 response，即把该 response 的 `response_mask` 置 0。

代码侧新增了两种 router mismatch RS 模式：

```bash
router.mismatch_rs_mode=threshold
router.mismatch_rs_mode=top_fraction
```

默认模式仍为 `threshold`，因此原 baseline 脚本行为不变。新脚本显式使用 `top_fraction`。

新增/调整的记录指标：

- `router/rollout_vs_train/rs_threshold`：实际过滤边界。top-fraction 模式下是本 batch 被过滤 response 中的最低 `seq_mismatch`。
- `router/rollout_vs_train/rs_config_threshold`：固定阈值配置值，便于和 threshold 模式对齐。
- `router/rollout_vs_train/rs_top_fraction`：top-fraction 模式下的目标过滤比例，当前为 `0.1`。

### 2026-06-20: 新增 overlap-based sequence mismatch + top 10% 过滤脚本

新增脚本：

```bash
/inspire/hdd/project/qianghuaxuexi/hujiarui-25046/verl-sequence/low_precision/run_grpo_qwen3_moe_30b_megatron_fp8_rollout_overlap_top10rs_local.sh
```

该脚本使用 overlap-based router mismatch 作为 sequence-level RS 的过滤分数，并继续采用 top 10% response 过滤：

```bash
router.mismatch_metric_mode=overlap_fraction
router.mismatch_rs_mode=top_fraction
router.mismatch_rs_fraction=0.1
```

overlap-based mismatch 定义：

```text
layer_mismatch = 1 - top8_overlap / 8
token_mismatch = mean_over_layers(layer_mismatch)
seq_mismatch = mean_over_response_tokens(token_mismatch)
```

相比原来的 `exact_set` 模式，新模式不会把“top8 只差 1 个 expert”和“top8 完全不重合”都视为同样严重的 layer mismatch，而是按 top8 overlap 程度给出连续分数。

代码侧新增了两种 sequence mismatch 计算模式：

```bash
router.mismatch_metric_mode=exact_set
router.mismatch_metric_mode=overlap_fraction
```

默认模式仍为 `exact_set`，因此原 baseline 和 top10rs 脚本行为不变。新 overlap 脚本显式使用 `overlap_fraction`。

### 2026-06-21: 固定阈值 sequence-RS 脚本阈值从 0.3 调整到 0.5

修改脚本：

```bash
/inspire/hdd/project/qianghuaxuexi/hujiarui-25046/verl-sequence/low_precision/run_grpo_qwen3_moe_30b_megatron_fp8_rollout_seqRS_threshold0.3.sh
```

默认 router mismatch RS 阈值由：

```bash
router_mismatch_rs_threshold=${ROUTER_MISMATCH_RS_THRESHOLD:-0.3}
```

改为：

```bash
router_mismatch_rs_threshold=${ROUTER_MISMATCH_RS_THRESHOLD:-0.5}
```

该脚本仍然使用 threshold 模式和 exact-set sequence mismatch。过滤条件变为：

```text
seq_mismatch > 0.5
```

### 2026-06-22: 训练 reward debug 落盘

新增 DAPO reward debug JSONL 输出，用于定位前几十步 reward 接近 -1 的原因。

改动文件：

```bash
verl/workers/reward_manager/dapo.py
low_precision/run_grpo_qwen3_moe_30b_megatron_fp8_rollout_overlap_top10rs_local.sh
```

新增环境变量：

```bash
VERL_REWARD_DEBUG_DIR
VERL_REWARD_DEBUG_STEPS
VERL_REWARD_DEBUG_SAMPLES
```

overlap top10rs 脚本默认写入：

```bash
${CKPTS_DIR}/reward_debug
```

默认记录前 40 个 debug step、每步最多 16 条训练 reward 样本。每条 JSONL 包含 prompt、response、response_tail_300、ground_truth、score、pred、response_length、是否包含 Answer:、是否包含 \boxed 等字段。

### 2026-06-22: threshold sequence-RS 脚本开启训练 reward debug

修改脚本：

```bash
low_precision/run_grpo_qwen3_moe_30b_megatron_fp8_rollout_seqRS_threshold.sh
```

该脚本默认导出：

```bash
VERL_REWARD_DEBUG_DIR=${CKPTS_DIR}/reward_debug
VERL_REWARD_DEBUG_STEPS=40
VERL_REWARD_DEBUG_SAMPLES=16
```

新启动任务后会在对应 checkpoint 目录下保存前 40 个 debug step 的训练 reward 样本 JSONL，用于检查低 reward 是否来自答案格式、parser 提取失败或模型真实答错。

### 2026-06-22: 本地 perf debug 诊断 validation/router/old-log-prob 慢点

新增本地 JSONL perf debug，用于定位 `timing_s/testing`、`timing_s/router_mismatch_metrics` 和 `timing_s/old_log_prob_forward` 在某个 step 突然增加的原因。

改动文件：

```bash
verl/trainer/ppo/ray_trainer.py
dapo/dapo_ray_trainer.py
low_precision/run_grpo_qwen3_moe_30b_megatron_fp8_rollout_overlap_top10rs_local.sh
low_precision/run_grpo_qwen3_moe_30b_megatron_fp8_rollout_seqRS_threshold.sh
```

两个脚本默认导出：

```bash
VERL_PERF_DEBUG_DIR=${CKPTS_DIR}/perf_debug
```

新增本地文件：

```text
perf_debug/router_mismatch_stepXXXX.jsonl
perf_debug/old_log_prob_forward_stepXXXX.jsonl
perf_debug/validation_stepXXXX.jsonl
```

记录内容包括 routed_experts shape/dtype/device、response_mask token 数、router metric mode、alignment、old_log_prob 返回 key、validation generate/decode/reward/log_generations 子阶段耗时、validation response length 统计以及 CUDA allocated/reserved memory 快照。

### 2026-06-22: reward debug 覆盖 agent loop rm_scores 快路径

修正 DAPO reward debug：训练任务中 agent loop 可能已经提前计算 reward，并通过 `rm_scores` 返回。此前 debug 只覆盖 fallback `compute_score` 分支，遇到 `rm_scores` 会直接返回，导致没有生成 `reward_debug`。

改动文件：

```bash
verl/workers/reward_manager/dapo.py
```

现在当 `rm_scores` 存在时也会记录本地 JSONL。每条样本包含 agent loop 给出的 `score`，并额外用当前 reward parser 对同一条 response 重新计算 `parser_score_debug`、`parser_acc_debug` 和 `pred`，用于判断低 reward 是模型真实答错、格式不匹配，还是 parser 提取失败。

### 2026-06-22: CPU router expert set compare benchmark

新增独立 microbenchmark：

```bash
low_precision/bench_router_set_compare_cpu.py
```

用于在 CPU 上模拟 `2048 responses × 16384 tokens × 48 MoE layers × top8` 的无序 top-k expert set 比较耗时。默认 dtype 对齐当前 debug 观测：

```text
rollout routed_experts: torch.int32
train routed_experts: torch.uint8
```

默认使用 token chunk 复用方式模拟完整计算量，避免直接申请两份完整 routed experts 大张量导致内存压力过大。

### 2026-06-22: benchmark 对齐实际 router mismatch metric 路径

扩展 `low_precision/bench_router_set_compare_cpu.py`，新增 `--mode exact_metrics`。

该模式直接调用训练代码中的 `compute_router_mismatch_metrics`，对齐真实脚本里的主要开销来源：

```text
rollout shape: [batch, response_len + rollout_extra_tokens, layers, topk]
train shape: [batch, response_len, layers, topk]
response_mask shape: [batch, response_len]
alignment candidates: configurable, default 1
metric_mode: exact_set
```

相比原来的 `core` 模式，新模式会包含 top-k 排序、alignment train 副本构造、全局 token match rate、逐层 match rate、sequence mismatch 统计等步骤，更接近训练时 `timing_s/router_mismatch_metrics` 的实际执行路径。新增 `--output-json` 用于把结果保存为结构化文件。

### 2026-06-23: DAPOTaskRunner CPU 数可配置

将 DAPO trainer 所在的 Ray actor 从固定 `num_cpus=1` 改为环境变量控制：

```bash
DAPO_TASK_RUNNER_NUM_CPUS
```

改动文件：

```bash
dapo/main_dapo.py
low_precision/run_grpo_qwen3_moe_30b_megatron_fp8_rollout_seqRS_threshold.sh
```

Python 入口兜底默认值为 `32`；`run_grpo_qwen3_moe_30b_megatron_fp8_rollout_seqRS_threshold.sh` 脚本默认导出 `64`。脚本同步设置：

```bash
OMP_NUM_THREADS=${DAPO_TASK_RUNNER_NUM_CPUS}
MKL_NUM_THREADS=${DAPO_TASK_RUNNER_NUM_CPUS}
OPENBLAS_NUM_THREADS=${DAPO_TASK_RUNNER_NUM_CPUS}
NUMEXPR_NUM_THREADS=${DAPO_TASK_RUNNER_NUM_CPUS}
```

目的：让 `router_mismatch_metrics` 这类在 DAPOTaskRunner/trainer 进程内执行的 CPU 大张量计算不再被 Ray actor 的 `num_cpus=1` 和单线程 CPU kernel 限制。

### 2026-06-23: 在 DAPO trainer 的 extract_reward 路径增加 reward debug

在 `dapo/dapo_ray_trainer.py` 中新增 `extract_reward` 后的本地 debug hook。

原因：当前 DAPO 训练循环使用 `extract_reward(new_batch)` 直接从 `rm_scores` 取 reward，之前加在 `verl/workers/reward_manager/dapo.py` 的 debug 在这条快路径上不会触发。

启用条件：

```bash
VERL_REWARD_DEBUG_DIR
VERL_REWARD_DEBUG_STEPS
VERL_REWARD_DEBUG_SAMPLES
```

输出文件：

```text
${VERL_REWARD_DEBUG_DIR}/extract_reward_debug_stepXXXX_pidYYYY.jsonl
```

每条记录包含 prompt、完整 response、response tail、ground truth、`rm_score`，以及使用当前 `default_compute_score` 对同一 response 重新解析得到的 `parser_score_debug`、`parser_acc_debug`、`parser_pred_debug` 和 `parser_error`。用途是定位训练前几步 reward 接近 `-1` 时，到底是答案格式不被 parser 接受、答案本身错误，还是数据源/ground truth 对不上。

### 2026-06-24: math_dapo reward parser 兼容 boxed 与 Answer 格式

改动文件：

```bash
verl/utils/reward_score/math_dapo.py
```

`compute_score` 的答案提取逻辑从只依赖 legacy `Answer:` 正则，改为：

```text
尾部 1000 字符内最后一个 \boxed{...} 优先
若没有 boxed，再 fallback 到 Answer: xxx
```

返回结构保持不变：

```python
{"score": reward, "acc": acc, "pred": pred}
```

目的：对齐 DeepScaleR 训练 prompt 中的 `output the final within \boxed{}` 要求，同时保留对 `Answer:` 输出的兼容，避免 boxed 正确答案被解析成 `[INVALID]` 后打成 `-1`。

### 2026-06-26: overlap mismatch RS 默认改为固定阈值 0.25

改动文件：

```bash
low_precision/run_grpo_qwen3_moe_30b_megatron_fp8_rollout_overlap_top10rs_local.sh
```

默认 router mismatch RS 从 top 10% 过滤改为固定阈值过滤：

```bash
router_mismatch_rs_threshold=${ROUTER_MISMATCH_RS_THRESHOLD:-0.25}
router_mismatch_rs_mode=${ROUTER_MISMATCH_RS_MODE:-threshold}
router_mismatch_rs_fraction=${ROUTER_MISMATCH_RS_FRACTION:-0.0}
```

实验名默认后缀同步改为 `overlap-mismatch-threshold0.25RS-16K`，避免继续显示 `TOP10RS` 造成混淆。仍可通过环境变量覆盖阈值、模式和过滤比例。

### 2026-06-26: router mismatch 训练过程分析数据 dump

改动文件：

```bash
dapo/dapo_ray_trainer.py
verl/utils/router_mismatch_metrics.py
low_precision/run_grpo_qwen3_moe_30b_megatron_fp8_rollout_overlap_top10rs_local.sh
```

`RouterMismatchResult` 新增 `token_mismatch`，复用已计算的 token 级 mismatch，避免 dump 时重新比较专家集合。

新增可控 dump 开关：

```bash
VERL_ROUTER_ANALYSIS_DUMP_DIR
VERL_ROUTER_ANALYSIS_DUMP_MODE=summary|tokens|sample|full|off
VERL_ROUTER_ANALYSIS_DUMP_EVERY_N
VERL_ROUTER_ANALYSIS_DUMP_STEPS
VERL_ROUTER_ANALYSIS_DUMP_SAMPLES
VERL_ROUTER_ANALYSIS_DUMP_FLOAT_DTYPE=float16|bfloat16|float32
VERL_ROUTER_ANALYSIS_DUMP_TOPK_TOKENS
```

默认脚本配置为 `tokens` 模式，输出到：

```text
${CKPTS_DIR}/router_analysis_dump/router_analysis_stepXXXX_pidYYYY.pt
```

保存内容包括：

- metadata：step、alignment、route shape/dtype、response token 数、metric 名称等；
- response_summary：`seq_mismatch`、`seq_valid_token_count`、`seq_prob_diff`、`seq_logprob_diff`、`seq_reward`；
- token_data：`responses`、`response_mask`、`attention_mask`、`token_mismatch`、`old_log_probs`、`rollout_log_probs`、`logprob_delta`、`logprob_diff`、`prob_diff`。
- response_summary.extreme_tokens：每条 response 内 `token_mismatch` 最大的 topK token 位置、token id、`token_mismatch`、`old_log_probs`、`rollout_log_probs`、`prob_diff`、`logprob_diff`，用于定位序列内极端 mismatch token。
- response_summary.extreme_prob_diff_tokens：每条 response 内 `prob_diff` 最大的 topK token 位置、token id、`prob_diff`、`logprob_diff`、`token_mismatch`、两侧 logprob，用于定位训推 sampled-token 概率差异最大的 token。

`sample/full` 模式额外保存两侧 routed experts 原始张量；`sample` 仅保存前 `VERL_ROUTER_ANALYSIS_DUMP_SAMPLES` 条 response，`full` 保存全 batch，磁盘和耗时开销很大，默认不开。

### 2026-07-02: Megatron torch_dist checkpoint 严格 resume 兼容补丁

改动文件：

```bash
verl/utils/megatron/dist_checkpointing.py
```

问题：

`global_step_200/actor/dist_ckpt/.metadata` 是 PyTorch DCP 原生 `Metadata`，包含 `storage_data/state_dict_metadata`，但没有 Megatron-Core 0.14 load 逻辑期望的 `mcore_data` 字段。严格 resume 时会在 `get_reformulation_metadata()` 中报：

```text
AttributeError: 'Metadata' object has no attribute 'mcore_data'
```

处理：

在本地 verl 的 Megatron checkpoint load wrapper 里 monkey patch `megatron.core.dist_checkpointing.strategies.torch.get_reformulation_metadata`。当 checkpoint metadata 缺少 `mcore_data` 时，为 N-D flattened tensors 合成 identity reformulation metadata，让加载路径按当前 tensor formulation 读取。

适用前提：

- resume 时 Megatron 并行配置必须和保存 checkpoint 时一致，例如当前实验为 `TP=4, PP=1, EP=4, ETP=2`；
- 不用于改变 TP/PP/EP/ETP 后的 resharding 加载。

### 2026-07-07: expert usage distribution drift 指标与 top8 RS 脚本

改动文件：

```bash
verl/utils/router_mismatch_metrics.py
verl/trainer/ppo/ray_trainer.py
verl/trainer/config/ppo_trainer.yaml
low_precision/run_grpo_qwen3_moe_30b_megatron_fp8_rollout_usage_l1_top8rs_local.sh
```

新增 `router.mismatch_metric_mode=expert_usage_l1`：

- 对每条 response、每个 MoE 层分别统计 rollout/train 两侧 top-k expert usage 分布；
- 使用 Total Variation / L1 距离：

```text
D_{i,l} = 0.5 * sum_e |p_{i,l}(e) - q_{i,l}(e)|
```

- 序列级分数为所有 MoE 层平均；
- `seq_mismatch` 使用短序列 shrinkage 后的 expert usage L1 分数，现有 `router.enable_mismatch_rs` 过滤逻辑可直接复用；
- 脚本默认 `router.expert_usage_smoothing_tau=4096.0`，相当于约 512 个 token 的 top8 expert slots prior 强度；
- 额外记录 raw/smoothed 全局指标、shrink factor、每层 smoothed/raw expert usage L1，方便后续比较不同距离和聚合方式。

新增脚本：

```bash
low_precision/run_grpo_qwen3_30b_a3b_expert_distribution_l1_top8RS.sh
```

默认设置：

```bash
router.mismatch_metric_mode=expert_usage_l1
router.mismatch_rs_mode=top_fraction
router.mismatch_rs_fraction=0.08
router.expert_usage_smoothing_tau=4096.0
```

`tau=4096` 是第一版保守默认值，依据：

- 既有 `threshold0.25 + TIS-C2 resume200` dump 显示短回答存在明显 bias：`0-2048` 桶过滤率约 22.31%，`2048-4096` 桶约 18.00%，而 `8192-12288` 桶仅约 2.01%；
- 同一分析中 `seq_mismatch vs length` 相关系数约 `-0.4466`，短序列更容易得到高 mismatch；
- `TIS-C2 noRS` dump 也显示长度桶越短，`seq_mismatch_mean` 越高：`0-2048` 约 0.2408，`12288-16384` 约 0.2007；
- 但 overlap+TIS dump 中 fixed threshold 过滤比例偏低：`0.045RS+TIS` 约 0.61%，`0.04RS+TIS` 约 3.25%。因此第一版不宜把 shrinkage 设置得太强，避免过度改变排序；
- `tau=4096` 对应 top8 下约 512 token prior 强度：长度 512/1024/2048 token 的 shrink factor 分别约 0.500/0.667/0.800，长度 4096/8192 token 分别约 0.889/0.941。

额外记录：

```text
expert_usage_l1_raw_p90
expert_usage_l1_smoothing_delta_mean
expert_usage_l1_shrink_factor_p10/p50/p90
```

用于后续消融 `tau=2048/4096/8192`、换距离函数、换聚合方式时判断 smoothing 对分数分布和排序的影响。

### 2026-07-07: expert distribution L1 脚本 checkpoint 保留策略

改动文件：

```bash
low_precision/run_grpo_qwen3_30b_a3b_expert_distribution_l1_top8RS.sh
```

默认 checkpoint 策略改为：

```bash
trainer.save_freq=30
trainer.max_actor_ckpt_to_keep=1
trainer.max_critic_ckpt_to_keep=1
```

含义：每 30 step 保存一次 checkpoint；保存新 checkpoint 后最多只保留最近 1 个 checkpoint，旧 checkpoint 会被 checkpoint manager 清理。当前 GRPO 无 critic，但同步设置 `max_critic_ckpt_to_keep=1`，避免未来脚本配置变化时保留策略不一致。

### 2026-07-08: expert usage L1 长度分桶过滤脚本

改动文件：

```bash
verl/trainer/ppo/ray_trainer.py
verl/trainer/config/ppo_trainer.yaml
low_precision/run_grpo_qwen3_30b_a3b_expert_distribution_l1_lengthbucket_top8RS_TIS.sh
```

新增 `router.mismatch_rs_mode=length_bucket_top_fraction`。过滤逻辑从全 batch 直接 top-k 改为按 response 有效长度分桶后，在每个桶内分别过滤最高 `router.mismatch_rs_fraction` 的 response。默认桶边界：

```bash
router.mismatch_rs_length_bucket_edges=[2048,4096,8192,12288]
```

新脚本默认配置：

```bash
EXPERIMENT_NAME_BASE=GRPO-DEEPSCALER-Qwen3-30B-A3B-base-MEGATRON-VLLM-FP8-usage-l1-lengthbucket-top8RS-TIS-C2-16K
algorithm.rollout_correction.rollout_is=token
algorithm.rollout_correction.rollout_is_threshold=2.0
router.enable_mismatch_rs=True
router.mismatch_rs_mode=length_bucket_top_fraction
router.mismatch_rs_fraction=0.08
router.mismatch_metric_mode=expert_usage_l1
router.expert_usage_smoothing_tau=4096.0
trainer.val_before_train=True
trainer.save_freq=30
trainer.max_actor_ckpt_to_keep=1
trainer.max_critic_ckpt_to_keep=1
```

目的：修复全局 top8% 过滤中观察到的长度 bias。当前 dump 显示全局 top8% 主要过滤短且正确 response；长度分桶使短 response 只和短 response 比、长 response 只和长 response 比，避免分数与长度的系统性负相关直接转化为过滤偏置。

额外记录的分桶指标：

```text
router/rollout_vs_train/rs_bucket_{i}_count
router/rollout_vs_train/rs_bucket_{i}_rejected_count
router/rollout_vs_train/rs_bucket_{i}_rejected_fraction
router/rollout_vs_train/rs_bucket_{i}_threshold
router/rollout_vs_train/rs_bucket_{i}_score_mean
router/rollout_vs_train/rs_bucket_{i}_length_mean
```

### 2026-07-11: 长度桶 MAD z-score 全局 Top8% 与 prompt 保护

改动文件：

```bash
verl/trainer/ppo/ray_trainer.py
verl/trainer/config/ppo_trainer.yaml
low_precision/run_grpo_qwen3_30b_a3b_expert_distribution_l1_madz_top8RS_TIS.sh
```

新增 `router.mismatch_rs_mode=length_bucket_mad_zscore_top_fraction`：先在每个 response
长度桶内用 `median/MAD` 标准化 Expert Usage L1 分数，再对整个 batch 的 z-score 全局排序，
默认过滤最高 8%。默认每个 prompt 最多过滤 2 条 response；配置了该限制但 batch 缺少 `uid`
时会直接报错，避免静默退化为无 prompt 保护的过滤。

默认配置：

```bash
algorithm.rollout_correction.rollout_is=token
algorithm.rollout_correction.rollout_is_threshold=2.0
router.mismatch_rs_mode=length_bucket_mad_zscore_top_fraction
router.mismatch_rs_fraction=0.08
router.mismatch_rs_max_reject_per_prompt=2
router.mismatch_rs_mad_epsilon=1e-6
router.mismatch_metric_mode=expert_usage_l1
router.expert_usage_smoothing_tau=4096.0
```

新增记录包括各桶原始分数的 mean/median/MAD、z-score mean/max、实际过滤率、目标过滤数、
因 prompt 上限跳过的候选数，以及过滤/保留 response 的平均长度、reward 均值和正 reward 比例。

使用 `256 prompts x 8 responses` 的 CPU 合成 batch 验证：2048 条 response 过滤 164 条
（`ceil(2048 * 0.08)`），每个 prompt 最多过滤 2 条。

### 2026-07-14: 连续长度条件分位数 Top8% 过滤

改动文件：

```bash
verl/utils/router_mismatch_metrics.py
verl/trainer/ppo/ray_trainer.py
verl/trainer/config/ppo_trainer.yaml
tests/utils/test_router_mismatch_metrics_on_cpu.py
low_precision/run_grpo_qwen3_30b_a3b_expert_distribution_l1_conditional_percentile_top8RS_TIS.sh
```

新增 `router.mismatch_rs_mode=length_conditional_percentile_top_fraction`。先按 response
有效长度排序，每条 response 在长度相邻的滑动窗口中计算 Expert Usage L1 的经验百分位；
同分采用 midpoint rank。达到最大生成长度的右截断 response 在数量足够时单独成组计算百分位。
随后按条件百分位全局排序过滤，并复用已有的每个 prompt 最大过滤数量限制。

默认配置：

```bash
algorithm.rollout_correction.rollout_is=token
algorithm.rollout_correction.rollout_is_threshold=2.0
router.mismatch_rs_mode=length_conditional_percentile_top_fraction
router.mismatch_rs_fraction=0.08
router.mismatch_rs_max_reject_per_prompt=2
router.mismatch_rs_local_window=256
router.mismatch_rs_censored_length=16384
router.mismatch_rs_min_censored_count=32
router.mismatch_metric_mode=expert_usage_l1
router.expert_usage_smoothing_tau=4096.0
```

计算在 router score 所在设备上使用张量化的滑动窗口比较；默认 batch 2048、窗口 256，
约 52 万次标量比较。新增记录包括条件百分位均值、过滤/保留条件百分位均值、右截断
response 数、目标过滤数和因 prompt cap 跳过的候选数。原始 dump 已包含重算条件百分位所需的
`seq_mismatch` 和 `seq_valid_token_count`，因此未重复保存每条 response 的条件百分位。

CPU 测试覆盖 midpoint tie、右截断单独校准、无效 response，以及 trainer 端全局过滤和
prompt cap。合成 batch 中 8 条 response 按 25% 过滤 2 条，且同一 prompt 最多过滤 1 条。
## 2026-07-15 Expert distribution distance probe

新增低开销专家分布距离观测实验：

- 保持 `expert_usage_l1`（实际为平滑 TV）作为 Top8% RS 的真实过滤分数。
- 在现有逐层专家计数循环中复用同一份直方图，同时记录 raw TV、L2、L-infinity、Hellinger squared、normalized JS 和 drift effective support。
- 新增 `expert_counts` dump 模式，只在配置的 dump step 保存 rollout/train 两侧 `[response, layer, expert]` 紧凑计数和逐层距离，不保存完整 token/logprob 张量。
- dump 同时保存实际 `rs_reject_mask`，并用过滤前的有效 response mask 计算长度和概率差，确保可以精确审计实际过滤集合。
- 新测试脚本默认关闭训练前验证，运行 50 step，每 step dump 一次，每 5 step 保存 checkpoint。
- 继承基础脚本的 `max_actor_ckpt_to_keep=1` 和 `max_critic_ckpt_to_keep=1`，成功保存新 checkpoint 后删除旧 checkpoint。
- 默认开启 token-level TIS，截断阈值 `C=2`，不启用 verl 原生 rollout RS；Expert Usage TV Top8% RS 保持开启。

脚本：

```text
low_precision/run_grpo_qwen3_30b_a3b_expert_distribution_distance_probe_top8RS.sh
```

该实验不更改训练使用的 RS 分数，目的是从相同 response 和相同专家计数离线比较不同距离的排序、长度偏差和训练风险相关性。
