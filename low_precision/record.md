# Router Mismatch Sequence-RS Experiment Plan

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
