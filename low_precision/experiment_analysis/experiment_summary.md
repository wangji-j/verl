# FP8 Rollout / Router Mismatch 实验记录

更新时间：2026-07-02

本文档汇总当前 `verl-sequence` 和相关历史目录下的 Qwen3-30B-A3B MoE 强化学习实验。信息来源包括：

- 本地脚本：`/inspire/hdd/project/qianghuaxuexi/hujiarui-25046/verl-sequence/low_precision/*.sh`
- checkpoint 和 dump：`/inspire/hdd/project/qianghuaxuexi/hujiarui-25046/ckpts/jly-DAPO-DEEPSCALER-FP8-ROLLOUT`
- 本地 W&B metadata / output logs / run logs
- 已生成的 dump 分析文件：
  - `threshold0.5RS_dump_analysis.json`
  - `threshold0.03RS_dump_sample_analysis.json`
  - `threshold0.045RS_group_extreme_analysis.json`
- 之前手工检查和调试结论。

W&B 本地 metadata 没有记录完整 Docker image tag，因此“镜像版本”部分只写可验证的软件版本和你之前明确描述过的镜像口径。

## 1. 总目标

实验目标是在 **Qwen3-30B-A3B MoE + Megatron 训练 + vLLM FP8 rollout** 场景下，研究 rollout 端和训练端的不一致是否会导致 GRPO 训练不稳定，并尝试用 router mismatch 相关指标做诊断和过滤。

核心问题：

1. FP8 rollout / BF16 train 下，MoE router 选择是否明显不一致。
2. 这种不一致是否和 reward 崩、response length 塌缩/爆长、clip ratio 升高相关。
3. 是否能用 sequence-level router mismatch 做 rejection sampling，过滤高风险 response。
4. 固定阈值、top-k 比例、overlap_fraction 等方案哪个更稳定。
5. 额外 mismatch 指标和 dump 对训练耗时的影响有多大。

## 2. 环境和镜像信息

当前 `verl-sequence` 环境可观测版本：

| 组件 | 版本 |
| --- | --- |
| Python | 3.12.11 |
| vLLM | 0.17.0 |
| Megatron-Core | 0.14.0rc7 |
| PyTorch | 2.10.0 |
| Ray | 2.49.2 |
| W&B | 0.22.3 |
| Transformers | 4.57.1 |

当前主实验口径：

- 镜像：用户侧描述为 `vLLM 0.17.0 + Megatron` 环境。
- 训练后端：Megatron。
- Rollout 后端：vLLM async。
- Rollout 量化：`actor_rollout_ref.rollout.quantization=fp8`。
- vLLM V1：脚本中 `VLLM_USE_V1=1`。
- 硬件：多次实验使用 2 节点 16 卡，单节点配置为 8 * H200 141GB，CPU 约 180 核，内存约 1800GB。

早期对比环境：

- 旧脚本/旧镜像口径包含 `vLLM 0.11.0 + FSDP`。
- 当前 `verl-sequence` 主要实验不再使用该 FSDP 旧环境。

## 3. 公共训练设置

当前主线 Megatron GRPO 脚本共享设置大致如下：

| 项 | 设置 |
| --- | --- |
| 模型 | `/inspire/hdd/project/qianghuaxuexi/public/models/Qwen3-30B-A3B` |
| 训练集 | `/inspire/hdd/project/qianghuaxuexi/public/datasets/deepscaler/train.parquet` |
| 验证集 | `/inspire/hdd/project/qianghuaxuexi/public/datasets/aime_2024/aime24_aime25_x32.parquet` |
| 算法 | `algorithm.adv_estimator=grpo` |
| Recipe | `python3 -m dapo.main_dapo --config-name=dapo_megatron_trainer` |
| Reward manager | `reward.reward_manager.name=dapo` |
| prompt batch | 256 |
| 每 prompt response 数 | `n=8` |
| 最大 prompt 长度 | 2048 |
| 最大 response 长度 | 16384 |
| 训练节点 | 2 节点 * 8 GPU |
| Megatron TP/PP/EP/ETP | `train_tp=4`, `train_pp=1`, `train_ep=4`, `train_etp=2` |
| Rollout TP | `gen_tp=2` |
| KL reward | `algorithm.use_kl_in_reward=False`, `kl_coef=0` |
| Actor KL loss | `use_kl_loss=True`, `kl_loss_coef=0.001`, `kl_loss_type=low_var_kl` |
| PPO clip | low=0.2, high=0.27, c=10.0 |
| 验证 | 多数主脚本 `trainer.val_before_train=True`, `test_freq=10` |
| checkpoint | `save_freq=50`, `resume_mode=auto`, `max_actor_ckpt_to_keep=5` |

注意：虽然入口叫 `dapo/main_dapo.py`，脚本中实际 `adv_estimator=grpo`。这里的 DAPO recipe 更像训练 recipe / runner / 配置组织方式，不代表算法一定是 DAPO 算法；具体算法由 `algorithm.adv_estimator` 和相关 correction 设置决定。

## 4. Router mismatch 定义

目前存在两类 sequence mismatch 方案。

### 4.1 exact_set / 普通聚合

某个 token、某一层：

```text
layer_match = 1[set(rollout_top8_experts) == set(train_top8_experts)]
layer_mismatch = 1 - layer_match
```

即 top8 expert id 不考虑顺序，只要集合一致就算 match。

某个 token：

```text
token_mismatch = mean_over_moe_layers(layer_mismatch)
```

某条 response：

```text
seq_mismatch = mean_over_valid_response_tokens(token_mismatch)
```

特点：

- 语义清晰，但比较严格。
- 一个 layer 的 8 个 expert 只要少一个不同，该 layer 就是 mismatch。
- 比 `response_token_match_rate` 更合理，因为不要求所有层同时一致。

### 4.2 overlap_fraction / 平滑聚合

某个 token、某一层：

```text
layer_mismatch = 1 - |set(rollout_top8) intersect set(train_top8)| / 8
```

然后：

```text
token_mismatch = mean_over_layers(layer_mismatch)
seq_mismatch = mean_over_valid_response_tokens(token_mismatch)
```

特点：

- 比 exact_set 更平滑。
- 如果 8 个 expert 中有 7 个相同，只给 1/8 的 mismatch，而不是整层 mismatch。
- 更适合做连续阈值过滤，但绝对阈值尺度和 exact_set 不可直接比较。

## 5. 主要实验矩阵

### 5.1 Baseline / mismatch-only

代表脚本：

- `run_grpo_qwen3_moe_30b_megatron_fp8_rollout_seqRS_threshold.sh` 通过启动参数关闭 RS。
- `GRPO-...sequence-mismatch-TIS-C2-noRS`

典型设置：

- FP8 rollout + Megatron train。
- 开启 router mismatch metrics。
- 不开启 router mismatch RS。
- 可选 TIS C=2。

观察：

- 可用于观察 `seq_mismatch_mean`、`response_token_match_rate`、`training/rollout_probs_diff_mean` 和 reward/length 的自然演化。
- 某些 noRS 记录型实验在约 50 step 附近出现 reward 下跌、entropy 更明显下降、response length 从较短转向接近 16K、clip ratio 快速升高。

解释：

- 这更像训练进入不稳定区间后的长度分布崩坏，而不是单纯 “logit diff 变大”。
- 观察到 `training/rollout_probs_diff_mean` 下降但 `seq_mismatch_mean` 上升，说明“采样 token 的概率差异”和“MoE router 路径一致性”不是同一个信号。模型可以在 sampled token probability 上更接近，但内部 routing path 更不一致。

### 5.2 exact_set 固定阈值 RS

代表目录：

- `GRPO-...sequence-mismatch-threshold0.25_20260625_221424`
- `GRPO-...sequence-mismatch-threshold0.3_20260625_201633`
- `GRPO-...sequence-mismatch-threshold0.5_*`
- `GRPO-...sequence-mismatch-threshold0.25-TIS-C2_20260626_072854`

过滤逻辑：

```text
if seq_mismatch > threshold:
    mask/drop this response
```

现象：

- 阈值过高时，前期几乎不过滤，无法阻止早期曲线跟 baseline 一样下跌。
- 阈值过低时，前期能稳住一部分，但后期过滤比例会快速升高，可能达到 40%-60%，validate 反而崩。
- 固定阈值容易出现阶段错配：前期阈值显得太松，后期又可能太严。

解释：

- `seq_mismatch` 的分布不是静态的，会随训练阶段、response length、模型 entropy、生成格式变化而变。
- 用单个绝对阈值跨 500 step 过滤，容易把“训练阶段变化”误当成“异常样本”。
- 后期高 mismatch 样本不一定都是坏样本；大量过滤会改变训练分布，可能引入长度/难度/格式 bias。

### 5.3 exact_set Top10RS

代表脚本：

- `run_grpo_qwen3_moe_30b_megatron_fp8_rollout_top10rs_local.sh`

设置：

- `router.mismatch_rs_mode=top_fraction`
- `router.mismatch_rs_fraction=0.1`
- 默认过滤当前 batch 内 seq_mismatch 最高的 10% response。

优点：

- 不依赖固定绝对阈值。
- 每个 step 控制过滤比例，避免固定阈值在不同训练阶段过松/过严。

风险：

- 如果某个 step 整体 mismatch 都很低，仍然会过滤 10%，可能误伤。
- 如果训练后期整体 mismatch 急剧升高，只过滤 10% 可能不够。

更合理的后续方向：

- 用 top fraction 作为主策略，但加 warmup / cap / floor。
- 例如前期 5%-10%，中期 10%-15%，当 `seq_mismatch_mean` 快速上升时允许提升到 20%-30%，但不建议无上限涨到 50%。

### 5.4 overlap_fraction 固定阈值 RS

代表脚本：

- `run_grpo_qwen3_moe_30b_megatron_fp8_rollout_overlap_top10rs_local.sh`

当前脚本默认：

- `router.mismatch_metric_mode=overlap_fraction`
- `router.mismatch_rs_mode=threshold`
- `router.mismatch_rs_threshold=0.045`
- 默认开启 dump：`VERL_ROUTER_ANALYSIS_DUMP_MODE=tokens`

已分析实验：

| 实验 | dump | 结论 |
| --- | --- | --- |
| threshold0.5RS | 68 steps / 56G | 阈值 0.5 基本不过滤，seq_mismatch p99 约 0.064 |
| threshold0.03RS | 采样分析到 113 step | 0.03 前期过度过滤，后期可达接近 100% |
| threshold0.045RS | 32 steps / 27G | 前期平均过滤约 1.6%，过于温和 |
| threshold0.05RS | 有 run logs / reward debug | 与 0.045 接近，需结合 W&B 曲线继续看 |

关键数值：

- threshold0.5RS：`seq_mismatch_mean=0.0355`, p50=0.0333, p90=0.0468, p99=0.0639，因此 0.5 对 overlap_fraction 完全不合适。
- threshold0.03RS：step 1 `seq_mismatch_mean=0.0321`，从一开始就会大量过滤；step 113 `seq_mismatch_mean=0.0506`，reward 已接近 -0.91。
- threshold0.045RS：前 32 step 平均 reject fraction 约 1.58%，过滤样本并不明显更差，甚至在部分统计里 rejected reward 更高、length 更短。

解释：

- overlap_fraction 的数值尺度远小于 exact_set，不能沿用 exact_set 的阈值。
- 0.03 太激进，0.045 太保守，0.5 无效。
- 仅用固定阈值仍然容易受训练阶段影响，后续应该优先比较同 reject fraction 下的 exact_set vs overlap_fraction，而不是比较绝对阈值。

### 5.5 TIS C=2

常用启动覆盖：

```bash
algorithm.rollout_correction.rollout_is=token
algorithm.rollout_correction.rollout_is_threshold=2.0
algorithm.rollout_correction.rollout_is_batch_normalize=false
algorithm.rollout_correction.rollout_rs=null
algorithm.rollout_correction.rollout_rs_threshold=null
```

作用：

- TIS 是 token-level importance sampling correction。
- 它不丢样本，而是用 train/rollout 概率比对 token loss 做权重修正，并用阈值 C=2 截断极端权重。

观察：

- verl FP8 文档中，Qwen3-30B-A3B MoE 的 FP8 rollout 实验明确使用 token-level TIS C=2，并指出 MoE 模型 rollout/train mismatch 更高，即使 BF16 也需要 correction。
- 当前实验中 TIS/noRS 可以作为重要 reference，用来判断 router RS 是否比概率比修正更有效。

解释：

- TIS 修正的是输出 token 概率分布差异。
- router mismatch RS 修正/过滤的是 MoE 内部路径差异。
- 两者不是同一个信号，可以互补，也可能互相干扰；消融必须分开做。

### 5.6 R3 / routing replay

历史问题：

- 早期尝试 `router.enable_mismatch_metrics=True` 或 routed experts 返回时，vLLM 端出现过 `--enable-return-routed-experts` 不识别问题。
- 后来确认当前 vLLM 0.17.0 环境支持本地代码路径，但不同镜像/入口下仍要看 vLLM server 是否接收相关参数。

现状：

- `run_grpo_qwen3_moe_30b_megatron_fp8_rollout_r3_local.sh` 是 R3 相关脚本。
- R3 类实验可以记录 rollout/logprob/update forward 三段路由关系，但会增加实现和性能复杂度。

建议：

- R3 更适合作为诊断实验，不建议直接和 sequence RS 同时打开做首轮主实验。

### 5.7 Reward parser debug

代表目录：

- `debug-reward-first-steps_20260623_233918`
- `debug-reward-boxed-parser_20260624_011438`

现象：

- 早期一些 GRPO base 实验 reward 接近 -1、validate 接近 0，不符合对 Qwen3-30B-A3B 的直觉。
- Debug 显示一个重要原因是答案格式 / reward parser 不匹配。
- 后续将 parser 改为优先读取最后 1000 字符内的 final `\boxed{...}`，fallback 到 `Answer:`。

解释：

- 初始 reward 低不一定说明模型完全不会做题，可能是输出格式和打分器抽取格式不一致。
- 后续 reward 能涨，可能是模型逐渐学会更符合 reward parser 的输出格式。
- 这解释了为什么同模型同数据，在不同 recipe 或不同框架中初始 reward 会明显不同。

### 5.8 Logprob / performance debug

代表目录：

- `debug-logprob-mbs4-tok24576_20260624_025917`
- `debug-logprob-mbs4-tok32768_20260624_044059`
- `debug-logprob-mbs8-tok32768_20260625_002315`

目的：

- 测试 `old_log_prob_forward` 的 micro batch size 和 token limit 是否能提升速度且不 OOM。

现象：

- 某些 debug 中 `old_log_prob_forward` 可到 540s 左右，比正式较优设置的约 300s 慢。
- `router_mismatch_metrics` 也可能单独占 700s 甚至更高。

解释：

- `old_log_prob_forward` 是 actor 端重新 forward 计算 old logprob，不是 reward parser。
- `router_mismatch_metrics` 是对大 routed expert tensor 做比较和聚合，CPU/内存搬运会非常重。

已做优化：

- `DAPO_TASK_RUNNER_NUM_CPUS=64`
- alignment warmup 后冻结，不再每步重复比较 0/-1/1。
- 尽量保持 routed expert compact dtype，避免无意义 int64。
- unordered top8 的排序结果复用。
- 可关闭分位数和过多 per-layer 记录。

## 6. 当前 dump 数据

当前仍存在的 router dump：

| 实验 | 文件数 | 大小 | 内容 |
| --- | ---: | ---: | --- |
| overlap threshold0.045RS 20260701_012432 | 32 | 27G | tokens dump, top32 |
| overlap threshold0.045RS 20260701_202845 | 1 | 842M | tokens dump |
| overlap threshold0.5RS 20260626_070924 | 68 | 56G | tokens dump, top32 |
| sequence mismatch TIS-C2-noRS | 19 | 16G | tokens dump, top64 |
| sequence threshold0.25-TIS-C2 resume200 | 2 | 1.7G | resume 后 201-202 step dump |

已经删除或只保留 JSON 结果的 dump：

- overlap threshold0.03RS：本体 dump 已不在，保留 `threshold0.03RS_dump_sample_analysis.json`。

## 7. 关键现象和解释

### 7.1 前几十步 reward 接近 -1

观察：

- 一些 base GRPO 实验前几十步 reward 接近 -1，validate 接近 0。
- 另一个 siirl 脚本初始分数可在 0.4 左右。

当前解释：

1. Reward parser / answer format 是首要怀疑点。
2. DAPO recipe、prompt template、是否 instruct 模型、验证抽取规则都可能影响初始分数。
3. 训练集 reward 是 -1/1，验证集常按 0/1 accuracy 统计，两者尺度不同，但不能解释同配置初始差距。

结论：

- “模型不会做题”不是唯一解释。
- 更可信的是格式抽取和 recipe/template 造成的 reward 可见性差异。

### 7.2 reward 崩、length 崩、clip ratio 升高

观察：

- 某些 mismatch-only 或过滤不合适实验在约 50 step 附近 reward 下跌。
- `entropy` 下降更明显。
- `response_length/mean` 先下降几千 token，然后可能涨到接近 16K。
- `response_length/clip_ratio` 可升到 0.9。
- `pg_loss`, `kl_loss`, `grad_norm` 可先上涨后快速变化。
- step time 增加，主要在 `update_actor` 和 `old_log_prob`。

解释：

- 这更像策略分布进入不稳定区，模型生成长度策略发生塌缩/爆长。
- 长 response 会直接放大 old_log_prob 和 update_actor 的 token 计算量。
- 不是单独某个 mismatch token 造成，而是模型整体生成分布发生变化。

### 7.3 `training/rollout_probs_diff_mean` 下降但 `seq_mismatch_mean` 上升

观察：

- 训练不稳定时，`training/rollout_probs_diff_mean` 有时下降到接近 0。
- 同时 router sequence mismatch 反而升高。

解释：

- `training/rollout_probs_diff_mean` 比较的是 rollout / actor train 对已采样 token 的概率差异。
- `seq_mismatch_mean` 比较的是 MoE router 选择路径。
- 两者衡量对象不同：输出 token probability 可接近，但内部 expert path 可分叉。
- 对 MoE 来说，相同 token probability 不代表相同专家路径。

### 7.4 `response_token_match_rate` 很低但 per-layer match 高

观察：

- 每层 `layer_i/response_token_match_rate` 可在 0.8 左右。
- 但全局 `response_token_match_rate` 可能只有 0.002。

解释：

- 全局 token match 往往要求该 token 的所有 MoE 层都 match。
- 如果单层 match rate 是 0.8，48 层全部 match 的概率近似 `0.8^48 ≈ 2e-5`，自然会非常低。
- 因此不能用全局 `response_token_match_rate` 直接判断是否严重异常。

### 7.5 极端 token 是否导致整条序列崩

已分析 threshold0.5RS 和 threshold0.045RS dump。

发现：

- 极端 token 稀疏。
- token_mismatch >= 0.2 的 token 通常只占 response token 的小比例。
- 对 seq_mismatch 贡献有限。
- 极端 token 分布在中段较多，不集中在最后答案。
- 事件前后窗口没有稳定显示 “极端 token 后 mismatch 持续上升”。

解释：

- 目前没有强证据支持“少数极端 token 直接导致后续 token 自回归连锁崩坏”。
- 更像是：极端 token 是一条 response 处在不稳定路由状态下的局部表现，属于 response-level instability indicator。

### 7.6 过滤是否偏向难题

已分析 threshold0.045RS 的 prompt group。

发现：

- `group_reject_fraction` 和 `group_reward_mean` 相关性很弱。
- `group_reject_fraction` 和 `group_length_mean` 相关性也弱。
- 前 32 step 中，过滤没有明显集中在低 reward 难题上。

解释：

- sequence mismatch 当前不是纯 difficulty 指标。
- 它更接近“路由一致性/数值路径稳定性”指标。
- 如果直接过滤，可能误伤短且高 reward 的 response。

### 7.7 固定阈值的问题

观察：

- 阈值过大：不过滤，等价 mismatch-only。
- 阈值过小：前期过滤过多，后期可能几乎全过滤。
- 后期 mismatch mean 变高时，过滤比例也会变高；过滤更多后，mean 仍可能更高。

解释：

- 因为过滤不改变 rollout 分布本身，只改变训练 loss mask。
- 如果策略已经向高 mismatch 区域漂移，过滤更多不会立刻降低新生成 response 的 mismatch。
- 高过滤比例还会改变训练样本分布，可能导致 validate 崩。

结论：

- 固定阈值不是最稳的主方案。
- 建议转向 adaptive top fraction / controlled reject fraction。

## 8. 推荐后续实验

### 8.1 主线：受控过滤比例

建议从固定阈值切到比例控制：

```text
每个 step 在当前 batch 内按 seq_mismatch 排序，过滤 top p%
```

初始建议：

- warmup 0-10 step：只记录，不过滤。
- 10-50 step：过滤 8%-10%。
- 50 step 后：过滤 10%-15%。
- 如果 `seq_mismatch_mean` 快速上升，可以允许最高 25%-30%，但不建议长期 40%-50%。

必须记录：

- rejected fraction
- rejected reward mean / kept reward mean
- rejected length mean / kept length mean
- rejected prompt group reward mean
- rejected seq_mismatch mean
- val accuracy

### 8.2 exact_set vs overlap_fraction 公平对比

不要比较绝对 threshold，应该比较相同 reject fraction：

- exact_set top10
- overlap_fraction top10
- exact_set top15
- overlap_fraction top15

观察：

- 哪个对 reward/val 更稳。
- 哪个更少引入长度 bias。
- 哪个额外开销更低。

### 8.3 TIS 和 router RS 消融

建议四组：

| TIS | Router RS | 目的 |
| --- | --- | --- |
| off | off | baseline |
| on C=2 | off | 只看概率比 correction |
| off | on top10 | 只看 router RS |
| on C=2 | on top10 | 看互补或冲突 |

### 8.4 继续分析 dump

需要继续关注：

- response 内 token mismatch 轨迹。
- rejected vs kept 的 reward/length/accuracy。
- 按 prompt group 的 8 条 response 是否整体都低 reward。
- 高 mismatch response 是否更容易 clip 到 16K。
- mismatch 突增前后的 entropy、clip ratio、length 是否有先导信号。

## 9. 代码改动记录

当前 `verl-sequence` 的关键本地改动：

1. `verl/utils/router_mismatch_metrics.py`
   - 支持 exact_set 和 overlap_fraction 两种 metric mode。
   - 支持 sequence-level `token_mismatch` / `seq_mismatch`。
   - top8 expert 比较忽略顺序。
   - alignment warmup 后固定。
   - 尽量保持 compact dtype，减少 int64。

2. `dapo/dapo_ray_trainer.py`
   - 增加 router mismatch metrics timing。
   - 增加 router mismatch RS。
   - 增加 router analysis dump。
   - 增加 reward debug dump。

3. `verl/utils/reward_score/math_dapo.py`
   - reward parser 优先解析最后 boxed answer，fallback 到 `Answer:`。

4. `verl/trainer/main_ppo.py` / `dapo/main_dapo.py`
   - 支持 `DAPO_TASK_RUNNER_NUM_CPUS=64`，缓解 CPU 端 mismatch 计算瓶颈。

5. `verl/utils/megatron/dist_checkpointing.py`
   - 增加 Megatron dist checkpoint metadata 兼容 patch，用于严格 resume 时绕过 `.metadata` 缺少 `mcore_data` 的问题。

## 10. 论文和资料

### PPO / GRPO / DAPO

- Schulman et al., 2017, *Proximal Policy Optimization Algorithms*, arXiv:1707.06347.
  https://arxiv.org/abs/1707.06347

- Shao et al., 2024, *DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models*, arXiv:2402.03300.
  该文引入 GRPO，说明 GRPO 是 PPO 的变体，用 group-relative advantage 降低 value model 需求。
  https://arxiv.org/abs/2402.03300

- Yu et al., 2025, *DAPO: An Open-Source LLM Reinforcement Learning System at Scale*, arXiv:2503.14476.
  DAPO 提出 decoupled clipping、dynamic sampling 等 recipe，面向大规模可复现 RL 训练。
  https://arxiv.org/abs/2503.14476

### FP8 / vLLM / verl

- verl 文档：*verl 中的 FP8 强化学习*。
  文档明确区分仅 FP8 rollout 和 FP8 E2E，并在 Qwen3-30B-A3B MoE 实验中使用 token-level TIS C=2；文档也指出 MoE 模型 rollout/train mismatch 更高，即使 BF16 也需要 rollout correction。
  https://verl.org.cn/en/latest/advance/fp8.html

- Qiu et al., 2026, *FP8-RL: A Practical and Stable Low-Precision Stack for LLM Reinforcement Learning*, arXiv:2601.18150.
  verl FP8 文档引用的技术报告。
  https://arxiv.org/abs/2601.18150

- Micikevicius et al., 2022, *FP8 Formats for Deep Learning*, arXiv:2209.05433.
  讨论 E4M3/E5M2 FP8 格式和深度学习训练/推理适用性。
  https://arxiv.org/abs/2209.05433

### Off-policy / correction 相关

- PPO 的 clipped ratio 本质上是在限制 policy update 的 off-policy 偏移。
- TIS 用 train/rollout token probability ratio 对 loss 做 token-level correction，并用阈值裁剪极端 ratio。
- 当前 router mismatch RS 不做概率重加权，而是根据 MoE routing path 的 sequence-level mismatch 过滤整条 response。

## 11. 当前判断

1. 初始 reward 低，优先解释是格式抽取/模板不一致，而不是模型能力异常。
2. reward 崩更像长度分布和 policy update 不稳定共同导致，不应简单归因于单个 extreme token。
3. router mismatch 是有价值的诊断信号，但不是直接 reward/quality 指标。
4. 固定阈值 RS 不稳，因为 mismatch 分布随训练阶段漂移。
5. 更合理的过滤策略是 controlled reject fraction，并且必须监控 rejected 的 reward、length、prompt group。
6. overlap_fraction 比 exact_set 更平滑，但绝对阈值必须重新标定。
7. TIS 仍应作为强 baseline，因为官方 FP8-RL/verl 文档在 Qwen3-30B-A3B MoE 上也采用 token-level TIS C=2。
8. mismatch dump 很有分析价值，但 full/token dump 成本高，建议只在诊断实验中启用，正式长跑只保留 summary 或低频 dump。
