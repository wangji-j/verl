# FP8 Rollout MoE Router Stability 实验计划

## 1. 研究目标

本研究关注以下训练设置：

- rollout：vLLM FP8
- train：Megatron BF16
- 模型：Qwen3-30B-A3B Base
- 算法：GRPO
- 训练数据：DeepScaleR
- 数学验证：AIME24/AIME25

FP8 rollout 与 BF16 train 天然存在数值和执行路径差异。研究目标不是强制两端完全一致，而是识别会造成训练风险的训推偏移，并用尽可能轻量、低偏置的方法降低训练崩溃风险。

核心研究问题：

1. 哪种 expert usage 分布距离最适合衡量当前 MoE 训推路由漂移？
2. 过滤多少 response 能稳定训练，同时不过度减少有效训练信号？
3. 过滤是否带来长度、对错和题目难度偏置？
4. 路由漂移对策略崩溃是否存在滞后效应？
5. 随着一个 rollout batch 内 optimizer mini-step 数增加，Router RS、TIS、R2、R3 分别能缓解哪类 off-policy 问题？
6. 数学场景中的结论能否推广到通用 reasoning 和 coding？

## 2. 当前假设

### H1：路由漂移是风险信号，但不是策略风险本身

Expert usage drift 可能早于或伴随以下指标恶化：

- `training/rollout_probs_diff_mean`
- `training/rollout_actor_probs_pearson_corr`
- `rollout_corr/log_ppl_diff`
- `actor/pg_clipfrac`
- `actor/ppo_kl`
- `actor/grad_norm`
- validation accuracy

路由漂移与输出概率偏移不要求同步。路由变化可能暂时被网络其余部分补偿，直到偏移累积超过稳定区间后才反映到 reward、长度和 validation。

### H2：距离指标必须匹配当前漂移的粒度

当前 expert usage drift 较分散，不只是单一专家的极端变化。因此：

- Smooth TV/L1 是当前主候选；
- L2 可作为更强调集中漂移的对照；
- Hellinger 和 JS 可作为分布几何对照；
- `L∞` 只适合检测单专家尖峰，不宜直接作为唯一过滤依据；
- KL 对低计数和零概率敏感，短序列中容易被采样噪声放大，不作为首轮主过滤指标。

### H3：固定过滤比例比固定原始阈值更适合非平稳训练

已有固定阈值实验出现：

- 前期过滤不足；
- 后期过滤比例快速上升；
- validation 与 reward 走势分离；
- 长度和正确率偏置。

因此首轮比例消融使用 MAD-Z 条件标准化后进行全局 Top-k 过滤，并限制每个 prompt 最多过滤 2 条 response。

### H4：TIS、R2、R3 和 Router RS 解决的问题不同

| 方法 | 主要作用 | 不直接解决的问题 |
|---|---|---|
| TIS-C2 | 用 token 级 train/rollout 概率比修正 policy gradient，并裁剪极端比值 | 不强制两端选择相同专家 |
| R2 | 回放 train old-log-prob forward 的路由，使 update forward 与旧训练前向更一致 | 不直接对齐 vLLM rollout 路由 |
| R3 | 将 rollout 路由带入训练侧回放，尽量对齐 rollout、old-log-prob 和 update | 不能消除全部 FP8/BF16 数值差异 |
| Router RS | 丢弃路由漂移风险较高的整条 response | 会减少训练 token，并可能引入选择偏置 |

## 3. 固定实验设置

除专门消融的变量外，以下配置保持不变：

| 项目 | 固定值 |
|---|---|
| 模型 | Qwen3-30B-A3B Base |
| rollout 精度 | FP8 |
| train 精度 | BF16 |
| 训练引擎 | Megatron |
| rollout 引擎 | vLLM |
| 算法 | GRPO |
| prompt batch size | 256 |
| 每个 prompt 的 response 数 | 8 |
| 每步 response 数 | 2048 |
| 最大 response 长度 | 16384 |
| 学习率 | 3e-6 |
| KL loss | `low_var_kl`, coefficient 0.001 |
| validation 间隔 | 10 steps |
| checkpoint 间隔 | 30 steps |
| checkpoint 保留数 | 1 |
| 主训练资源 | 2 nodes × 8 H200 |

所有比较实验必须额外记录：

- 完整启动命令；
- Git commit；
- 镜像、PyTorch、Megatron Core、vLLM 版本；
- 随机种子；
- checkpoint 是否 resume；
- 实际过滤 response 比例和 token 比例；
- 每步 wall time 和各阶段 timing。

## 4. 实验阶段总览

| 阶段 | 目的 | 主要变量 | 初筛规模 | 最终验证 |
|---|---|---|---|---|
| P1 | 选择距离指标 | TV、L2、L∞、Hellinger、JS | 30–50 steps，1 seed | 候选 2 个，3 seeds |
| P2 | 选择过滤比例 | 0%、1%、2%、3% | 100–150 steps，1 seed | 最优 2 个比例，3 seeds |
| P3 | 验证长度/难度偏置 | MAD-Z、prompt cap、条件校准 | 与 P2 同步分析 | 最终配置报告 |
| P4 | 扩展到 reasoning/coding | 数据集和 reward verifier | 小规模 smoke test | 每领域 3 seeds |
| P5 | off-policy 压力测试 | mini-step 1/2/4/8 | 先测 1 和 8 | 再补 2 和 4 |
| P6 | 方法对照 | None、TIS、R2、R3、RS、RS+TIS | 端点筛选 | 完整矩阵的必要子集 |

---

## 5. P1：距离指标选择

### 5.1 目标

先不让 Router RS 改变训练样本，只记录同一条 response、同一 MoE 层上的 rollout/train expert usage counts 和候选距离。这样可以避免“指标选择”和“过滤效果”互相污染。

每条 response、每层得到两侧专家频率分布：

```text
p_rollout(layer, expert | response)
p_train(layer, expert | response)
```

当前 probe 记录：

- raw TV/L1；
- shrinkage 后 TV/L1；
- L2；
- L∞；
- Hellinger²；
- normalized JS；
- effective support；
- rollout/train expert counts。

### 5.2 主实验：TIS-C2 + no Router RS

```bash
cd /inspire/hdd/project/qianghuaxuexi/hujiarui-25046/verl-sequence

EXPERIMENT_NAME_BASE=GRPO-DEEPSCALER-Qwen3-30B-A3B-base-MEGATRON-VLLM-FP8-distance-probe-noRS-TIS-C2-16K \
ENABLE_ROUTER_MISMATCH_RS=False \
VAL_BEFORE_TRAIN=False \
TOTAL_TRAINING_STEPS=30 \
SAVE_FREQ=30 \
VERL_ROUTER_ANALYSIS_DUMP_MODE=expert_counts \
VERL_ROUTER_ANALYSIS_DUMP_EVERY_N=1 \
VERL_ROUTER_ANALYSIS_DUMP_STEPS=0 \
VERL_ROUTER_ANALYSIS_DUMP_SAMPLES=0 \
VERL_ROUTER_ANALYSIS_DUMP_TOPK_TOKENS=0 \
bash low_precision/run_grpo_qwen3_30b_a3b_expert_distribution_distance_probe_top8RS.sh \
  router.enable_mismatch_rs=False
```

说明：

- 开启 TIS-C2；
- 不做 Router RS；
- 不做训练前验证，节省 probe 启动时间；
- 每一步保存 compact expert counts；
- 不保存体积较大的 token 轨迹。

### 5.3 对照实验：no TIS + no Router RS

```bash
cd /inspire/hdd/project/qianghuaxuexi/hujiarui-25046/verl-sequence

EXPERIMENT_NAME_BASE=GRPO-DEEPSCALER-Qwen3-30B-A3B-base-MEGATRON-VLLM-FP8-distance-probe-noRS-noTIS-16K \
ENABLE_ROUTER_MISMATCH_RS=False \
VAL_BEFORE_TRAIN=False \
TOTAL_TRAINING_STEPS=30 \
SAVE_FREQ=30 \
VERL_ROUTER_ANALYSIS_DUMP_MODE=expert_counts \
VERL_ROUTER_ANALYSIS_DUMP_EVERY_N=1 \
VERL_ROUTER_ANALYSIS_DUMP_STEPS=0 \
VERL_ROUTER_ANALYSIS_DUMP_SAMPLES=0 \
VERL_ROUTER_ANALYSIS_DUMP_TOPK_TOKENS=0 \
bash low_precision/run_grpo_qwen3_30b_a3b_expert_distribution_l1_top8RS.sh \
  algorithm.rollout_correction.rollout_is=null \
  algorithm.rollout_correction.rollout_rs=null \
  router.enable_mismatch_rs=False
```

### 5.4 离线分析

对每种候选距离执行：

1. 计算不同距离的 Spearman rank correlation；
2. 比较 Top1%、Top2%、Top3%、Top5%、Top8% response 集合的 Jaccard；
3. 计算距离与 response length、reward、正确率、prompt group 难度的关系；
4. 检查短序列方差和 effective support；
5. 计算每层漂移以及层间聚合后的稳定性；
6. 做 lead-lag 分析，检查距离是否领先于：
   - `pg_clipfrac`
   - `ppo_kl`
   - `grad_norm`
   - `rollout_probs_diff_mean`
   - validation drop
7. 比较计算开销和 dump 体积。

### 5.5 距离选择标准

距离指标按以下优先级选择：

1. 对后续训练不稳定有可重复的关联；
2. 对长度、正确率和 prompt 难度的非预期偏置较小；
3. 跨 step 和随机种子排序稳定；
4. 能区分正常采样噪声与系统性漂移；
5. 计算和存储开销可接受。

当前预期主候选为 smooth TV，备选为 L2。该结论必须由 P1 dump 数据确认后再固定。

---

## 6. P2：MAD-Z 过滤比例消融

### 6.1 比较组

保持 TIS-C2，使用同一 MAD-Z、同一 prompt cap，仅改变过滤比例：

| 组别 | Router RS | TIS | 过滤比例 |
|---|---:|---:|---:|
| M0 | off | on | 0% |
| M1 | on | on | 1% |
| M2 | on | on | 2% |
| M3 | on | on | 3% |
| 历史参考 | on | on | 5% / 8% |

首轮不引入 batch refill，以免无法区分“过滤比例变化”和“补样策略变化”。

### 6.2 TIS-only 基线

```bash
cd /inspire/hdd/project/qianghuaxuexi/hujiarui-25046/verl-sequence

EXPERIMENT_NAME_BASE=GRPO-DEEPSCALER-Qwen3-30B-A3B-base-MEGATRON-VLLM-FP8-TIS-C2-noRS-16K \
ENABLE_ROUTER_MISMATCH_RS=False \
SAVE_FREQ=30 \
VERL_REWARD_DEBUG_STEPS=0 \
VERL_REWARD_DEBUG_SAMPLES=0 \
VERL_ROUTER_ANALYSIS_DUMP_MODE=expert_counts \
VERL_ROUTER_ANALYSIS_DUMP_EVERY_N=5 \
VERL_ROUTER_ANALYSIS_DUMP_STEPS=0 \
VERL_ROUTER_ANALYSIS_DUMP_TOPK_TOKENS=0 \
bash low_precision/run_grpo_qwen3_30b_a3b_expert_distribution_l1_madz_top8RS_TIS.sh \
  router.enable_mismatch_rs=False
```

### 6.3 MAD-Z Top1%

```bash
cd /inspire/hdd/project/qianghuaxuexi/hujiarui-25046/verl-sequence

ROUTER_MISMATCH_RS_FRACTION=0.01 \
EXPERIMENT_NAME_BASE=GRPO-DEEPSCALER-Qwen3-30B-A3B-base-MEGATRON-VLLM-FP8-usage-l1-madz-top1RS-cap2-TIS-C2-16K \
SAVE_FREQ=30 \
VERL_REWARD_DEBUG_STEPS=0 \
VERL_REWARD_DEBUG_SAMPLES=0 \
VERL_ROUTER_ANALYSIS_DUMP_MODE=expert_counts \
VERL_ROUTER_ANALYSIS_DUMP_EVERY_N=5 \
VERL_ROUTER_ANALYSIS_DUMP_STEPS=0 \
VERL_ROUTER_ANALYSIS_DUMP_TOPK_TOKENS=0 \
bash low_precision/run_grpo_qwen3_30b_a3b_expert_distribution_l1_madz_top8RS_TIS.sh
```

### 6.4 MAD-Z Top2%

```bash
cd /inspire/hdd/project/qianghuaxuexi/hujiarui-25046/verl-sequence

ROUTER_MISMATCH_RS_FRACTION=0.02 \
EXPERIMENT_NAME_BASE=GRPO-DEEPSCALER-Qwen3-30B-A3B-base-MEGATRON-VLLM-FP8-usage-l1-madz-top2RS-cap2-TIS-C2-16K \
SAVE_FREQ=30 \
VERL_REWARD_DEBUG_STEPS=0 \
VERL_REWARD_DEBUG_SAMPLES=0 \
VERL_ROUTER_ANALYSIS_DUMP_MODE=expert_counts \
VERL_ROUTER_ANALYSIS_DUMP_EVERY_N=5 \
VERL_ROUTER_ANALYSIS_DUMP_STEPS=0 \
VERL_ROUTER_ANALYSIS_DUMP_TOPK_TOKENS=0 \
bash low_precision/run_grpo_qwen3_30b_a3b_expert_distribution_l1_madz_top8RS_TIS.sh
```

### 6.5 MAD-Z Top3%

```bash
cd /inspire/hdd/project/qianghuaxuexi/hujiarui-25046/verl-sequence

ROUTER_MISMATCH_RS_FRACTION=0.03 \
EXPERIMENT_NAME_BASE=GRPO-DEEPSCALER-Qwen3-30B-A3B-base-MEGATRON-VLLM-FP8-usage-l1-madz-top3RS-cap2-TIS-C2-16K \
SAVE_FREQ=30 \
VERL_REWARD_DEBUG_STEPS=0 \
VERL_REWARD_DEBUG_SAMPLES=0 \
VERL_ROUTER_ANALYSIS_DUMP_MODE=expert_counts \
VERL_ROUTER_ANALYSIS_DUMP_EVERY_N=5 \
VERL_ROUTER_ANALYSIS_DUMP_STEPS=0 \
VERL_ROUTER_ANALYSIS_DUMP_TOPK_TOKENS=0 \
bash low_precision/run_grpo_qwen3_30b_a3b_expert_distribution_l1_madz_top8RS_TIS.sh
```

### 6.6 评估指标

主指标：

- AIME24/AIME25 validation accuracy；
- validation peak、final、峰值后下降幅度；
- validation curve AUC；
- 首次持续崩溃 step；
- wall-clock 到达同一 validation 水平的时间。

训练稳定性：

- `actor/entropy`
- `actor/pg_clipfrac`
- `actor/ppo_kl`
- `actor/kl_loss`
- `actor/grad_norm`
- `training/rollout_probs_diff_mean`
- `training/rollout_actor_probs_pearson_corr`
- `rollout_corr/log_ppl_diff`
- `rollout_corr/ppl_ratio`
- response length 与 clip ratio

过滤行为：

- response reject fraction；
- rejected token fraction；
- 被过滤和保留 response 的平均/中位长度；
- 被过滤和保留 response 的正确率；
- 每个 prompt 被过滤 0/1/2 条的比例；
- 简单/中等/困难 prompt group 的过滤率；
- 有效训练 token 数。

### 6.7 决策规则

1. 单 seed 跑到至少出现一次历史崩溃窗口，建议 100–150 steps；
2. 如果某组出现明显 OOM、NaN 或 validation 连续 3 次评估恶化，先停止并保留 dump；
3. 选出稳定性最好且过滤最少的两个比例；
4. 两个候选比例各跑 3 seeds；
5. 只有在 3 seeds 上都优于 TIS-only，才进入跨领域实验。

---

## 7. P3：偏置与机制分析

### 7.1 长度偏置

报告以下两种统计，不能只看 `response_length/mean`：

- response 级过滤比例；
- token 级过滤比例。

如果过滤 2% response 却移除远低于 2% 的 token，说明偏向短序列；反之则偏向长序列。

同时按长度分桶报告：

```text
[0, 2048)
[2048, 4096)
[4096, 8192)
[8192, 12288)
[12288, 16384]
```

### 7.2 对错偏置

分别报告：

- rejected accuracy；
- retained accuracy；
- 在每个长度桶内的 rejected/retained accuracy；
- 控制长度后的正确率差异。

这样可以区分：

- “正确 response 更短，所以更容易被过滤”；
- “在相同长度下，正确 response 仍更容易被过滤”。

### 7.3 Prompt group 难度

一个 prompt 的 8 条 response 作为一组。主要难度代理使用该组 reward/accuracy 均值，而不是长度：

```text
group_accuracy = 8 条 response 中正确 response 的比例
```

分组建议：

- 困难：`group_accuracy <= 0.25`
- 中等：`0.25 < group_accuracy < 0.75`
- 简单：`group_accuracy >= 0.75`

报告每组过滤率、平均过滤条数，以及 cap2 是否真正生效。

### 7.4 滞后效应

对 router drift 和策略/验证指标做：

- step-wise cross correlation；
- 一阶差分相关；
- drift 超过历史分位点后的事件研究；
- 检查 drift 在第 `t` 步是否预测 `t+1` 到 `t+K` 的 validation 或 policy 指标。

这一步用于验证“路由漂移先累积，策略输出后崩溃”的假设，而不是只做同时相关。

---

## 8. P4：推广到 Reasoning 和 Coding

### 8.1 原则

先在数学场景固定：

- 距离定义；
- shrinkage 参数；
- MAD-Z 校准；
- 过滤比例；
- prompt cap。

跨领域首轮不重新调参，以检验泛化性。只有确认失效机制后再进行领域特定调参。

### 8.2 Reasoning

建议顺序：

1. 单轮、可自动评分的 reasoning 数据；
2. GPQA/MMLU-Pro 等独立验证；
3. 再扩展到多轮或工具调用 reasoning。

需要记录：

- domain/task id；
- response length；
- reward verifier 类型；
- 各 domain 的过滤率和 validation；
- 是否存在领域特定的 router drift 基线。

### 8.3 Coding

建议先使用可执行 unit tests 的单轮代码生成任务，再考虑 SWE-bench 一类 agentic coding。

原因：

- 单轮 coding 能维持与数学实验相近的数据流；
- unit tests 提供明确 reward；
- SWE-bench 同时引入工具、环境、轨迹长度和多轮交互，难以定位变量。

### 8.4 实现状态

当前仓库已有数学训练脚本，但没有完成并核验 reasoning/coding 专用数据、reward manager 和验证脚本。因此本阶段命令暂不写成“可直接运行”，避免复用错误的 DAPO 数学 scorer。

实施前需要新增并 smoke test：

- reasoning 数据适配脚本；
- coding prompt/template；
- coding sandbox/test reward；
- 独立验证集；
- 对应实验启动脚本。

---

## 9. P5：Off-policy Mini-step 压力测试

### 9.1 Mini-step 定义

当前每步：

- 256 prompts；
- 每个 prompt 生成 8 responses；
- 共 2048 responses；
- `ppo_epochs=1`。

`TRAIN_PROMPT_MINI_BSZ=256` 时，一个 rollout batch 只做 1 个 optimizer mini-step。若保持总 batch 不变，则：

| 目标 mini-step 数 | `TRAIN_PROMPT_MINI_BSZ` | 每个 mini-step 的 response 数 |
|---:|---:|---:|
| 1 | 256 | 2048 |
| 2 | 128 | 1024 |
| 4 | 64 | 512 |
| 8 | 32 | 256 |

后续 mini-step 使用前面 mini-step 已更新过的参数，但 old log probs 和 rollout 数据仍来自同一旧策略，因此 mini-step 越多，后续数据越 off-policy。

这里改变的是 PPO mini-batch，不是 GPU micro-batch。`ppo_micro_batch_size_per_gpu` 暂时保持不变。

### 9.2 方法矩阵

| 方法 ID | Router RS | TIS | R2 | R3 |
|---|---:|---:|---:|---:|
| O0 | off | off | off | off |
| O1 | off | on | off | off |
| O2 | off | off | on | off |
| O3 | off | off | off | on |
| O4 | best MAD-Z RS | off | off | off |
| O5 | best MAD-Z RS | on | off | off |

完整矩阵为 4 个 mini-step × 6 种方法，共 24 组。为节省资源：

1. 先在 mini-step=1 和 8 上测试全部 6 种方法；
2. 淘汰明显无效或成本过高的方法；
3. 只对保留方法补 mini-step=2 和 4；
4. 最终关键比较跑 3 seeds。

### 9.3 O0：无修正基线模板

以 mini-step=8 为例：

```bash
cd /inspire/hdd/project/qianghuaxuexi/hujiarui-25046/verl-sequence

TRAIN_PROMPT_MINI_BSZ=32 \
ENABLE_ROUTER_MISMATCH_RS=False \
EXPERIMENT_NAME_BASE=GRPO-DEEPSCALER-Qwen3-30B-A3B-base-MEGATRON-VLLM-FP8-offpolicy-k8-noCorrection-16K \
SAVE_FREQ=30 \
VERL_REWARD_DEBUG_STEPS=0 \
VERL_REWARD_DEBUG_SAMPLES=0 \
VERL_ROUTER_ANALYSIS_DUMP_MODE=off \
bash low_precision/run_grpo_qwen3_30b_a3b_expert_distribution_l1_top8RS.sh \
  algorithm.rollout_correction.rollout_is=null \
  algorithm.rollout_correction.rollout_rs=null \
  router.enable_mismatch_rs=False
```

将 `TRAIN_PROMPT_MINI_BSZ` 替换为 256/128/64/32，即可对应 k=1/2/4/8。

### 9.4 O1：TIS-C2

```bash
cd /inspire/hdd/project/qianghuaxuexi/hujiarui-25046/verl-sequence

TRAIN_PROMPT_MINI_BSZ=32 \
ENABLE_ROUTER_MISMATCH_RS=False \
EXPERIMENT_NAME_BASE=GRPO-DEEPSCALER-Qwen3-30B-A3B-base-MEGATRON-VLLM-FP8-offpolicy-k8-TIS-C2-16K \
SAVE_FREQ=30 \
VERL_REWARD_DEBUG_STEPS=0 \
VERL_REWARD_DEBUG_SAMPLES=0 \
VERL_ROUTER_ANALYSIS_DUMP_MODE=off \
bash low_precision/run_grpo_qwen3_30b_a3b_expert_distribution_l1_madz_top8RS_TIS.sh \
  router.enable_mismatch_rs=False
```

### 9.5 O4：MAD-Z RS only

以下 `<BEST_FRACTION>` 在 P2 完成后替换：

```bash
cd /inspire/hdd/project/qianghuaxuexi/hujiarui-25046/verl-sequence

TRAIN_PROMPT_MINI_BSZ=32 \
ROUTER_MISMATCH_RS_FRACTION=<BEST_FRACTION> \
EXPERIMENT_NAME_BASE=GRPO-DEEPSCALER-Qwen3-30B-A3B-base-MEGATRON-VLLM-FP8-offpolicy-k8-madzRS-noTIS-16K \
SAVE_FREQ=30 \
VERL_REWARD_DEBUG_STEPS=0 \
VERL_REWARD_DEBUG_SAMPLES=0 \
VERL_ROUTER_ANALYSIS_DUMP_MODE=off \
bash low_precision/run_grpo_qwen3_30b_a3b_expert_distribution_l1_madz_top8RS_TIS.sh \
  algorithm.rollout_correction.rollout_is=null \
  algorithm.rollout_correction.rollout_rs=null
```

### 9.6 O5：MAD-Z RS + TIS

```bash
cd /inspire/hdd/project/qianghuaxuexi/hujiarui-25046/verl-sequence

TRAIN_PROMPT_MINI_BSZ=32 \
ROUTER_MISMATCH_RS_FRACTION=<BEST_FRACTION> \
EXPERIMENT_NAME_BASE=GRPO-DEEPSCALER-Qwen3-30B-A3B-base-MEGATRON-VLLM-FP8-offpolicy-k8-madzRS-TIS-C2-16K \
SAVE_FREQ=30 \
VERL_REWARD_DEBUG_STEPS=0 \
VERL_REWARD_DEBUG_SAMPLES=0 \
VERL_ROUTER_ANALYSIS_DUMP_MODE=off \
bash low_precision/run_grpo_qwen3_30b_a3b_expert_distribution_l1_madz_top8RS_TIS.sh
```

### 9.7 O2：R2

```bash
cd /inspire/hdd/project/qianghuaxuexi/hujiarui-25046/verl-sequence

WORKING_DIR=/inspire/hdd/project/qianghuaxuexi/hujiarui-25046/verl-sequence \
RECIPE_DIR=/inspire/hdd/project/qianghuaxuexi/hujiarui-25046/verl-sequence \
TRAIN_PROMPT_MINI_BSZ=32 \
ROUTING_REPLAY_MODE=R2 \
ENABLE_ROLLOUT_ROUTING_REPLAY=False \
EXPERIMENT_NAME_BASE=GRPO-DEEPSCALER-Qwen3-30B-A3B-base-MEGATRON-VLLM-FP8-offpolicy-k8-R2-16K \
SAVE_FREQ=30 \
bash low_precision/run_grpo_qwen3_moe_30b_megatron_fp8_rollout_r3_local.sh
```

注意：R2 模式必须设置 `ENABLE_ROLLOUT_ROUTING_REPLAY=False`。

### 9.8 O3：R3

```bash
cd /inspire/hdd/project/qianghuaxuexi/hujiarui-25046/verl-sequence

WORKING_DIR=/inspire/hdd/project/qianghuaxuexi/hujiarui-25046/verl-sequence \
RECIPE_DIR=/inspire/hdd/project/qianghuaxuexi/hujiarui-25046/verl-sequence \
TRAIN_PROMPT_MINI_BSZ=32 \
ROUTING_REPLAY_MODE=R3 \
ENABLE_ROLLOUT_ROUTING_REPLAY=True \
EXPERIMENT_NAME_BASE=GRPO-DEEPSCALER-Qwen3-30B-A3B-base-MEGATRON-VLLM-FP8-offpolicy-k8-R3-16K \
SAVE_FREQ=30 \
bash low_precision/run_grpo_qwen3_moe_30b_megatron_fp8_rollout_r3_local.sh
```

### 9.9 R2/R3 前置检查

当前 R3 脚本默认的 `WORKING_DIR` 和 `RECIPE_DIR` 仍指向旧的 `jly-verl/public` 路径，因此上述命令显式覆盖到 `verl-sequence`。正式大任务前必须跑 1–2 steps smoke test，确认：

- 实际 import 路径来自 `verl-sequence`；
- vLLM 能返回 routed experts；
- routed expert tensor 非空；
- R2/R3 replay 指标出现；
- 没有命中旧版本 routing map 静默空操作问题；
- 当前 vLLM/Megatron 镜像与 checkpoint 格式一致。

---

## 10. P6：方法组合的后续消融

完成主矩阵后，再考虑以下组合：

- R2 + TIS；
- R3 + TIS；
- R2 + Router RS；
- R3 + Router RS。

这些组合不放入第一轮，原因是它们同时改变路由约束、概率修正和样本选择，若直接获得收益，很难定位贡献来源。

推荐顺序：

1. 单独确认 TIS、R2、R3、RS 的作用；
2. 在最高 off-policy 压力 k=8 下测试组合；
3. 只将有明确互补性的组合扩展到 k=2/4；
4. 对最佳组合跑 3 seeds。

## 11. 统一评价与统计

### 11.1 对齐比较尺度

Router RS 会减少训练 response/token。所有结果同时按以下横轴报告：

- optimizer step；
- rollout 生成 token；
- 实际参与 loss 的 token；
- wall-clock time。

否则过滤组和未过滤组仅按 step 比较并不完全公平。

### 11.2 随机种子

- 初筛：1 seed；
- 候选确认：3 seeds；
- 最终跨方法结论：至少 3 seeds。

尽可能固定 prompt 顺序和采样 seed，使用 paired comparison。

### 11.3 核心汇总表

每个实验最终至少给出：

| 类别 | 指标 |
|---|---|
| 性能 | best/final val、val AUC、reward |
| 稳定性 | collapse step、entropy、clipfrac、KL、grad norm |
| 训推偏移 | probs diff、Pearson、log PPL diff、router distance |
| 过滤 | response/token reject fraction、group cap |
| 偏置 | rejected/retained length、accuracy、group difficulty |
| 成本 | gen、old log prob、router metrics、update actor、testing、step time |

## 12. 实验命名规范

推荐格式：

```text
GRPO-Qwen3-30B-A3B-FP8R-BF16T-{domain}-{metric}-{correction}-{fraction}-k{mini_steps}-seed{seed}
```

示例：

```text
GRPO-Qwen3-30B-A3B-FP8R-BF16T-math-TV-MADZRS-TIS-C2-top2-k4-seed1
```

命名中必须区分：

- `noRS` / `top1RS` / `top2RS` / `top3RS`；
- `noTIS` / `TIS-C2`；
- `R2` / `R3`；
- `k1` / `k2` / `k4` / `k8`；
- `resume` 与从初始权重启动。

## 13. Dump 与存储策略

### 距离选择阶段

- `expert_counts`
- 每一步 dump
- 不保存 top token
- 30–50 steps

### 长训练阶段

- 前 30 steps 可每步 dump；
- 后续每 5 steps dump；
- 默认使用 `expert_counts`；
- 只有分析 token 位置轨迹时才单独开启 `tokens`；
- checkpoint 每 30 steps，最多保留 1 个。

Dump 目录必须位于对应实验 checkpoint 目录下，避免多个实验写入同一目录。

## 14. 执行优先级

### 第一优先级

1. 完成 P1 clean distance probe；
2. 离线确定 smooth TV 与 L2 中的主指标；
3. 跑 P2 的 0%、1%、2%、3%；
4. 选择两个比例做 3 seeds。

### 第二优先级

1. k=1 与 k=8 的 O0–O5；
2. 判断 R2/R3 是否在高 off-policy 下具有明显优势；
3. 对保留方法补 k=2/4。

### 第三优先级

1. 单轮 reasoning；
2. 单轮 coding + unit tests；
3. 最后扩展 agentic coding。

## 15. 当前已实现与待实现

### 已实现

- Expert Usage smooth TV/L1；
- compact expert counts dump；
- 多距离 probe；
- MAD-Z + global top fraction；
- prompt 最大过滤 2 条；
- TIS-C2；
- R2/R3 基础脚本；
- checkpoint 保留 1 个；
- 长度、正确率和 prompt group 的离线分析基础。

### 待核验

- 距离 probe 的最终主指标；
- 1%/2%/3% 最佳过滤比例；
- R2/R3 在 `verl-sequence` 当前镜像中的完整 smoke test；
- R3 routed expert 返回和 replay 指标；
- mini-step 2/4/8 的实际 optimizer step 计数日志。

### 待实现

- reasoning 专用训练/验证脚本；
- coding sandbox reward；
- 跨领域统一 dump schema；
- 自动生成实验汇总表；
- 按有效训练 token 对齐的曲线；
- 多随机种子置信区间报告。

## 16. 最终论文级证据链

最终结论需要按以下顺序建立：

1. FP8 rollout/BF16 train 下存在可测量的 router drift；
2. 该 drift 与后续策略不稳定存在时间关系，而非仅同时相关；
3. 所选距离比替代距离更稳定、偏置更小或预测性更强；
4. 小比例 Router RS 能延迟/避免崩溃；
5. 收益不是简单来自减少训练 token；
6. 长度、正确率和 prompt 难度偏置受控；
7. 在更高 mini-step off-policy 条件下仍有效；
8. 与 TIS、R2、R3 的作用边界和互补性清晰；
9. 结果能从数学推广到至少一个 reasoning 和一个 coding 场景。
