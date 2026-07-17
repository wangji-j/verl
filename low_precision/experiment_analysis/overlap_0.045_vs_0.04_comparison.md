# overlap 0.045RS vs 0.04RS 对比分析

分析对象：

```text
0.045:
/inspire/hdd/project/qianghuaxuexi/hujiarui-25046/ckpts/jly-DAPO-DEEPSCALER-FP8-ROLLOUT/GRPO-DEEPSCALER-Qwen3-30B-A3B-base-MEGATRON-VLLM-FP8-overlap-mismatch-threshold0.045RS-16K_20260701_202845/router_analysis_dump

0.04:
/inspire/hdd/project/qianghuaxuexi/hujiarui-25046/ckpts/jly-DAPO-DEEPSCALER-FP8-ROLLOUT/GRPO-DEEPSCALER-Qwen3-30B-A3B-base-MEGATRON-VLLM-FP8-overlap-mismatch-threshold0.04RS-16K_20260703_020122/router_analysis_dump
```

聚合 JSON：

```text
/inspire/hdd/project/qianghuaxuexi/hujiarui-25046/verl-sequence/low_precision/experiment_analysis/overlap_threshold0.045RS_20260701_202845_analysis.json
/inspire/hdd/project/qianghuaxuexi/hujiarui-25046/verl-sequence/low_precision/experiment_analysis/overlap_threshold0.04RS_20260703_020122_analysis.json
```

## 1. 总体结论

两个实验都使用 `overlap_fraction` sequence mismatch，但阈值不同：

- `0.045RS`：过滤较弱，step5-50 多数只有 0.5%-3%，step55 升到 15.8%。
- `0.04RS`：过滤明显更强，前 55 step 平均约 9.3%；step65 后迅速升到 30%-50%。

核心现象：

1. 两组前期都出现 train reward 上升、response length 快速下降。
2. `0.04RS` 虽然过滤更强，但没有阻止早期短答案化。
3. `0.04RS` 后期进入长输出、高截断、高过滤、低 reward 的不稳定阶段。
4. `0.045RS` 只 dump 到 step55，已经看到 step55 的过滤和 mismatch 突增，但没有后续 dump 判断是否会像 0.04 一样继续恶化。
5. 极端 token 不是主要原因；更像是整体生成分布和长度分布漂移。
6. train reward 初期不崩，是因为训练集上短答案/格式化策略仍能提高 reward；val 更早崩，是因为这种策略泛化差。

## 2. 公平窗口对比：step 5-55

两个实验共同覆盖 step5-55。这个窗口内平均值：

| 指标 | 0.045RS | 0.04RS |
| --- | ---: | ---: |
| reject fraction | 0.0328 | 0.0929 |
| train reward mean | 0.3065 | 0.3237 |
| train acc | 0.6532 | 0.6618 |
| response length mean | 5570 | 5680 |
| seq mismatch mean | 0.03354 | 0.03340 |

结论：

- `0.04RS` 的过滤比例约为 `0.045RS` 的 2.8 倍。
- 但训练 reward、acc、length、seq mismatch 在前 55 step 差异不大。
- 说明把阈值从 0.045 降到 0.04，并没有显著改变早期训练轨迹。

这支持一个判断：早期 val 崩不是因为过滤阈值太松这么简单；更可能是策略本身向短答案/训练集适配方向漂移。

## 3. Step 级动态

### 3.1 0.045RS

| step | reject | reward | acc | length | seq mismatch |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 5 | 0.005 | 0.136 | 0.568 | 7734 | 0.0322 |
| 10 | 0.015 | 0.163 | 0.582 | 7683 | 0.0328 |
| 20 | 0.021 | 0.345 | 0.672 | 6108 | 0.0332 |
| 30 | 0.017 | 0.341 | 0.670 | 4746 | 0.0333 |
| 40 | 0.025 | 0.470 | 0.735 | 3659 | 0.0336 |
| 50 | 0.028 | 0.280 | 0.640 | 4197 | 0.0330 |
| 55 | 0.158 | 0.065 | 0.533 | 6186 | 0.0366 |

现象：

- step5-40：reward 上升，length 快速下降。
- step50：reward 开始下降。
- step55：reject fraction 和 length 同时上升，说明不稳定信号开始出现。

### 3.2 0.04RS

| step | reject | reward | acc | length | seq mismatch |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 5 | 0.053 | 0.130 | 0.565 | 7794 | 0.0323 |
| 10 | 0.075 | 0.141 | 0.570 | 7801 | 0.0328 |
| 20 | 0.067 | 0.369 | 0.685 | 6320 | 0.0333 |
| 30 | 0.096 | 0.341 | 0.670 | 5147 | 0.0338 |
| 40 | 0.107 | 0.495 | 0.748 | 3714 | 0.0336 |
| 50 | 0.083 | 0.292 | 0.646 | 4572 | 0.0334 |
| 55 | 0.109 | 0.234 | 0.617 | 4829 | 0.0332 |
| 60 | 0.154 | 0.171 | 0.585 | 6752 | 0.0350 |
| 65 | 0.327 | 0.028 | 0.514 | 8088 | 0.0389 |
| 70 | 0.561 | -0.178 | 0.411 | 9977 | 0.0458 |
| 80 | 0.403 | -0.087 | 0.457 | 7779 | 0.0408 |
| 95 | 0.366 | -0.015 | 0.493 | 6793 | 0.0412 |

现象：

- step5-40：reward 同样上升，length 同样下降。
- step60 之后：length 重新上升，seq mismatch 上升。
- step65-70：reject fraction 从 15.4% 到 56.1%，reward 转负。

解释：

- `0.04RS` 前期更强过滤没有阻止短答案化。
- 后期过滤比例暴涨更像训练已经进入不稳定阶段后的伴随信号，而不是提前保护信号。

## 4. 被过滤序列特性

### 4.1 全局 kept vs rejected

| 实验 | 集合 | 数量 | length mean | reward mean | acc | seq mismatch |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| 0.045RS | k 9224 | -0.057 | 0.472 | 0.0511 |
| 0.04RS | kept | 30786 | 5248 | 0.330 | 0.665 | 0.0326 |
| 0.04RS | rejected | 8126 | 11511 | -0.418 | 0.291 | 0.0512 |

结论：

- 两组 rejected 都更长、更低 reward、更低 acc。
- `0.04RS` 更明显：rejected length 11511，acc 只有 0.291。
- 这和 `threshold0.25 + TIS` 那组相反；在 overlap 阈值实验中，RS 主要在过滤长错样本。

### 4.2 长度桶

#### 0.045RS

| length bucket | reject fraction | acc | rejected acc |
| --- | ---: | ---: | ---: |
| 0-2048 | 0.040 | 0.850 | 0.862 |
| 2048-4096 | 0.018 | 0.732 | 0.873 |
| 4096-8192 | 0.011 | 0.643 | 0.826 |
| 8192-12288 | 0.007 | 0.534 | 1.000 |
| 12288-16384 | 0.146 | 0.204 | 0.032 |

#### 0.04RS

| length bucket | reject fraction | acc | rejected acc |
| --- | ---: | ---: | ---: |
| 0-2048 | 0.163 | 0.843 | 0.843 |
| 2048-4096 | 0.096 | 0.740 | 0.787 |
| 4096-8192 | 0.049 | 0.647 | 0.687 |
| 8192-12288 | 0.046 | 0.533 | 0.616 |
| 12288-16384 | 0.661 | 0.074 | 0.010 |

结论：

- 两组在最长 bucket `12288-16384` 都显著过滤更多。
- `0.04RS` 对长输出过滤非常强：最长 bucket 过滤 66.1%，而且这些被过滤样本几乎全错。
- 但 `0.04RS` 也明显增加了短样本过滤：0-2048 bucket 从 4.0% 提高到 16.3%。

因此，0.04 的问题是：

```text
它确实更强地过滤长错样本，
但也开始更多过滤短且相对高正确率样本。
```

## 5. Prompt group 难度 bias

### 5.1 0.045RS

| group 类型 | reject fraction | reward mean | length mean | seq mismatch |
| --- | ---: | ---: | ---: | ---: |
| hard | 0.034 | -0.923 | 7641 | 0.0325 |
| mid | 0.055 | 0.021 | 8013 | 0.0325 |
| easy | 0.029 | 0.940 | 4275 | 0.0342 |
ept | 21788 | 5446 | 0.319 | 0.659 | 0.0329 |
| 0.045RS | rejected | 740 |
group 相关性：

```text
reject vs reward: -0.022
reject vs length:  0.063
reject vs seqmis:  0.573
```

0.045RS 在 group 层面没有明显难度偏置。

### 5.2 0.04RS

| group 类型 | reject fraction | reward mean | length mean | seq mismatch |
| --- | ---: | ---: | ---: | ---: |
| hard | 0.337 | -0.930 | 9823 | 0.0391 |
| mid | 0.231 | 0.012 | 8758 | 0.0367 |
| easy | 0.121 | 0.927 | 4059 | 0.0347 |

group 相关性：

```text
reject vs reward: -0.308
reject vs length:  0.430
reject vs seqmis:  0.859
```

0.04RS 已经明显偏向过滤 hard / long group。这个方向比 0.045 更符合“过滤坏样本”的直觉，但问题是过滤比例后期过大，可能改变训练分布。

## 6. 极端 token 影响

### 6.1 极端 token 占比

| threshold | 0.045 token frac | 0.045 mass contrib | 0.04 token frac | 0.04 mass contrib |
| --- | ---: | ---: | ---: | ---: |
| token_mismatch >= 0.1 | 3.31% | 15.53% | 4.50% | 17.93% |
| >= 0.15 | 1.26% | 8.10% | 1.89% | 10.13% |
| >= 0.2 | 0.57% | 4.50% | 0.91% | 5.93% |
| >= 0.25 | 0.26% | 2.45% | 0.42% | 3.19% |
| >= 0.3 | 0.12% | 1.27% | 0.17% | 1.53% |

结论：

- 0.04 后期更不稳定，因此极端 token 比例略高。
- 但高阈值 extreme token 仍然占比很小。
- `>=0.3` token 只占 0.17%，贡献 1.53% mismatch mass。

这说明极端 token 不是主因。

### 6.2 极端 token 前后窗口

| 窗口 | 0.045 top event 后-前 | 0.045 random 后-前 | 0.04 top event 后-前 | 0.04 random 后-前 |
| --- | ---: | ---: | ---: | ---: |
| 64 | 0.00027 | 0.00035 | 0.00034 | 0.00028 |
| 256 | 0.00123 | 0.00082 | 0.00146 | 0.00084 |
| 1024 | 0.00117 | 0.00085 | 0.00179 | 0.00135 |

结论：

- top event 后窗口只比前窗口高非常小。
- 与 random 位置差距也很小。
- 没有强证据说明极端 token 导致后续 mismatch 级联。

因此，两组实验的崩溃更像整体生成分布漂移，不是少数 extreme token 触发。

## 7. Validation 长度和截断

### 7.1 0.045RS validation

| step | val length | clip ratio |
| ---: | ---: | ---: |
| 0 | 12623 | 0.416 |
| 10 | 12160 | 0.372 |
| 20 | 11154 | 0.309 |
| 30 | 9957 | 0.222 |
| 40 | 9162 | 0.134 |
| 50 | 9069 | 0.118 |

0.045RS 的 val 输出持续变短，clip ratio 持续下降。

解释：

- val 崩不是因为输出爆长或截断变严重。
- 更像是推理过程被压缩，模型学到短答案/格式化捷径，导致泛化下降。

### 7.2 0.04RS validation

| step | val length | clip ratio |
| ---: | ---: | ---: |
| 0 | 12575 | 0.417 |
| 10 | 12252 | 0.380 |
| 20 | 11378 | 0.322 |
| 30 | 10299 | 0.248 |
| 40 | 9520 | 0.184 |
| 50 | 9802 | 0.199 |
| 60 | 12357 | 0.462 |
| 70 | 14785 | 0.820 |
| 80 | 12513 | 0.574 |
| 90 | 11276 | 0.490 |
| 100 | 11088 | 0.482 |
| 130 | 11029 | 0.426 |

0.04RS 有两个阶段：

1. step0-40：val length 下降，类似 0.045RS。
2. step60-70：val length 和 clip ratio 暴涨，step70 clip ratio 到 0.82。

解释：

- 前期 val 崩仍然可能来自短答案化/推理压缩。
- 后期则出现明显长输出/截断问题，说明模型进入另一种不稳定状态。

## 8. 为什么 val 先崩，但 reward 初看没有崩

### 8.1 train reward 是训练集即时反馈

训练 reward 来自当前 train prompts，容易被以下因素提升：

- 输出更短；
- 更符合 parser；
- 更快到达 boxed/final answer；
- 适配训练集分布。

这些不一定代表验证集推理能力提升。

### 8.2 validation 更早暴露泛化问题

验证集 AIME 更依赖完整推理和泛化。模型如果学到短答案/格式化捷径：

```text
train reward 可以先涨
val acc 可以先跌
```

这就是 0.045RS 和 0.04RS 前期共同看到的现象。

### 8.3 RS 对早期漂移没有足够保护

0.045RS 前期过滤太弱，基本没有改变训练分布。

0.04RS 虽然过滤更强，但前期主要还是没有阻止 reward 上涨和 length 下降。说明早期问题不是简单靠固定阈值多过滤一些就能解决。

### 8.4 后期 reward 才崩是滞后现象

0.04RS 到 step65-70：

```text
reject fraction: 0.327 -> 0.561
train reward: 0.028 -> -0.178
length: 8088 -> 9977
val clip ratio: step70 0.820
```

此时训练侧也开始明显崩。也就是说 val 是先导信号，train reward 是滞后信号。

## 9. 对两个实验的判断

### 0.045RS

优点：

- 过滤温和，基本不严重扰动训练分布。

问题：

- 前期太弱，没阻止短答案化和 val 泛化下降。
- step55 已出现 reject/mismatch/length 突升迹象，但缺少后续 dump。

### 0.04RS

优点：

- 更强过滤，后期确实主要过滤长错样本。
- group 层面更偏向 hard/long group，方向上比 0.045 更像“过滤坏样本”。

问题：

- 后期过滤比例过大，step70 达到 56%。
- 长输出和 clip ratio 暴涨，说明训练已经进入不稳定区。
- 前期仍没有阻止短答案化，val 仍可能先崩。

## 10. 后续建议

1. 不建议继续用固定阈值作为主策略。

固定阈值在不同阶段表现不同：

```text
前期：可能太弱或只轻微改变训练。
后期：可能突然过滤过多。
```

2. 更建议使用动态比例或分阶段策略：

```text
warmup: 只记录，不过滤
early: top 5%-10%
mid: top 10%-15%
late: 允许升到 20%-25%，但设置 hard cap
```

3. 必须增加长度分桶控制。

0.04 后期主要过滤长错样本，但也会提高短样本过滤。建议：

```text
按 length bucket 内部过滤 top p%
```

而不是全 batch 统一阈值。

4. val 先崩时，优先检查输出文本而不是只看 reward。

重点看：

- val 是否推理被压缩；
- final answer 是否更早出现；
- 是否省略步骤；
- 是否格式化但逻辑不足。

5. router mismatch 应该作为诊断信号，而不是单独训练质量指标。

它能发现一部分长错样本，但不能解释所有 val collapse。
