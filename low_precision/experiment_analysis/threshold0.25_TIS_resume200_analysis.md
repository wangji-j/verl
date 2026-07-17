# threshold0.25 + TIS-C2 resume200 router dump 重新分析

分析对象：

```text
/inspire/hdd/project/qianghuaxuexi/hujiarui-25046/ckpts/jly-DAPO-DEEPSCALER-FP8-ROLLOUT/GRPO-DEEPSCALER-Qwen3-30B-A3B-base-MEGATRON-VLLM-FP8-sequence-mismatch-threshold0.25-TIS-C2_20260626_072854/router_analysis_dump_resume200
```

分析输出 JSON：

```text
/inspire/hdd/project/qianghuaxuexi/hujiarui-25046/verl-sequence/low_precision/experiment_analysis/threshold0.25_TIS_resume200_dump_analysis.json
```

本次重算覆盖 step 201-364，共 164 个 dump 文件，335872 条 response。过滤判定按实验配置重算：

```text
rejected = seq_mismatch > 0.25
```

## 1. 总体结论

全量重算后，`threshold0.25 + TIS-C2` 仍然存在明显 bias，但 bias 在后期比早期有所减弱。

最重要的结论是：

- 全量 step201-364 中，rejected response 仍然更短、更容易答对、reward 更高。
- 后期 step301-364 中，这个偏置没有消失：rejected 仍比 kept 短约 2786 tokens，accuracy 高约 13.59 个百分点。
- 最后 20 个 dump step345-364 中，偏置进一步减弱，但仍然不是“优先过滤错误/长/难 response”。
- prompt group 层面也一样：后期 easy group 的平均过滤率仍高于 hard group。

所以，这组固定阈值 `0.25` 的 sequence RS 不宜解释成“筛掉低质量样本”。它更像是在筛掉一部分短、正确、router seq_mismatch 偏高的样本。

## 2. 分阶段过滤概况

| 窗口 | 样本数 | 过滤比例 | 平均长度 | reward mean | acc | seq_mismatch mean | rejected 平均长度 | rejected acc | rejected reward |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| early 201-244 | 90112 | 7.03% | 6983 | 0.4472 | 72.36% | 0.2128 | 4177 | 85.86% | 0.7172 |
| middle 245-300 | 114688 | 8.95% | 6783 | 0.4517 | 72.58% | 0.2165 | 4257 | 84.55% | 0.6910 |
| late 301-364 | 131072 | 7.70% | 6982 | 0.4647 | 73.23% | 0.2147 | 4411 | 85.78% | 0.7155 |
| last20 345-364 | 40960 | 6.98% | 7177 | 0.4769 | 73.85% | 0.2133 | 4371 | 86.07% | 0.7215 |
| all 201-364 | 335872 | 7.95% | 6915 | 0.4555 | 72.78% | 0.2148 | 4296 | 85.32% | 0.7065 |

观察：

1. 过滤比例从早期约 7.03% 升到后期约 7.70%，说明后期 threshold=0.25 确实过滤更多。
2. response 平均长度从早期 6983 降到后期 6982，后期整体生成更短。
3. 后期 rejected 的 reward/acc 仍高于 kept，不是低 reward 样本优先被过滤。

## 3. 全量 length / correctness bias

全量 kept vs rejected：

| 集合 | 数量 | 平均长度 | reward mean | accuracy | seq_mismatch mean |
| --- | ---: | ---: | ---: | ---: | ---: |
| kept | 309183 | 7141 | 0.4339 | 71.69% | 0.2105 |
| rejected | 26689 | 4296 | 0.7065 | 85.32% | 0.2644 |

全量相关系数：

| 相关项 | Pearson corr |
| --- | ---: |
| seq_mismatch vs length | -0.4466 |
| seq_mismatch vs reward | 0.2200 |
| reject vs length | -0.2146 |
| reject vs reward | 0.0828 |

结论：

- `seq_mismatch` 与长度显著负相关，response 越短 mismatch 越高。
- `seq_mismatch` 与 reward 正相关，正确/高 reward 的 response mismatch 反而更高。
- 因为固定阈值直接按 `seq_mismatch > 0.25` 过滤，所以最终 reject 也表现为偏短、偏正确。

## 4. 后期 bias 是否仍存在

后期 step301-364 kept vs rejected：

| 集合 | 数量 | 平均长度 | reward mean | accuracy | seq_mismatch mean |
| --- | ---: | ---: | ---: | ---: | ---: |
| kept | 120984 | 7196 | 0.4438 | 72.19% | 0.2105 |
| rejected | 10088 | 4411 | 0.7155 | 85.78% | 0.2645 |

后期相关系数：

| 相关项 | Pearson corr |
| --- | ---: |
| seq_mismatch vs length | -0.4385 |
| seq_mismatch vs reward | 0.2235 |
| reject vs length | -0.2094 |
| reject vs reward | 0.0818 |

后期结论：

- bias 减弱但没有反转。
- rejected 平均长度仍明显短于 kept。
- rejected accuracy/reward 仍高于 kept。
- 因此，后期过滤更多并不意味着它开始过滤“更差样本”；它还是保留了长度和对错偏置。

最后 20 个 dump step345-364：

| 集合 | 数量 | 平均长度 | reward mean | accuracy | seq_mismatch mean |
| --- | ---: | ---: | ---: | ---: | ---: |
| kept | 38102 | 7387 | 0.4586 | 72.93% | 0.2095 |
| rejected | 2858 | 4371 | 0.7215 | 86.07% | 0.2642 |

最后 20 step 的结论更接近“过滤趋于中性”，但 rejected 仍然略短、略正确。这说明训练后期分布变化后，固定阈值的偏置会变弱，但不能完全消除。

## 5. 按长度桶分析

### 5.1 全量 step201-364

| 长度桶 | 样本数 | 过滤比例 | acc | reward mean | seq_mismatch mean | rejected acc | rejected 平均长度 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0-2048 | 15304 | 22.31% | 92.49% | 0.8498 | 0.2355 | 92.80% | 1666 |
| 2048-4096 | 65161 | 18.00% | 85.42% | 0.7084 | 0.2295 | 88.28% | 3029 |
| 4096-8192 | 151351 | 6.45% | 77.56% | 0.5513 | 0.2152 | 82.25% | 5595 |
| 8192-12288 | 72394 | 2.01% | 64.11% | 0.2823 | 0.2038 | 72.92% | 9654 |
| 12288-16384 | 31662 | 1.05% | 34.16% | -0.3168 | 0.1976 | 48.34% | 14461 |
| >=16384 | 0 | NA | NA | NA | NA | NA | NA |

全量看，短回答过滤率最高，长回答过滤率最低；短回答同时正确率更高。

### 5.2 后期 step301-364

| 长度桶 | 样本数 | 过滤比例 | acc | reward mean | seq_mismatch mean | rejected acc | rejected 平均长度 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0-2048 | 5482 | 19.81% | 93.03% | 0.8606 | 0.2339 | 92.82% | 1677 |
| 2048-4096 | 24521 | 17.91% | 86.50% | 0.7299 | 0.2294 | 88.64% | 3041 |
| 4096-8192 | 59344 | 6.58% | 78.40% | 0.5681 | 0.2155 | 83.30% | 5609 |
| 8192-12288 | 29389 | 1.98% | 64.21% | 0.2843 | 0.2042 | 75.56% | 9664 |
| 12288-16384 | 12336 | 1.02% | 34.70% | -0.3061 | 0.1979 | 49.21% | 14350 |
| >=16384 | 0 | NA | NA | NA | NA | NA | NA |

后期仍然存在长度偏置：短桶过滤比例高于长桶。只是相比早期，长桶过滤率也抬高了，所以全局偏置变弱。

### 5.3 最后 20 step345-364

| 长度桶 | 样本数 | 过滤比例 | acc | reward mean | seq_mismatch mean | rejected acc | rejected 平均长度 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0-2048 | 1645 | 19.21% | 93.68% | 0.8736 | 0.2336 | 94.30% | 1693 |
| 2048-4096 | 7223 | 17.89% | 87.71% | 0.7541 | 0.2297 | 89.32% | 3030 |
| 4096-8192 | 18191 | 5.66% | 80.06% | 0.6012 | 0.2146 | 81.94% | 5584 |
| 8192-12288 | 9673 | 1.85% | 65.82% | 0.3164 | 0.2029 | 78.21% | 9586 |
| 12288-16384 | 4228 | 0.97% | 34.06% | -0.3188 | 0.1962 | 58.54% | 14060 |
| >=16384 | 0 | NA | NA | NA | NA | NA | NA |

最后 20 step 中，中短长度桶的过滤比例仍偏高；长输出桶虽然过滤率上升，但 rejected acc 不高，说明这里已经混入了一部分长错样本。即使如此，全局 rejected 仍没有比 kept 更差。

## 6. Prompt group 难易 bias

### 6.1 全量 group

group 相关系数：

| 相关项 | Pearson corr |
| --- | ---: |
| group reject fraction vs group reward mean | 0.1060 |
| group reject fraction vs group length mean | -0.2571 |
| group reject fraction vs group seq_mismatch mean | 0.6820 |

| group 类型 | group 数 | reject fraction mean | reward mean | acc mean | length mean | seq_mismatch mean |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| hard reward <= -0.5 | 10199 | 4.40% | -0.9460 | 2.70% | 8658 | 0.2066 |
| mid -0.5~0.5 | 1961 | 3.17% | 0.0133 | 50.66% | 10295 | 0.2023 |
| easy reward >= 0.5 | 29824 | 9.47% | 0.9639 | 98.20% | 6096 | 0.2184 |

### 6.2 后期 group step301-364

| 相关项 | Pearson corr |
| --- | ---: |
| group reject fraction vs group reward mean | 0.1039 |
| group reject fraction vs group length mean | -0.2500 |
| group reject fraction vs group seq_mismatch mean | 0.6801 |

| group 类型 | group 数 | reject fraction mean | reward mean | acc mean | length mean | seq_mismatch mean |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| hard reward <= -0.5 | 3908 | 4.21% | -0.9456 | 2.72% | 8734 | 0.2063 |
| mid -0.5~0.5 | 749 | 3.22% | 0.0160 | 50.80% | 10517 | 0.2017 |
| easy reward >= 0.5 | 11727 | 9.14% | 0.9633 | 98.17% | 6173 | 0.2183 |

结论：

- 全量和后期都不是 hard group 过滤更多。
- 后期 easy group 的过滤率仍高于 hard group。
- group reject fraction 与 group length mean 仍为负相关，说明更短的 prompt group 更容易被过滤。

## 7. 为什么会出现这种 bias

当前 `seq_mismatch` 是把 response 内有效 token 的 token mismatch 做平均：

```text
seq_mismatch = mean_over_response_tokens(token_mismatch)
```

这带来两个问题：

1. 短 response 更容易被局部高 mismatch token 拉高均值。
2. 在这组数学任务里，短 response 往往更容易答对，长 response 往往更容易是冗长推理、卡住或错误。

所以固定阈值 `0.25` 实际上把“短且正确”与“高 seq_mismatch”耦合起来了。它过滤的是一个统计分布特征，而不是可靠的训练风险。

## 8. 建议

如果继续基于 sequence mismatch 做 RS，不建议继续使用全局固定阈值。更合理的做法是：

1. 按 response length bucket 做阈值归一化或分桶 top-p 过滤。
2. 至少记录每个长度桶内的 reject fraction / rejected reward / rejected length。
3. 如果要引入 position 信息，也应先做 length-bucket + position-bucket normalization，避免 tail mismatch 和短答案 bias 叠加。
4. 不要只看全局过滤比例 8%-15%，还要看过滤对象是否偏正确、偏短、偏 easy group。

当前这组数据的直接结论是：`threshold0.25 + TIS-C2` 的后期过滤仍有长度 bias 和对错 bias，只是比早期弱；它没有稳定地筛掉更错或更难的 response。
