# Expert Usage L1 Length Calibration Feasibility

## Data

- Experiment: `usage-l1-baseline_20260708_225230`
- Mode: Expert Usage L1 metrics only, no router RS filtering
- Dumps: 50 consecutive steps (`step 1` through `step 50`)
- Responses: 102,400 (`2,048` per step)
- Fields used: `seq_mismatch`, `seq_valid_token_count`, `seq_reward`
- Correct response definition: `seq_reward > 0`

The no-RS baseline is used because its response distribution has not already been modified by router filtering.
Calibration is fitted independently inside every step, so temporal drift in the overall L1 scale is not mistaken for a length effect.

## Methods

Raw score:

```text
raw_score(i) = seq_mismatch(i)
```

Continuous local median/MAD calibration:

```text
z_score(i) = (raw_score(i) - local_median(length_i))
             / (local_MAD(length_i) + epsilon)
```

The local curves use 65 interpolated knots and a 256-response nearest-length window.

Conditional-percentile calibration:

```text
percentile_score(i)
    = empirical CDF of raw_score(i)
      among the 256 responses with the nearest lengths
```

Responses at the 16,384-token maximum are treated as a separate right-censored group.

## Score Correlations

| Score | Spearman with length | Pearson with length | Spearman with reward |
|---|---:|---:|---:|
| Raw Expert Usage L1 | -0.8631 | -0.6038 | +0.2803 |
| Local median/MAD Z | -0.0126 | +0.0891 | +0.0560 |
| Conditional percentile | -0.0212 | -0.0195 | +0.0563 |

The raw score has a very strong negative relationship with response length. This is not a small secondary effect: length largely determines the raw ranking. Both calibrations remove almost all monotonic score-length correlation.

## Top-5% Filtering

Values are averages over 50 steps.

| Ranking score | Rejected/kept mean-length ratio | Rejected minus kept correct rate | Length-decile rejection range |
|---|---:|---:|---:|
| Raw L1 | 0.491 | +15.92 pp | 0.1% to 26.4% |
| Local median/MAD Z | 1.210 | +0.81 pp | 3.6% to 7.3% |
| Conditional percentile | 1.049 | +4.39 pp | 4.7% to 5.4% |

For raw L1, the shortest length decile has a 26.4% rejection rate even though the global target is 5%. Deciles 7-9 are almost never rejected. The raw filter therefore acts primarily as a short-response selector.

Conditional percentile produces the most uniform rejection rate across lengths. Its rejected/kept mean-length ratio is close to one, and its per-step median ratio is 0.996.

The 10th/50th/90th percentiles of the per-step rejected/kept mean-length ratio are:

| Ranking score | P10 | P50 | P90 |
|---|---:|---:|---:|
| Raw L1 | 0.260 | 0.310 | 1.160 |
| Local median/MAD Z | 0.923 | 1.145 | 1.746 |
| Conditional percentile | 0.945 | 0.996 | 1.343 |

MAD-Z corrects the center and scale but not the shape of the conditional score distribution. Different lengths have different upper-tail shapes, so Top-5% selection retains a reverse long-response bias. Conditional percentile directly calibrates the tail probability and is more suitable for percentile filtering.

## Top-3% and Top-8%

| Fraction | Score | Rejected/kept length ratio | Correct-rate gap |
|---|---|---:|---:|
| 3% | Raw L1 | 0.628 | +13.14 pp |
| 3% | Conditional percentile | 1.058 | +4.79 pp |
| 8% | Raw L1 | 0.416 | +17.23 pp |
| 8% | Conditional percentile | 1.029 | +3.68 pp |

The raw bias becomes stronger at higher rejection fractions. Conditional-percentile calibration remains close to length-neutral for all three tested fractions.

## Interpretation of the Remaining Correctness Gap

After conditioning on length, the score-reward Spearman correlation falls from `0.2803` to about `0.056`. The remaining Top-5% correct-rate gap is `+4.39 pp`.

This residual must not automatically be labeled a bias. It means that among responses of similar length, correct responses still have slightly higher router-distribution drift. Removing it would require conditioning on reward, which would explicitly alter the learning target and could hide a genuine relationship between correctness and routing sensitivity.

Therefore:

- Length should be treated as a nuisance variable and calibrated out.
- Reward should remain an audit variable, not an input to the first calibration implementation.
- Correctness bias should be monitored after calibration rather than forcibly normalized away.

## Feasibility Conclusion

Continuous length calibration is strongly supported by the existing dumps. However, local median/MAD Z-score is not the best online ranking score because Top-K filtering depends on conditional tails, not only conditional mean and variance.

Recommended first implementation:

```text
1. Compute raw Expert Usage L1 for all 2,048 responses.
2. Estimate each response's score percentile among nearby response lengths.
3. Treat max-length truncated responses as a separate censored group.
4. Globally reject the highest 3% or 5% conditional percentiles.
5. Keep TIS-C2 as the token-level correction.
6. Log rejected/kept length, correctness, max-length rate, and per-decile rejection rate.
```

For the stated goal of correction rather than distribution enforcement, start with Top-3%; use Top-5% as the primary stronger ablation.

## Limitation

The dumps save per-response Expert Usage L1 but not per-response expert histograms/counts. Consequently, they support empirical conditional calibration but cannot reconstruct the proposed multinomial-null bootstrap exactly.

To evaluate a theoretical null-standardized score later, future dumps need per-response, per-layer expert counts (or compact normalized histograms). Full token-level routed-expert tensors are not required.

## Artifacts

- Analysis script: `analyze_usage_l1_length_calibration.py`
- Full result: `usage_l1_length_calibration_final.json`
