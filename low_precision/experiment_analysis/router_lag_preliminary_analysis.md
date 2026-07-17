# Router Drift Lag: Preliminary Analysis

## Question

Does rollout/train router drift accumulate first and only later cause policy divergence and validation collapse?

This analysis combines existing local W&B histories with selected response-level `.pt` dumps. No model inference was rerun.

## Available evidence

The strongest observational datasets are:

- `usage-l1-baseline_20260708_225230`: steps 0-48, no router filtering, Expert Usage L1.
- `sequence-mismatch-TIS-C2-noRS_20260626_075522`: steps 0-359 after merging strict-resume histories, exact-set sequence mismatch.
- `sequence-mismatch-threshold0.25-TIS-C2_20260626_072854`: steps 0-363, exact-set sequence mismatch with filtering and TIS.
- `usage-l1-lengthbucket-top8RS-TIS-C2-16K_20260710_004104`: steps 0-125, Expert Usage L1.
- `usage-l1-madz-top8RS-cap2-TIS-C2-16K_20260711_073635`: steps 0-69, Expert Usage L1.

The baseline Expert Usage L1 run is the cleanest dataset for the collapse mechanism, but it contains only five validation evaluations. The long runs have enough validation points, but use a different router metric and/or active interventions.

## Step-level result

For each experiment, the analysis computed correlations between router drift at step `t` and policy metrics at `t+k`.

The first-difference correlation

`corr(delta router(t), delta rollout_probs_diff(t+1))`

was positive in all six analyzed experiments:

| Experiment | N | Pearson r |
|---|---:|---:|
| Usage L1 baseline | 46 | 0.450 |
| Exact mismatch, TIS, no RS | 357 | 0.299 |
| Exact mismatch, threshold 0.25 + TIS | 361 | 0.241 |
| Usage L1 global top8 RS | 39 | 0.450 |
| Usage L1 length-bucket top8 + TIS | 123 | 0.287 |
| Usage L1 MAD-z top8 cap2 + TIS | 67 | 0.402 |

This is repeatable evidence that changes in router state and next-step policy discrepancy are temporally coupled. It is not yet evidence that router drift independently causes the discrepancy.

After residualizing both variables against current probability-diff change, response-length change, reward change, and entropy change, the partial correlations became:

| Experiment | Partial r | Nominal p |
|---|---:|---:|
| Usage L1 baseline | 0.096 | 0.524 |
| Exact mismatch, TIS, no RS | 0.089 | 0.092 |
| Exact mismatch, threshold 0.25 + TIS | 0.113 | 0.032 |
| Usage L1 global top8 RS | 0.202 | 0.217 |
| Usage L1 length-bucket top8 + TIS | -0.023 | 0.803 |
| Usage L1 MAD-z top8 cap2 + TIS | -0.014 | 0.911 |

Only one result remains nominally significant, without correction for multiple testing or autocorrelation. Therefore the current data do not establish independent one-step predictive power.

## Collapse trajectory in the clean baseline

The no-filter Expert Usage L1 baseline shows the following aggregate trajectory:

| Step | Usage L1 mean | Probability diff | Entropy | Mean length | Mean validation accuracy |
|---:|---:|---:|---:|---:|---:|
| 10 | 0.006045 | 0.01255 | 0.17787 | 7611 | 0.6073 |
| 20 | 0.006343 | 0.01503 | 0.08143 | 6065 | 0.6438 |
| 30 | 0.006669 | 0.01637 | 0.07122 | 4446 | 0.6281 |
| 40 | 0.007086 | 0.01738 | 0.05981 | 2999 | 0.5365 |

Router drift rises before the large validation drop at step 40, but probability diff, entropy collapse, and length collapse are already changing at the same time. This supports a multi-stage instability interpretation, but cannot identify router drift as the first cause.

## Response-level result: opposite direction

Selected dumps from the same baseline produce this comparison between the highest 8% Usage L1 responses and the remaining responses:

| Step | Corr(score, prob diff) | Top8 prob diff | Rest prob diff | Top8 length | Rest length | Top8 accuracy | Rest accuracy |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 10 | -0.245 | 0.01063 | 0.01257 | 2375 | 8067 | 0.780 | 0.574 |
| 20 | -0.328 | 0.01251 | 0.01458 | 1794 | 6437 | 0.933 | 0.656 |
| 30 | -0.412 | 0.01212 | 0.01578 | 1490 | 4703 | 0.854 | 0.653 |
| 40 | -0.404 | 0.01314 | 0.01685 | 1523 | 3128 | 0.872 | 0.718 |

Across steps, aggregate Usage L1 and aggregate probability diff rise together. Within each step, high Usage L1 responses have lower probability diff and are much shorter and more often correct.

This is a multilevel reversal similar to Simpson's paradox. It means:

- Expert Usage L1 may describe a deteriorating global training state.
- The same raw score is not currently a valid response-level estimate of harmful policy discrepancy.
- Filtering the highest-score responses can remove short, correct samples even when the aggregate score tracks collapse.

## Validation lead-lag result

- Expert Usage L1 baseline has only five validation points, so a numerical lead-lag estimate is not reliable.
- Exact sequence mismatch with TIS and no RS does not show the expected negative relationship with future validation. Its Spearman correlations for leads of 0/10/20/30/40/50 steps are `0.088/0.216/0.390/0.514/0.614/0.595`.
- Threshold0.25 + TIS gives a weak negative association near a 30-step lead (`rho=-0.226`, `n=33`), but it is not stable across leads or experiments.

The existing data therefore do not show a robust universal lag from the current router metrics to future validation degradation.

## Current conclusion

The data support the following limited statement:

> Router drift, probability divergence, entropy reduction, and length instability co-evolve during collapse. Router changes contain temporal information about next-step probability discrepancy, but current metrics have not demonstrated independent or robust prediction of future validation collapse.

The stronger claim that router drift is a latent cause with a fixed delay remains plausible but unverified.

The most important finding is the mismatch between scales: a router metric can track global training instability while being biased or directionally wrong for response-level filtering.

## Decisive next test

Use a shared checkpoint before collapse and fork identical data-order runs into no intervention, TIS-only, router-filter-only, and TIS+router-filter branches. A causal router mechanism requires the intervention to first reduce router drift, then reduce probability divergence, and finally delay or prevent validation degradation. Existing observational dumps cannot replace this intervention test.
