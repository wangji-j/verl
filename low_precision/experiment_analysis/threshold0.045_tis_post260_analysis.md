# Threshold 0.045 + TIS: Post-260 Validation Analysis

## Data availability

Experiment:

`GRPO-DEEPSCALER-Qwen3-30B-A3B-base-MEGATRON-VLLM-FP8-overlap-mismatch-threshold0.045RS-TIS-C2-16K_20260702_201754`

The response/token dump for this experiment was previously deleted. This analysis uses the complete local W&B history for steps 0-353. It can establish step-level timing and associations, but cannot recover which individual responses or tokens caused the late decline.

The actual run had token-level TIS with C=2, overlap mismatch metrics, and fixed threshold 0.045 filtering.

## Validation trajectory

Validation does not suddenly begin falling at step 260:

- AIME25 reaches its local high near step 180: 60.63%.
- AIME24 reaches its high near step 200: 75.42%.
- The mean validation score is 67.29% at steps 180 and 200.
- The decline becomes continuous after step 260 and bottoms at 62.92% at steps 320-330.
- It partially recovers to 63.91% at step 350.

From step 260 to 350:

- AIME24 slope: -0.113 percentage points per 10 steps; nominal p=0.258.
- AIME25 slope: -0.531 percentage points per 10 steps; nominal p=0.0008.
- Mean validation slope: -0.322 percentage points per 10 steps; nominal p=0.0044.

The late decline is therefore primarily driven by AIME25, not a uniform collapse across both validation sets. This is more consistent with degradation on the harder/generalization-sensitive distribution than with a complete numerical training collapse.

## Phase comparison

The table reports medians over all training steps in each interval.

| Metric | 180-249 | 250-279 | 280-319 | 320-353 |
|---|---:|---:|---:|---:|
| Sequence mismatch mean | 0.030701 | 0.030929 | 0.031027 | 0.030688 |
| RS rejected fraction | 0.00439 | 0.00439 | 0.00586 | 0.00342 |
| Rollout probability diff mean | 0.010250 | 0.010574 | 0.010619 | 0.010590 |
| Rollout/actor probability correlation | 0.997482 | 0.996982 | 0.997146 | 0.997336 |
| Rollout/train KL | 0.002811 | 0.002938 | 0.002921 | 0.002871 |
| TIS ratio standard deviation | 0.073980 | 0.075697 | 0.075476 | 0.074727 |
| TIS effective sample size | 0.994556 | 0.994302 | 0.994335 | 0.994446 |
| Actor entropy | 0.469784 | 0.440255 | 0.450872 | 0.474102 |
| Actor KL loss to reference | 0.088419 | 0.080689 | 0.085773 | 0.096464 |
| Actor gradient norm | 0.013086 | 0.012740 | 0.012583 | 0.011846 |
| Train reward mean | 0.450685 | 0.445722 | 0.467683 | 0.461690 |
| Response length mean | 6739.8 | 6496.8 | 6614.2 | 7229.6 |

## What changes before the sustained decline

Comparing steps 180-249 with 250-279:

- Probability diff rises about 3.2%.
- Rollout/train KL rises about 4.5%.
- TIS ratio standard deviation rises about 2.3%.
- Rollout/actor probability correlation decreases.
- Entropy falls about 6.3%.
- Mean response length falls about 3.6%.
- Router sequence mismatch rises only about 0.7%.

The probability-level mismatch and entropy change are much clearer than the router mismatch change. Router overlap mismatch alone is therefore not a sufficient explanation for the later validation decline.

## What happens while validation falls

During steps 280-319:

- Probability diff, KL, and TIS variance remain at their elevated level.
- Mean validation falls from 64.74% at step 280 to 62.92% at step 320.
- Train reward increases relative to the 180-249 window.
- Gradient norm and PPO clipping do not explode.
- Fixed-threshold RS rejects only about 0.6% of responses at the median.

The simultaneous increase in training reward and decrease in validation is a train/validation decoupling signal. It supports policy over-optimization or generalization degradation rather than reward-training failure.

The rejection rate is too small for this run to be considered a strong filtering intervention. At threshold 0.045, the experiment behaves much closer to TIS-only training than to an 8-10% RS experiment.

## Why validation remains low after mismatch partly recovers

After step 320:

- Sequence mismatch, probability diff, rollout/train KL, and TIS variance partially recover.
- Entropy also recovers.
- Actor KL loss to the reference rises to its highest window median.
- Mean response length rises to about 7230 tokens.
- Validation remains near its low point before a small recovery.

This separation supports a lagged or hysteretic process:

1. Probability-level train/rollout divergence and lower entropy appear before the sustained validation decline.
2. Validation falls while these discrepancies remain elevated.
3. The immediate rollout/train discrepancy later improves, but the actor has already moved farther from the reference policy and its response behavior has shifted.
4. Validation therefore does not recover immediately with the mismatch metrics.

This is compatible with cumulative policy over-optimization, but it is observational evidence rather than causal proof.

## Indicators associated with lower validation

Using 10-step metric averages at validation checkpoints from step 180 onward, nominal Spearman correlations with validation are:

| Metric | Spearman rho with validation |
|---|---:|
| Rollout probability diff mean | -0.748 |
| Rollout/train KL | -0.572 |
| TIS ratio standard deviation | -0.547 |
| Router sequence mismatch mean | -0.492 |
| Response length mean | -0.448 |
| Rollout/actor probability correlation | +0.471 |
| Actor gradient norm | +0.501 |

These correlations contain common time trends and serial dependence, so their nominal p-values should not be interpreted as independent significance tests. Their useful role is ranking candidate indicators.

Probability diff, rollout/train KL, TIS ratio variance, and probability correlation are more informative for this decline than raw router mismatch or rejection fraction.

## Factors not supported by the run

- No gradient explosion: gradient norm decreases late.
- No large PPO clipping event: `pg_clipfrac` remains very small.
- No continuously worsening router mismatch after 260.
- No large RS intervention: rejection remains around 0.3-0.6% for most late windows.
- No universal benchmark collapse: AIME25 drives most of the decline.

## Current interpretation

The most consistent explanation is a two-stage process:

1. Around steps 250-279, probability-level train/rollout divergence increases and entropy decreases. TIS limits extreme ratios but does not prevent the policy from continuing to optimize.
2. From step 280 onward, training reward remains healthy or increases while AIME25 validation degrades. Later, immediate rollout/train mismatch partly recovers, but actor-reference divergence and response behavior remain shifted.

Therefore the post-260 decline is better described as delayed generalization degradation or policy over-optimization under persistent low-precision mismatch, not as an instantaneous router-mismatch explosion.

The deleted response-level dump prevents testing whether the late phase preferentially filtered correct/short/easy responses. That selection-bias question cannot be answered from this run's remaining W&B metrics.
