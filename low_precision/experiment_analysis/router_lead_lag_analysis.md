# Router Drift Lead-Lag Analysis

This is an observational analysis of local W&B histories associated with existing router dumps.
Raw level correlations can be trend-driven; first-difference results are reported separately.

| Experiment | Steps | Router metric | Val points | Best lead to prob diff | Best lead to validation |
|---|---:|---|---:|---|---|
| usage_l1_baseline | 0-48 | `router/rollout_vs_train/expert_usage_l1_mean` | 5 | k=1, rho=0.975, n=47 | n/a |
| sequence_mismatch_tis_no_rs | 0-359 | `router/rollout_vs_train/seq_mismatch_mean` | 36 | k=1, rho=0.530, n=358 | k=0, rho=0.088, n=35 |
| sequence_mismatch_threshold025_tis | 0-363 | `router/rollout_vs_train/seq_mismatch_mean` | 37 | k=1, rho=0.269, n=362 | k=30, rho=-0.226, n=33 |
| usage_l1_top8_rs | 0-41 | `router/rollout_vs_train/expert_usage_l1_mean` | 5 | k=1, rho=0.903, n=40 | n/a |
| usage_l1_lengthbucket_top8_tis | 0-125 | `router/rollout_vs_train/expert_usage_l1_mean` | 13 | k=30, rho=0.030, n=95 | k=0, rho=0.042, n=12 |
| usage_l1_madz_top8_cap2_tis | 0-69 | `router/rollout_vs_train/expert_usage_l1_mean` | 7 | k=30, rho=0.404, n=39 | n/a |

## Per-experiment details

### usage_l1_baseline

- `training/rollout_probs_diff_mean`: level k=1, rho=0.975; first-difference k=1, rho=0.421
- `training/rollout_actor_probs_pearson_corr`: level k=1, rho=-0.979; first-difference k=1, rho=-0.163
- `actor/pg_clipfrac`: level k=1, rho=0.911; first-difference k=20, rho=0.306
- `actor/grad_norm`: level k=0, rho=0.970; first-difference k=5, rho=0.339
- `actor/entropy`: level k=0, rho=-0.956; first-difference k=0, rho=-0.100
- `critic/rewards/mean`: level k=0, rho=0.514; first-difference k=5, rho=0.254
- `response_length/mean`: level k=30, rho=-0.505; first-difference k=30, rho=0.554

### sequence_mismatch_tis_no_rs

- `training/rollout_probs_diff_mean`: level k=1, rho=0.530; first-difference k=1, rho=0.281
- `training/rollout_actor_probs_pearson_corr`: level k=0, rho=-0.521; first-difference k=30, rho=-0.163
- `actor/pg_clipfrac`: level k=5, rho=0.464; first-difference k=5, rho=0.093
- `actor/grad_norm`: level k=30, rho=0.187; first-difference k=5, rho=0.065
- `actor/entropy`: level k=0, rho=-0.712; first-difference k=0, rho=-0.592
- `critic/rewards/mean`: level k=0, rho=-0.039; first-difference k=0, rho=0.283
- `response_length/mean`: level k=1, rho=0.187; first-difference k=1, rho=0.315
- validation lead-lag: k=0: rho=0.088 (n=35), k=10: rho=0.216 (n=34), k=20: rho=0.390 (n=33), k=30: rho=0.514 (n=32), k=40: rho=0.614 (n=31), k=50: rho=0.595 (n=30)

### sequence_mismatch_threshold025_tis

- `training/rollout_probs_diff_mean`: level k=1, rho=0.269; first-difference k=1, rho=0.232
- `training/rollout_actor_probs_pearson_corr`: level k=0, rho=-0.429; first-difference k=0, rho=-0.118
- `actor/pg_clipfrac`: level k=1, rho=0.446; first-difference k=1, rho=0.111
- `actor/grad_norm`: level k=30, rho=0.237; first-difference k=1, rho=0.061
- `actor/entropy`: level k=0, rho=-0.815; first-difference k=0, rho=-0.587
- `critic/rewards/mean`: level k=20, rho=-0.077; first-difference k=0, rho=0.224
- `response_length/mean`: level k=1, rho=-0.060; first-difference k=1, rho=0.351
- validation lead-lag: k=0: rho=0.152 (n=36), k=10: rho=0.065 (n=35), k=20: rho=-0.095 (n=34), k=30: rho=-0.226 (n=33), k=40: rho=-0.139 (n=32), k=50: rho=-0.025 (n=31)

### usage_l1_top8_rs

- `training/rollout_probs_diff_mean`: level k=1, rho=0.903; first-difference k=1, rho=0.482
- `training/rollout_actor_probs_pearson_corr`: level k=2, rho=-0.961; first-difference k=5, rho=-0.359
- `actor/pg_clipfrac`: level k=1, rho=0.823; first-difference k=30, rho=0.358
- `actor/grad_norm`: level k=0, rho=0.963; first-difference k=30, rho=0.527
- `actor/entropy`: level k=2, rho=-0.927; first-difference k=20, rho=-0.224
- `critic/rewards/mean`: level k=0, rho=0.728; first-difference k=5, rho=0.370
- `response_length/mean`: level k=30, rho=-0.427; first-difference k=30, rho=0.248

### usage_l1_lengthbucket_top8_tis

- `training/rollout_probs_diff_mean`: level k=30, rho=0.030; first-difference k=1, rho=0.282
- `training/rollout_actor_probs_pearson_corr`: level k=1, rho=0.029; first-difference k=1, rho=-0.110
- `actor/pg_clipfrac`: level k=1, rho=0.074; first-difference k=1, rho=0.168
- `actor/grad_norm`: level k=10, rho=0.189; first-difference k=10, rho=0.076
- `actor/entropy`: level k=0, rho=-0.270; first-difference k=0, rho=-0.492
- `critic/rewards/mean`: level k=0, rho=0.237; first-difference k=0, rho=0.367
- `response_length/mean`: level k=30, rho=0.045; first-difference k=1, rho=0.476
- validation lead-lag: k=0: rho=0.042 (n=12), k=10: rho=0.345 (n=11), k=20: rho=0.515 (n=10), k=30: rho=0.183 (n=9), k=40: rho=0.833 (n=8), k=50: rho=0.107 (n=7)

### usage_l1_madz_top8_cap2_tis

- `training/rollout_probs_diff_mean`: level k=30, rho=0.404; first-difference k=1, rho=0.465
- `training/rollout_actor_probs_pearson_corr`: level k=30, rho=-0.447; first-difference k=5, rho=-0.220
- `actor/pg_clipfrac`: level k=20, rho=0.393; first-difference k=10, rho=0.258
- `actor/grad_norm`: level k=1, rho=0.380; first-difference k=10, rho=0.235
- `actor/entropy`: level k=30, rho=-0.573; first-difference k=0, rho=-0.476
- `critic/rewards/mean`: level k=0, rho=0.078; first-difference k=5, rho=0.252
- `response_length/mean`: level k=30, rho=0.497; first-difference k=1, rho=0.489
- validation lead-lag: k=0: rho=0.200 (n=6)
