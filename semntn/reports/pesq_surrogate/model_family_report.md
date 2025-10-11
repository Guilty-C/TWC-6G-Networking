# PESQ Surrogate Model Comparison

- Configuration: `semntn\configs\pesq_surrogate.yaml`
- Best model family: **rational_physics**

## Quantitative summary

| family | type | num_basis | iterations | train_rmse | train_mae | train_r2 | test_rmse | test_mae | test_r2 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| rational_physics | rational | 15 | 2000 | 0.1277 | 0.1065 | 0.9115 | 0.1278 | 0.1021 | 0.9144 |
| linear_physics | linear | 15 | 2000 | 0.1339 | 0.1127 | 0.9027 | 0.1366 | 0.1124 | 0.9023 |
| logistic_physics | logistic | 15 | 2000 | 0.1629 | 0.1346 | 0.8561 | 0.1638 | 0.1319 | 0.8594 |

## Verification curves

![rational_physics_snr_db](rational_physics_snr_db.png)
![rational_physics_packet_loss](rational_physics_packet_loss.png)
![rational_physics_delay_ms](rational_physics_delay_ms.png)
![rational_physics_mos_baseline](rational_physics_mos_baseline.png)
![rational_physics_semantic_factor](rational_physics_semantic_factor.png)
