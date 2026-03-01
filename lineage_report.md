# Experiment Lineage Report
## IDS568 Milestone 3 — MLOps Workflow Automation

**Date:** 2026-02-28  
**Dataset:** scikit-learn Diabetes Dataset (442 samples, 10 features)  
**Model Type:** RandomForestRegressor  
**Experiment Name:** `diabetes_rf_experiment`  
**Registry:** `DiabetesRFModel`

---

## 1. Experiment Run Summary

Five experiments were conducted with systematically varied hyperparameters. All runs used `random_state=42` and an 80/20 train/test split for reproducibility.

| Run Name | n_estimators | max_depth | min_samples_split | R² Score | RMSE | Model Hash (truncated) |
|---|---|---|---|---|---|---|
| grandiose-bird-892 | 50 | None | 2 | 0.4254 | 55.17 | `6d531573...` |
| auspicious-duck-731 | 100 | None | 2 | 0.4428 | 54.33 | `16a56eec...` |
| glamorous-dove-344 | 150 | 5 | 2 | **0.4551** | **53.73** | `7d208490...` |
| traveling-whale-979 | 200 | 10 | 5 | 0.4370 | 54.62 | `75a6dce3...` |
| monumental-skink-603 | 300 | 15 | 4 | 0.4311 | 54.90 | `ab6f1b02...` |

All runs are fully reproducible — given the same hyperparameters and `random_state=42`, the identical model hash will be produced.

---

## 2. Run Comparison & Analysis

### R² Score Comparison
```
grandiose-bird-892   |████████████████████░░░░░░░░░░|  0.4254
auspicious-duck-731  |█████████████████████░░░░░░░░░|  0.4428
glamorous-dove-344   |██████████████████████░░░░░░░░|  0.4551  ← BEST
traveling-whale-979  |████████████████████░░░░░░░░░░|  0.4370
monumental-skink-603 |████████████████████░░░░░░░░░░|  0.4311
```

### Key Observations

**Effect of n_estimators (no depth limit):**
Increasing from 50 → 100 trees improved R² from 0.4254 to 0.4428 (+0.017), but further increases to 200 and 300 trees without depth control showed diminishing returns and slight degradation, suggesting overfitting on training data.

**Effect of max_depth:**
Introducing `max_depth=5` with 150 trees produced the best result (R²=0.4551, RMSE=53.73). Constraining tree depth acts as regularization, preventing individual trees from memorizing noise in the training data.

**Effect of increasing max_depth (200, 300 trees):**
Deeper trees (depth=10, 15) with more estimators did not improve performance — R² dropped to 0.4370 and 0.4311 respectively. This indicates that on this relatively small dataset (442 samples), shallower trees generalize better.

**RMSE trend:**
RMSE inversely tracks R² as expected. The best model (150 trees, depth=5) achieves the lowest RMSE of 53.73, meaning predictions are off by ~54 units on the diabetes progression scale on average.

---

## 3. Production Candidate Selection

**Selected Run:** `glamorous-dove-344`  
**Hyperparameters:** `n_estimators=150`, `max_depth=5`, `min_samples_split=2`  
**R² Score:** 0.4551 | **RMSE:** 53.73  
**Model Hash:** `7d208490...`  
**MLflow Run ID:** logged in `tracking_evidence/run_150_5_2.json`

### Justification

`glamorous-dove-344` was selected as the production candidate for three reasons:

1. **Best predictive performance** — highest R² (0.4551) and lowest RMSE (53.73) across all five runs, exceeding the quality gate threshold of R² ≥ 0.40 with the most margin.

2. **Best regularization balance** — `max_depth=5` prevents overfitting while `n_estimators=150` provides enough ensemble diversity. The deeper/larger models (runs 4 and 5) showed that more complexity hurts on this dataset.

3. **Artifact integrity verified** — SHA-256 hash `7d208490...` is logged in MLflow, ensuring the registered model binary matches the training run exactly. Any tampering or corruption would produce a different hash.

### Registry Progression
```
Version 1 (glamorous-dove-344)
  None → Staging → Production
```
The model was first moved to **Staging** for validation against the quality gate (`model_validation.py`), then promoted to **Production** after passing all three gates (R², RMSE, model hash).

---

## 4. Identified Risks & Monitoring Needs

### Risk 1: Dataset Size Limitation
The sklearn Diabetes dataset has only 442 samples. The R² of ~0.455 is modest — in production, performance should be re-evaluated on larger, real-world data. Random forests typically benefit significantly from more training data.

**Mitigation:** Re-run experiments with a larger diabetes dataset (e.g., UCI or NHANES) and set a higher R² threshold (≥ 0.60) before production promotion.

### Risk 2: Feature Drift
The 10 input features (age, BMI, blood pressure, etc.) can shift over time in real patient populations due to demographic changes or measurement protocol updates.

**Monitoring:** Track input feature distributions (mean, std) per batch using tools like Evidently AI or a custom drift detector. Alert if any feature's mean shifts by more than 2 standard deviations from training distribution.

### Risk 3: Target Drift
The diabetes progression score distribution may shift seasonally or due to treatment protocol changes, degrading model performance silently.

**Monitoring:** Log prediction distribution per batch and compare against training target distribution using KL divergence. Alert if divergence exceeds 0.1.

### Risk 4: No Real-Time Serving Layer
Currently the model is registered in MLflow but not served via an API endpoint.

**Mitigation:** Deploy via `mlflow models serve` or wrap in a FastAPI endpoint:
```bash
mlflow models serve -m "models:/DiabetesRFModel/Production" --port 8888
```

### Risk 5: Single Model Architecture
Only RandomForestRegressor was explored. A GradientBoostingRegressor or XGBoost model may outperform on this dataset.

**Mitigation:** Extend the pipeline to support multi-model comparison in future milestones.

---

## 5. Lineage Traceability

Every artifact in this project can be fully traced:

| Artifact | Traceable To |
|---|---|
| `model_150_5_2.pkl` | Run `glamorous-dove-344`, hyperparams logged in MLflow |
| `DiabetesRFModel v1` | MLflow run ID stored in `run_150_5_2.json` |
| Training data | `sklearn_diabetes v1.0`, `random_state=42`, `test_size=0.2` |
| Model hash `7d208490...` | SHA-256 of `model_150_5_2.pkl`, logged as MLflow tag |
| CI validation | GitHub Actions run log, `model_validation.py` exit code 0 |

---

## 6. Conclusion

The experiment systematically explored five hyperparameter configurations and identified that moderate tree depth (`max_depth=5`) combined with 150 estimators provides the best generalization on the Diabetes dataset. The production model passes all quality gates, has verified artifact integrity via SHA-256 hashing, and is fully traceable to its training code, data version, and hyperparameters. Monitoring for feature and target drift is recommended before large-scale deployment.