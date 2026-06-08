# Experiment Log

## Purpose
Track all training experiments, hyperparameters, and results for reproducibility and analysis.

---

## Experiment 1: Initial PINN Training

**Date**: [YYYY-MM-DD]  
**Status**: [Not Started | In Progress | Completed | Failed]

### Configuration
```yaml
Model: PINNNetwork
Architecture: 6 hidden layers, 256 neurons
Activation: tanh
Domain: [-1, 1] × [-1, 1]
Reynolds: 100
Epochs: 5000
```

### Hyperparameters
| Parameter | Value |
|-----------|-------|
| Learning Rate | 0.001 |
| Batch Size | 32 |
| Optimizer | Adam |
| Scheduler | ExponentialLR |
| Seed | 42 |

### Loss Weights
| Component | Weight |
|-----------|--------|
| PDE Loss | 1.0 |
| BC Loss | 100.0 |
| Data Loss | 1.0 |

### Results
- **Final Training Loss**: [value]
- **Final Validation Loss**: [value]
- **PDE Residual**: [value]
- **BC Error**: [value]
- **Training Time**: [HH:MM:SS]
- **GPU Memory**: [GB]

### Metrics
```
MSE:     [value]
RMSE:    [value]
MAE:     [value]
R²:      [value]
```

### Observations
- [Key findings]
- [Convergence behavior]
- [Issues encountered]

### Files Generated
- Model: `checkpoints/pinn_exp1.pt`
- Logs: `logs/pinn_exp1.log`
- Figures: `figures/pinn_exp1_*.png`

### Next Steps
- [ ] Analyze residual distribution
- [ ] Compare with baseline
- [ ] Test on different Reynolds numbers

---

## Experiment 2: Surrogate Model Training

**Date**: [YYYY-MM-DD]  
**Status**: [Not Started | In Progress | Completed | Failed]

### Configuration
```yaml
Model: DeepSurrogate
Architecture: 6 hidden layers, 256 neurons
Dataset: Generated CFD data (5000 samples)
Train/Val/Test: 0.7/0.15/0.15
```

### Hyperparameters
| Parameter | Value |
|-----------|-------|
| Learning Rate | 0.001 |
| Batch Size | 128 |
| Optimizer | Adam |
| Scheduler | ReduceLROnPlateau |
| Dropout | 0.2 |
| Seed | 42 |

### Training Progress
| Epoch | Train Loss | Val Loss | LR |
|-------|-----------|---------|-----|
| 1 | [value] | [value] | 0.001 |
| 100 | [value] | [value] | [value] |
| 500 | [value] | [value] | [value] |
| 1000 | [value] | [value] | [value] |

### Results
- **Best Validation Loss**: [value]
- **Test R² Score**: [value]
- **Test RMSE**: [value]
- **Test MAE**: [value]
- **Training Time**: [HH:MM:SS]

### Cross-Validation (k=5)
| Fold | Train R² | Val R² | Test R² |
|------|----------|--------|---------|
| 1 | [value] | [value] | [value] |
| 2 | [value] | [value] | [value] |
| 3 | [value] | [value] | [value] |
| 4 | [value] | [value] | [value] |
| 5 | [value] | [value] | [value] |
| Mean | [value] | [value] | [value] |
| Std | [value] | [value] | [value] |

### Error Analysis
- **Max Absolute Error**: [value]
- **Mean Absolute Error**: [value]
- **95th Percentile Error**: [value]

### Observations
- [Key findings]
- [Model behavior]
- [Generalization ability]

### Files Generated
- Model: `checkpoints/surrogate_exp2.pt`
- Scaler: `checkpoints/surrogate_exp2_scaler.pkl`
- Logs: `logs/surrogate_exp2.log`
- Figures: `figures/surrogate_exp2_*.png`

### Comparison with Baseline
| Metric | Baseline | This Model |
|--------|----------|-----------|
| R² | 0.95 | [value] |
| RMSE | 0.05 | [value] |
| Inference Time (ms) | 2.5 | [value] |

---

## Experiment 3: [Experiment Name]

**Date**: [YYYY-MM-DD]  
**Status**: [Not Started | In Progress | Completed | Failed]

### [Complete above template for additional experiments]

---

## Summary Comparison

| Experiment | Model Type | Best Metric | Training Time | Status |
|-----------|-----------|-----------|--------------|--------|
| Exp 1 | PINN | R²: 0.98 | 4h 23m | ✓ Complete |
| Exp 2 | Surrogate | R²: 0.987 | 45m | ✓ Complete |
| Exp 3 | [Model] | [Metric] | [Time] | [Status] |

---

## Lessons Learned

### What Worked
- [Successful strategies]
- [Key insights]

### What Didn't Work
- [Failed approaches]
- [Reasons for failure]

### Recommendations for Next Experiments
- [Suggested improvements]
- [Alternative approaches to try]

---

## Computational Resources Summary

| Resource | Total Used | Cost Estimate |
|----------|-----------|---------------|
| GPU Hours | [hours] | $[estimate] |
| CPU Hours | [hours] | $[estimate] |
| Storage (GB) | [GB] | $[estimate] |

---

## Reproducibility Checklist

- [ ] All hyperparameters documented
- [ ] Random seeds recorded
- [ ] Data versions saved
- [ ] Code version (git commit) noted
- [ ] Environment (requirements.txt) verified
- [ ] Figures generated and saved
- [ ] Results replicated on fresh run

---

**Last Updated**: [Date]  
**Updated By**: [Name]  
**Repository**: [URL]  
**Zenodo Entry**: [DOI] (if published)
