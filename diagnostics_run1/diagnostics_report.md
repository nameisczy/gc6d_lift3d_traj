# Diagnostics Report

## Failure Mode Breakdown
- episodes: 200
- final_center_err <= 2cm: 16
- 2cm < final_center_err <= 5cm: 28
- final_center_err > 5cm: 156

## Key Metrics
- action MAE (dx,dy,dz,dgrip): {'dx': 0.004140452016144991, 'dy': 0.0029615736566483974, 'dz': 0.014294063672423363, 'dgrip': 0.07124993950128555}
- top1 nearest-GT center distance stats: {'mean': 0.062044985996326435, 'std': 0.04268339744861742, 'min': 0.0028785427566617727, 'max': 0.2279299795627594}
- top1 nearest-GT rotation trace stats: {'mean': 1.9438678431510925, 'std': 0.9715850349217324, 'min': -0.9986236691474915, 'max': 2.9531238079071045}
- GC6D feature norm: {'mean': 25.55675075531006, 'std': 2.2674388551123883e-05, 'min': 25.556625366210938, 'max': 25.55681037902832}

## Artifacts
- `diagnostics_summary.json`
- `action_diagnostics.png`
- `rollout_curves_best.png`, `rollout_curves_worst.png`, `rollout_curves_random.png`
- `grasp_case_best.png`, `grasp_case_worst.png`, `grasp_case_random.png`
- `pca_metaworld_vs_gc6d.png` (if `--metaworld-npz` provided)
- `pca_low_vs_high_error_gc6d.png`