# Classical Baseline Complexity Results:

Parameters are effective predictive values required by the fitted estimator. For logistic regression these are coefficients and intercepts. For PCA-RBF SVM they are PCA loadings, support vectors, dual coefficients, and intercepts. For XGBoost, decision-tree nodes are used as the conventional capacity unit. Scaler statistics and PCA centering vectors are reported separately as auxiliary state.

MACs follow the project's pure weighted-MAC convention and exclude fixed feature extraction, normalization, nonlinear functions, and comparisons. XGBoost performs tree comparisons rather than dense MACs, so F1/M-MACs is not applicable and F1/M-comparisons is supplied instead.

## Cross-Dataset Summary:

The efficiency ratios below are calculated from the cross-dataset mean F1 and cross-dataset mean complexity, matching the aggregation used for the neural-model comparison.

| Model | Mean Macro-F1 (%) | Mean Parameters | Mean MACs | F1/K-Params | F1/M-MACs | Mean Tree Comparisons | F1/M-Comparisons |
|---|---:|---:|---:|---:|---:|---:|---:|
| logistic_regression | 77.82 | 2,887 | 2,880 | 26.96 | 27,023.74 | N/A | N/A |
| rbf_svm | 82.02 | 309,526 | 309,499 | 0.26 | 265.01 | N/A | N/A |
| xgboost | 81.29 | 81,464 | N/A | 1.00 | N/A | 11,384 | 7,140.51 |

## Per-Dataset Results:

| Model | Dataset | Macro-F1 (%) | Parameters | Aux. state | MACs | F1/K-Params | F1/M-MACs | Tree comparisons | F1/M-comparisons |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| logistic_regression | ucihar | 92.70 | 1,140 | 378 | 1,134 | 81.31 | 81,744.87 | N/A | N/A |
| logistic_regression | motionsense | 89.43 | 762 | 252 | 756 | 117.36 | 118,293.52 | N/A | N/A |
| logistic_regression | wisdm | 75.12 | 384 | 126 | 378 | 195.63 | 198,738.04 | N/A | N/A |
| logistic_regression | pamap2 | 59.20 | 4,800 | 798 | 4,788 | 12.33 | 12,364.67 | N/A | N/A |
| logistic_regression | opportunity | 84.92 | 8,300 | 3,318 | 8,295 | 10.23 | 10,237.01 | N/A | N/A |
| logistic_regression | unimib | 62.16 | 576 | 126 | 567 | 107.91 | 109,624.91 | N/A | N/A |
| logistic_regression | skoda | 96.01 | 6,941 | 1,260 | 6,930 | 13.83 | 13,853.96 | N/A | N/A |
| logistic_regression | daphnet | 63.01 | 190 | 378 | 189 | 331.64 | 333,391.01 | N/A | N/A |
| rbf_svm | ucihar | 92.71 | 125,809 | 567 | 125,794 | 0.74 | 736.98 | N/A | N/A |
| rbf_svm | motionsense | 90.47 | 222,434 | 378 | 222,419 | 0.41 | 406.77 | N/A | N/A |
| rbf_svm | wisdm | 80.11 | 122,834 | 189 | 122,819 | 0.65 | 652.22 | N/A | N/A |
| rbf_svm | pamap2 | 77.23 | 501,867 | 1,197 | 501,801 | 0.15 | 153.91 | N/A | N/A |
| rbf_svm | opportunity | 88.29 | 308,649 | 4,977 | 308,639 | 0.29 | 286.05 | N/A | N/A |
| rbf_svm | unimib | 67.55 | 181,874 | 189 | 181,838 | 0.37 | 371.48 | N/A | N/A |
| rbf_svm | skoda | 96.92 | 549,643 | 1,890 | 549,588 | 0.18 | 176.35 | N/A | N/A |
| rbf_svm | daphnet | 62.90 | 463,093 | 567 | 463,092 | 0.14 | 135.82 | N/A | N/A |
| xgboost | ucihar | 92.81 | 44,330 | 0 | 0 | 2.09 | N/A | 7,944 | 11,682.05 |
| xgboost | motionsense | 93.51 | 86,877 | 0 | 0 | 1.08 | N/A | 10,559 | 8,856.56 |
| xgboost | wisdm | 84.61 | 67,526 | 0 | 0 | 1.25 | N/A | 9,543 | 8,865.63 |
| xgboost | pamap2 | 56.49 | 97,283 | 0 | 0 | 0.58 | N/A | 18,442 | 3,063.12 |
| xgboost | opportunity | 89.26 | 72,086 | 0 | 0 | 1.24 | N/A | 8,035 | 11,107.84 |
| xgboost | unimib | 65.27 | 120,453 | 0 | 0 | 0.54 | N/A | 15,943 | 4,094.16 |
| xgboost | skoda | 99.53 | 138,481 | 0 | 0 | 0.72 | N/A | 18,806 | 5,292.46 |
| xgboost | daphnet | 68.83 | 24,676 | 0 | 0 | 2.79 | N/A | 1,800 | 38,237.65 |

## Complexity:

Let `T` be window length, `C` the sensor channels, `D=21C` the engineered features, `K` the classes, `P` the retained PCA components, and `S` the support vectors.

- Feature extraction shared by all models is `O(C*T*log(T))`, dominated by per-channel FFTs.
- Logistic regression is `O(C*T*log(T) + D*K)` per window.
- PCA-RBF SVM is `O(C*T*log(T) + D*P + S*P + S*(K-1))` per window.
- XGBoost is `O(C*T*log(T) + N_trees*depth)` per window and is comparison-dominated.
