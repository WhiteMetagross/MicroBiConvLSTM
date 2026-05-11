# Pico 2 FP32 and INT8 Deployment Matrix

Generated: 2026-05-11T02:27:48.020973+00:00

## Coverage:

| Model | INT8 Runs | INT8 Fails | INT8 Missing | FP32 Runs | FP32 Fails | FP32 Missing |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| microbi | 8 | 0 | 0 | 8 | 0 | 0 |
| tinyhar | 7 | 1 | 0 | 5 | 3 | 0 |
| tinierhar | 8 | 0 | 0 | 7 | 1 | 0 |
| deepconvlstm | 1 | 7 | 0 | 1 | 7 | 0 |

## microbi:

| Dataset | Variant | Success | Latency (ms) | PyTorch Parity (%) | Arena Used (B) | Flash (B) | Top-1 Match | Failure | Source |
| --- | --- | --- | ---: | ---: | ---: | ---: | --- | --- | --- |
| ucihar | int8 | True | 78.914 | 89.874222 | 118340 | 669436 | True | - | microbi_pico2_metrics.json |
| ucihar | fp32 | True | 64.8 | 99.999992 | 130016 | 604628 | True | - | result.json |
| motionsense | int8 | True | 79.033 | 75.346001 | 117988 | 666740 | False | - | microbi_pico2_metrics.json |
| motionsense | fp32 | True | 62.331 | 99.999992 | 130016 | 602092 | False | - | result.json |
| wisdm | int8 | True | 76.634 | 91.356194 | 117988 | 664988 | True | - | microbi_pico2_metrics.json |
| wisdm | fp32 | True | 60.475 | 99.999992 | 130016 | 599620 | True | - | result.json |
| pamap2 | int8 | True | 76.969 | 89.002266 | 118724 | 675060 | False | - | microbi_pico2_metrics.json |
| pamap2 | fp32 | True | 69.473 | 99.999992 | 132992 | 614268 | False | - | result.json |
| opportunity | int8 | True | 89.878 | 94.709404 | 134436 | 710748 | True | - | microbi_pico2_metrics.json |
| opportunity | fp32 | True | 249.506 | 99.999992 | 194432 | 662628 | True | - | result.json |
| unimib | int8 | True | 77.154 | 77.168648 | 117988 | 665260 | True | - | microbi_pico2_metrics.json |
| unimib | fp32 | True | 60.455 | 99.999992 | 130016 | 600260 | True | - | result.json |
| skoda | int8 | True | 63.656 | 95.906418 | 91972 | 604444 | True | - | microbi_pico2_metrics.json |
| skoda | fp32 | True | 61.657 | 99.999992 | 108928 | 573924 | True | - | result.json |
| daphnet | int8 | True | 40.267 | 71.953323 | 59604 | 519876 | True | - | microbi_pico2_metrics.json |
| daphnet | fp32 | True | 35.32 | 99.999992 | 65248 | 509980 | True | - | result.json |

## tinyhar:

| Dataset | Variant | Success | Latency (ms) | PyTorch Parity (%) | Arena Used (B) | Flash (B) | Top-1 Match | Failure | Source |
| --- | --- | --- | ---: | ---: | ---: | ---: | --- | --- | --- |
| ucihar | int8 | True | 328.974 | 6.124 | 126916 | 597924 | False | - | pico2_deployment_status.json |
| ucihar | fp32 | True | 770.183 | 94.892021 | 313520 | 650356 | True | - | result.json |
| motionsense | int8 | True | 249.543 | 66.585 | - | - | True | - | baseline_full_20260503 |
| motionsense | fp32 | True | 567.605 | 88.628296 | 229424 | 634956 | True | - | result.json |
| wisdm | int8 | True | 171.717 | 96.965 | - | - | True | - | baseline_full_20260503 |
| wisdm | fp32 | True | 348.835 | 99.639427 | 145328 | 619620 | True | - | result.json |
| pamap2 | int8 | True | 641.389 | 80.223 | - | - | False | - | baseline_full_20260503 |
| pamap2 | fp32 | False | - | - | - | 702876 | - | Failed to resize buffer. Requested: 474240, available 306544, missing: 167696 | result.json |
| opportunity | int8 | False | - | - | - | 713412 | - | Failed to resize buffer. Requested: 492960, available 302012, missing: 190948 | result.json |
| opportunity | fp32 | False | - | - | - | 1008516 | - | Failed to resize buffer. Requested: 1971840, available 122224, missing: 1849616 | result.json |
| unimib | int8 | True | 170.132 | 88.44 | 84868 | 588180 | True | - | result.json |
| unimib | fp32 | True | 349.571 | 99.858528 | 145328 | 620260 | True | - | result.json |
| skoda | int8 | True | 742.74 | 84.792 | 213380 | 592052 | True | - | result.json |
| skoda | fp32 | False | - | - | - | 735188 | - | Failed to resize buffer. Requested: 576000, available 305376, missing: 270624 | result.json |
| daphnet | int8 | True | 173.879 | 0.0 | 67588 | 512420 | True | - | result.json |
| daphnet | fp32 | True | 394.464 | 64.901543 | 160880 | 601060 | True | - | result.json |

## tinierhar:

| Dataset | Variant | Success | Latency (ms) | PyTorch Parity (%) | Arena Used (B) | Flash (B) | Top-1 Match | Failure | Source |
| --- | --- | --- | ---: | ---: | ---: | ---: | --- | --- | --- |
| ucihar | int8 | True | 121.846 | 77.517 | 172468 | 691444 | True | - | pico2_deployment_status.json |
| ucihar | fp32 | True | 229.542 | 99.999992 | 239696 | 657340 | True | - | result.json |
| motionsense | int8 | True | 100.396 | 66.702 | 164788 | 685252 | True | - | pico2_deployment_status.json |
| motionsense | fp32 | True | 145.446 | 99.999985 | 208976 | 637324 | True | - | result.json |
| wisdm | int8 | True | 76.412 | 79.044 | 157108 | 688620 | True | - | result.json |
| wisdm | fp32 | True | 75.715 | 99.999992 | 178256 | 617388 | True | - | result.json |
| pamap2 | int8 | True | 205.091 | 16.485 | 198068 | 721908 | False | - | result.json |
| pamap2 | fp32 | True | 452.712 | 99.999992 | 342096 | 724836 | False | - | result.json |
| opportunity | int8 | True | 930.591 | 27.469 | 351668 | 844172 | False | - | result.json |
| opportunity | fp32 | False | - | - | - | 1123084 | - | Failed to resize buffer. Requested: 808960, available 278608, missing: 530352 | result.json |
| unimib | int8 | True | 77.248 | 81.975 | 157108 | 688836 | True | - | result.json |
| unimib | fp32 | True | 75.513 | 99.999992 | 178256 | 617828 | True | - | result.json |
| skoda | int8 | True | 257.794 | 84.647 | 182244 | 664164 | True | - | result.json |
| skoda | fp32 | True | 523.419 | 99.999992 | 387904 | 743492 | True | - | result.json |
| daphnet | int8 | True | 65.231 | 0.0 | 88340 | 547916 | True | - | result.json |
| daphnet | fp32 | True | 118.322 | 99.999954 | 121392 | 552684 | True | - | result.json |

## deepconvlstm:

| Dataset | Variant | Success | Latency (ms) | PyTorch Parity (%) | Arena Used (B) | Flash (B) | Top-1 Match | Failure | Source |
| --- | --- | --- | ---: | ---: | ---: | ---: | --- | --- | --- |
| ucihar | int8 | False | - | - | - | 2338700 | - | Failed to allocate tail memory. Requested: 16, available 0, missing: 16 | result.json |
| ucihar | fp32 | False | - | - | - | 1650476 | - | Failed to allocate tail memory. Requested: 60, available 12, missing: 48 | result.json |
| motionsense | int8 | False | - | - | - | 2336148 | - | Failed to allocate tail memory. Requested: 16, available 0, missing: 16 | result.json |
| motionsense | fp32 | False | - | - | - | 1645044 | - | Failed to allocate tail memory. Requested: 60, available 12, missing: 48 | result.json |
| wisdm | int8 | False | - | - | - | 2333692 | - | Failed to allocate tail memory. Requested: 16, available 0, missing: 16 | result.json |
| wisdm | fp32 | False | - | - | - | 1639708 | - | Failed to allocate tail memory. Requested: 60, available 12, missing: 48 | result.json |
| pamap2 | int8 | False | - | - | - | 2347716 | - | Failed to allocate tail memory. Requested: 16, available 0, missing: 16 | result.json |
| pamap2 | fp32 | False | - | - | - | 1670100 | - | Failed to allocate tail memory. Requested: 60, available 12, missing: 48 | result.json |
| opportunity | int8 | False | - | - | - | 2397676 | - | Failed to allocate tail memory. Requested: 60, available 44, missing: 16 | result.json |
| opportunity | fp32 | False | - | - | - | 1775612 | - | Failed to allocate tail memory. Requested: 60, available 12, missing: 48 | result.json |
| unimib | int8 | False | - | - | - | 2334892 | - | Failed to allocate tail memory. Requested: 60, available 44, missing: 16 | result.json |
| unimib | fp32 | False | - | - | - | 1640540 | - | Failed to allocate tail memory. Requested: 60, available 12, missing: 48 | result.json |
| skoda | int8 | False | - | - | - | 1787884 | - | Failed to allocate temp memory. Requested: 99576, available 77308, missing: 22268 | result.json |
| skoda | fp32 | False | - | - | - | 1510476 | - | Failed to allocate temp memory. Requested: 99456, available 79532, missing: 19924 | result.json |
| daphnet | int8 | True | 768.78 | 25.482 | 244084 | 1239620 | True | - | result.json |
| daphnet | fp32 | True | 2034.567 | 26.312119 | 287952 | 1273108 | True | - | result.json |
