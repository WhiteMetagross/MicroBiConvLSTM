# ESP32 FP32 and Quantized Deployment Matrix

Generated: 2026-05-11T13:57:28.217287+00:00

## Coverage:

| Model | Quantized Runs | Quantized Fails | Quantized Missing | FP32 Runs | FP32 Fails | FP32 Missing |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| microbi | 8 | 0 | 0 | 7 | 1 | 0 |
| tinyhar | 4 | 4 | 0 | 3 | 5 | 0 |
| tinierhar | 6 | 2 | 0 | 3 | 5 | 0 |
| deepconvlstm | 0 | 8 | 0 | 0 | 8 | 0 |

## microbi:

| Dataset | Variant | Quant Recipe | Success | Latency (ms) | PyTorch Parity (%) | Desktop Parity (%) | Arena Used (B) | Flash (B) | Top-1 Match | Failure | Source |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | --- | --- | --- |
| ucihar | int8 | full_int8 | True | 176.985 | 99.27562 | 99.27561496257279 | 109344 | - | True | - | esp32_paper_results.json |
| ucihar | fp32 | float32 | True | 72.29 | 99.999992 | 99.99999561432324 | 121356 | 493296 | True | - | esp32_native_deployment_status.json |
| motionsense | int8 | full_int8 | True | 180.574 | 96.05574 | 97.33711445586111 | 109032 | 561472 | - | - | esp32_paper_results.json |
| motionsense | fp32 | float32 | True | 75.134 | 99.999992 | 99.99998386094802 | 121356 | 490736 | False | - | esp32_native_deployment_status.json |
| wisdm | int8 | full_int8 | True | 185.365 | 98.155388 | 98.55791126312103 | 109032 | 559728 | - | - | esp32_paper_results.json |
| wisdm | fp32 | float32 | True | 75.907 | 99.999992 | 99.99999229480045 | 121356 | 488272 | True | - | esp32_native_deployment_status.json |
| pamap2 | int8 | full_int8 | True | 178.497 | 96.883163 | 96.88315775763189 | 109776 | - | False | - | esp32_paper_results.json |
| pamap2 | fp32 | float32 | True | 73.215 | 99.999992 | 99.9999965253679 | 124332 | 504624 | False | - | esp32_native_deployment_status.json |
| opportunity | int8 | full_int8 | True | 208.501 | 99.176315 | 99.1812629901176 | 125432 | - | True | - | esp32_paper_results.json |
| opportunity | fp32 | float32 | False | - | - | 99.99999380199546 | - | 551024 | - | Failed to allocate tensor arena from internal SRAM. | esp32_native_deployment_status.json |
| unimib | int8 | full_int8 | True | 176.451 | 95.906059 | 96.0075111174589 | 109040 | 560880 | - | - | esp32_paper_results.json |
| unimib | fp32 | float32 | True | 73.148 | 99.999992 | 99.99998743625844 | 121356 | 489760 | True | - | esp32_native_deployment_status.json |
| skoda | int8 | full_int8 | True | 146.551 | 99.58223 | 99.58223733114507 | 85152 | - | True | - | esp32_paper_results.json |
| skoda | fp32 | float32 | True | 60.735 | 99.999992 | 99.99999439860949 | 102444 | 463968 | True | - | esp32_native_deployment_status.json |
| daphnet | int8 | full_int8 | True | 95.161 | 98.60244 | 98.60243738520211 | 55000 | 413328 | - | - | esp32_paper_results.json |
| daphnet | fp32 | float32 | True | 40.897 | 99.999985 | 99.99998161266497 | 60940 | 397312 | True | - | esp32_native_deployment_status.json |

## tinyhar:

| Dataset | Variant | Quant Recipe | Success | Latency (ms) | PyTorch Parity (%) | Desktop Parity (%) | Arena Used (B) | Flash (B) | Top-1 Match | Failure | Source |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | --- | --- | --- |
| ucihar | int8 | int16_activations_int8_weights | False | - | - | 88.58516357322091 | - | 497552 | - | Failed to allocate tensor arena from internal SRAM. | baseline_int8_quantfix_20260511.json |
| ucihar | fp32 | float32 | False | - | - | 94.89201508882765 | - | 563168 | - | Failed to allocate tensor arena from internal SRAM. | esp32_native_deployment_status.json |
| motionsense | int8 | int16_activations_int8_weights | True | 735.682 | 88.105583 | 88.1072709815788 | 148108 | 497552 | True | - | baseline_int8_quantfix_20260511.json |
| motionsense | fp32 | float32 | False | - | - | 88.62829341199568 | - | 539376 | - | Failed to allocate tensor arena from internal SRAM. | esp32_native_deployment_status.json |
| wisdm | int8 | full_int8 | True | 347.674 | 99.205788 | 98.85554909580023 | 78400 | - | True | - | esp32_paper_results.json |
| wisdm | fp32 | float32 | True | 357.144 | 99.639427 | 99.63943046105041 | 141372 | 524064 | True | - | esp32_native_deployment_status.json |
| pamap2 | int8 | full_int8 | False | - | - | 85.1899012673794 | - | - | - | Failed to allocate tensor arena from internal SRAM. | esp32_paper_results.json |
| pamap2 | fp32 | float32 | False | - | - | 99.7877510000843 | - | 609008 | - | Failed to allocate tensor arena from internal SRAM. | esp32_native_deployment_status.json |
| opportunity | int8 | full_int8 | False | - | - | 87.78229093240644 | - | - | - | Failed to allocate tensor arena from internal SRAM. | esp32_paper_results.json |
| opportunity | fp32 | float32 | False | - | - | 98.98855831871884 | - | 912720 | - | Failed to allocate tensor arena from internal SRAM. | esp32_native_deployment_status.json |
| unimib | int8 | full_int8 | True | 346.616 | 97.251785 | 96.173519665147 | 78704 | - | True | - | esp32_paper_results.json |
| unimib | fp32 | float32 | True | 358.344 | 99.858528 | 99.85852722121622 | 141372 | 525536 | True | - | esp32_native_deployment_status.json |
| skoda | int8 | full_int8 | False | - | - | 93.06683264729647 | - | - | - | Failed to allocate tensor arena from internal SRAM. | esp32_paper_results.json |
| skoda | fp32 | float32 | False | - | - | 97.75885497916721 | - | 641008 | - | Failed to allocate tensor arena from internal SRAM. | esp32_native_deployment_status.json |
| daphnet | int8 | int16_activations_int8_weights | True | 518.13 | 69.742943 | 69.73239939207608 | 100684 | 430752 | True | - | baseline_int8_quantfix_20260511.json |
| daphnet | fp32 | float32 | True | 343.19 | 64.901558 | 64.90158881445186 | 159100 | 504176 | True | - | esp32_native_deployment_status.json |

## tinierhar:

| Dataset | Variant | Quant Recipe | Success | Latency (ms) | PyTorch Parity (%) | Desktop Parity (%) | Arena Used (B) | Flash (B) | Top-1 Match | Failure | Source |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | --- | --- | --- |
| ucihar | int8 | full_int8 | True | 209.216 | 92.011673 | 90.18308882451926 | 164848 | 619856 | True | - | esp32_paper_results.json |
| ucihar | fp32 | float32 | False | - | - | 99.99998033880533 | - | 577536 | - | Failed to allocate tensor arena from internal SRAM. | esp32_native_deployment_status.json |
| motionsense | int8 | int16_activations_int8_weights | True | 231.314 | 91.593452 | 91.59455204876933 | 173996 | 608832 | True | - | baseline_int8_quantfix_20260511.json |
| motionsense | fp32 | float32 | False | - | - | 99.99995185135256 | - | 556272 | - | Failed to allocate tensor arena from internal SRAM. | esp32_native_deployment_status.json |
| wisdm | int8 | full_int8 | True | 132.246 | 98.50573 | 98.23913997463649 | 149488 | 607536 | True | - | esp32_paper_results.json |
| wisdm | fp32 | float32 | True | 88.919 | 99.999992 | 99.99998473335033 | 171708 | 536320 | True | - | esp32_native_deployment_status.json |
| pamap2 | int8 | full_int8 | False | - | - | 74.76562879731183 | - | 644784 | - | Failed to resize buffer. Requested: 48640, available 38536, missing: 10104 | esp32_paper_results.json |
| pamap2 | fp32 | float32 | False | - | - | 99.99998181785963 | - | 645504 | - | Failed to allocate tensor arena from internal SRAM. | esp32_native_deployment_status.json |
| opportunity | int8 | full_int8 | False | - | - | 95.20163703914754 | - | 773904 | - | Failed to resize buffer. Requested: 202240, available 38536, missing: 163704 | esp32_paper_results.json |
| opportunity | fp32 | float32 | False | - | - | 99.99998248958109 | - | 1041776 | - | Failed to allocate tensor arena from internal SRAM. | esp32_native_deployment_status.json |
| unimib | int8 | full_int8 | True | 130.707 | 98.953369 | 98.85519429535582 | 149488 | 608640 | True | - | esp32_paper_results.json |
| unimib | fp32 | float32 | True | 86.743 | 99.999992 | 99.99998773601106 | 171708 | 537632 | True | - | esp32_native_deployment_status.json |
| skoda | int8 | full_int8 | True | 369.053 | 99.399971 | 99.3864845469894 | 176400 | 584512 | True | - | esp32_paper_results.json |
| skoda | fp32 | float32 | False | - | - | 99.99999370435563 | - | 663840 | - | Failed to allocate tensor arena from internal SRAM. | esp32_native_deployment_status.json |
| daphnet | int8 | int16_activations_int8_weights | True | 154.479 | 64.089409 | 64.08939793453486 | 97644 | 471696 | True | - | baseline_int8_quantfix_20260511.json |
| daphnet | fp32 | float32 | True | 113.649 | 99.999947 | 99.99993354047463 | 118396 | 470336 | True | - | esp32_native_deployment_status.json |

## deepconvlstm:

| Dataset | Variant | Quant Recipe | Success | Latency (ms) | PyTorch Parity (%) | Desktop Parity (%) | Arena Used (B) | Flash (B) | Top-1 Match | Failure | Source |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | --- | --- | --- |
| ucihar | int8 | full_int8 | False | - | - | 98.65932132128106 | - | - | - | Failed to allocate tensor arena from internal SRAM. | esp32_paper_results.json |
| ucihar | fp32 | float32 | False | - | - | 99.99999097666824 | - | 1528032 | - | Failed to allocate tensor arena from internal SRAM. | esp32_native_deployment_status.json |
| motionsense | int8 | full_int8 | False | - | - | 24.181919703907116 | - | - | - | Failed to allocate tensor arena from internal SRAM. | esp32_paper_results.json |
| motionsense | fp32 | float32 | False | - | - | 99.99995240610528 | - | 1523200 | - | Failed to allocate tensor arena from internal SRAM. | esp32_native_deployment_status.json |
| wisdm | int8 | full_int8 | False | - | - | 96.04841514432812 | - | - | - | Failed to allocate tensor arena from internal SRAM. | esp32_paper_results.json |
| wisdm | fp32 | float32 | False | - | - | 99.99999526979637 | - | 1523200 | - | Failed to allocate tensor arena from internal SRAM. | esp32_native_deployment_status.json |
| pamap2 | int8 | full_int8 | False | - | - | 98.13626765030007 | - | - | - | Failed to allocate tensor arena from internal SRAM. | esp32_paper_results.json |
| pamap2 | fp32 | float32 | False | - | - | 99.99999529873084 | - | 1549360 | - | Failed to allocate tensor arena from internal SRAM. | esp32_native_deployment_status.json |
| opportunity | int8 | full_int8 | False | - | - | 95.74195826671541 | - | - | - | Failed to allocate tensor arena from internal SRAM. | esp32_paper_results.json |
| opportunity | fp32 | float32 | False | - | - | 99.99999127675336 | - | 1654032 | - | Failed to allocate tensor arena from internal SRAM. | esp32_native_deployment_status.json |
| unimib | int8 | full_int8 | False | - | - | 90.31139476882994 | - | - | - | Failed to allocate tensor arena from internal SRAM. | esp32_paper_results.json |
| unimib | fp32 | float32 | False | - | - | 99.99999058861965 | - | 1524048 | - | Failed to allocate tensor arena from internal SRAM. | esp32_native_deployment_status.json |
| skoda | int8 | full_int8 | False | - | - | 97.64878586842514 | - | - | - | Failed to allocate tensor arena from internal SRAM. | esp32_paper_results.json |
| skoda | fp32 | float32 | False | - | - | 99.99999617759198 | - | 1393536 | - | Failed to allocate tensor arena from internal SRAM. | esp32_native_deployment_status.json |
| daphnet | int8 | full_int8 | False | - | - | 98.4039522029362 | - | - | - | Failed to allocate tensor arena from internal SRAM. | esp32_paper_results.json |
| daphnet | fp32 | float32 | False | - | - | 99.99999612782375 | - | 1149360 | - | Failed to allocate tensor arena from internal SRAM. | esp32_native_deployment_status.json |

