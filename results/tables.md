### Baselines across protocols (macro-F1 / benign wrongly flagged)

| model | random | structural-disjoint | source-disjoint |
|---|---|---|---|
| keyword | 0.456 / 37.4% | 0.465 / 39.1% | 0.297 / 43.9% |
| tfidf_lr | 0.876 / 11.0% | 0.849 / 13.4% | 0.653 / 64.2% |
| tfidf_svm | 0.932 / 10.1% | 0.907 / 10.5% | 0.647 / 65.7% |

Source identification from text: accuracy 91.7%, balanced accuracy 76.0% over 10 sources (chance 10.0%).

### Source-disjoint test (unseen sources), default threshold

| model | macro-F1 | benign wrongly flagged | malicious caught | PR-AUC | recall @ 2% FPR* |
|---|---|---|---|---|---|
| TF-IDF all/word_svm | 0.647 | 65.7% | 96.1% | 0.840 | 8.3% |
| TF-IDF all/char_lr | 0.655 | 63.9% | 95.4% | 0.859 | 20.4% |
| TF-IDF group_balanced/char_svm | 0.641 | 66.6% | 96.1% | 0.894 | 31.3% |
| TF-IDF mixed_only/word_lr | 0.683 | 3.7% | 50.4% | 0.928 | 46.0% |
| TF-IDF mixed_only/char_lr | 0.737 | 10.6% | 63.6% | 0.910 | 38.5% |
| TF-IDF size_matched/word_lr (mean of 3 seeds) | 0.673 | 58.0% | 92.2% | 0.800 | 8.3% |
| TF-IDF size_matched/char_lr (mean of 3 seeds) | 0.660 | 59.7% | 91.5% | 0.851 | 16.1% |
| modernbert_source_disjoint_all_s0 | 0.565 | 78.9% | 98.3% | 0.659 | 2.0% |
| modernbert_source_disjoint_mixed_only_s0 | 0.639 | 20.0% | 53.4% | 0.787 | 5.5% |
| lora_qwen2505b_source_disjoint_all_s0 | 0.550 | 80.0% | 96.9% | 0.789 | 15.3% |
| lora_qwen2505b_source_disjoint_mixed_only_s0 | 0.398 | 1.8% | 11.8% | 0.780 | 12.9% |

*recall @ 2% FPR picks the threshold on the test set itself: a ceiling, not a deployable number.

### Source-disjoint test with the threshold fixed on validation (2% FPR there)

| model | benign wrongly flagged (test) | malicious caught (test) |
|---|---|---|
| TF-IDF all/word_svm | 21.7% | 66.6% |
| TF-IDF all/char_lr | 7.9% | 44.9% |
| TF-IDF mixed_only/word_lr | 0.2% | 19.9% |
| TF-IDF mixed_only/char_lr | 1.1% | 32.2% |
| modernbert_source_disjoint_all_s0 | 47.2% | 71.8% |
| modernbert_source_disjoint_mixed_only_s0 | 20.0% | 56.4% |

### Inference (modernbert_source_disjoint_mixed_only_s0, CPU, 4 threads, single request)

| format | size | p50 latency | p95 latency |
|---|---|---|---|
| PyTorch fp32 | - | 154.3 ms | 180.6 ms |
| ONNX fp32 | 599.1 MB | 429.7 ms | 472.8 ms |
| ONNX int8 | 150.7 MB | 229.1 ms | 249.9 ms |

int8 vs PyTorch on the 0.657 -> 0.65 macro-F1 test set: max probability difference 0.72924, 31 verdicts flipped, tokenizer mismatches 0/300.

### Head-only vs whole-skill scoring (modernbert_source_disjoint_mixed_only_s0, threshold fixed on validation)

| scoring | benign wrongly flagged | malicious caught | PR-AUC |
|---|---|---|---|
| first 512 tokens | 20.0% | 56.4% | 0.787 |
| max over windows (up to ~3k tokens) | 20.0% | 51.8% | 0.777 |

