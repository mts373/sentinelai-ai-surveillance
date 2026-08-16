\# SentinelAI — Experiment Tracking



This document records the model-development experiments performed during the SentinelAI project.



MLflow was integrated after the initial training and evaluation work. Therefore, the first four MLflow runs are explicitly recorded as historical experiments. The MLflow records preserve the parameters, metrics, tags, and notes associated with those completed experiments.



\---



\## Experiment 1 — Baseline QLoRA / SFT



\### Objective



Establish the initial fine-tuned SentinelAI model based on Qwen2.5-VL-7B-Instruct.



\### Configuration



| Parameter | Value |

|---|---|

| Base model | Qwen/Qwen2.5-VL-7B-Instruct |

| Method | QLoRA |

| Dataset | `sft\_train.json` |

| Epochs | 1 |

| Batch size | 1 |

| Gradient accumulation | 8 |

| Learning rate | `1e-4` |

| Warmup steps | 2 |

| LoRA rank | 16 |

| LoRA alpha | 32 |

| LoRA dropout | 0.05 |

| Minimum pixels | 200704 |

| Maximum pixels | 602112 |



\### MLflow Run



`baseline-qlora-sft-historical`



\### Interpretation



This experiment established the baseline fine-tuned model that was subsequently evaluated using the human-ground-truth temporal evaluation pipeline.



\---



\## Experiment 2 — Human-Ground-Truth Temporal Evaluation



\### Objective



Evaluate the baseline model at the temporal-window level and identify errors for targeted corrective training.



\### Configuration



| Parameter | Value |

|---|---|

| Evaluation level | Temporal window |

| Evaluation samples | 265 |

| Classes | Normal, Fire, Fight, Road Accident |



\### Result



\*\*Accuracy: 88.30%\*\*



\### Error Analysis



\- Total temporal windows: 265

\- Incorrect windows: 31



The incorrect windows were used as the basis for constructing the targeted corrective dataset.



\### MLflow Run



`human-ground-truth-temporal-evaluation`



\### Significance



This experiment changed the development strategy from simply training the model further to identifying specific temporal failure cases.



\---



\## Experiment 3 — Corrective QLoRA v3



\### Objective



Improve the model using examples derived from errors identified during the human-ground-truth temporal evaluation.



\### Corrective Dataset



\*\*139 records\*\*



\### Training Configuration



| Parameter | Value |

|---|---|

| Base model | Qwen/Qwen2.5-VL-7B-Instruct |

| Method | QLoRA |

| Dataset | `corrective\_sft\_train.json` |

| Epochs | 1 |

| Batch size | 1 |

| Gradient accumulation | 8 |

| Effective batch size | 8 |

| Learning rate | `5e-5` |

| Warmup steps | 2 |

| LoRA rank | 16 |

| LoRA alpha | 32 |

| LoRA dropout | 0.05 |

| Video sampling FPS | 0.5 |

| Maximum video frames | 8 |

| Maximum pixels/frame | 401408 |

| Memory optimization | 4-bit QLoRA + gradient checkpointing + controlled video sampling |



\### Training Result



\*\*Training loss: 1.9483672976493835\*\*



\### MLflow Run



`corrective-qlora-v3-historical`



\### Purpose



The corrective model was created specifically to address errors discovered during Experiment 2.



It was subsequently tested on a fresh held-out video set.



\---



\## Experiment 4 — Fresh Held-Out Video-Level Evaluation



\### Objective



Compare the original LoRA model against the corrective QLoRA v3 model on a fresh held-out video-level evaluation.



\### Evaluation



\- Videos evaluated: 13

\- Models compared: 2

\- Classes: Normal, Fire, Fight, Road Accident



\### Results



| Model | Correct Videos | Accuracy |

|---|---:|---:|

| Original LoRA | 12 / 13 | 92.31% |

| Corrective QLoRA v3 | 12 / 13 | 92.31% |



\### Improvement



\*\*0.00 percentage points\*\*



\### Per-Class Results



| Class | Original LoRA | Corrective v3 |

|---|---:|---:|

| Normal | 100.00% | 100.00% |

| Fire | 100.00% | 100.00% |

| Fight | 0.00% | 0.00% |

| Road Accident | 100.00% | 100.00% |



\### MLflow Run



`fresh-holdout-video-level-comparison`



\---



\# Experiment Comparison



| Stage | Experiment | Main Result |

|---|---|---|

| 1 | Baseline QLoRA/SFT | Initial fine-tuned model |

| 2 | Human-ground-truth temporal evaluation | 88.30% on 265 windows |

| 3 | Corrective QLoRA v3 | Training loss 1.94837 |

| 4 | Fresh video-level comparison | 92.31% for both models |



\---



\# Development Decision



The corrective training experiment did not improve the fresh video-level accuracy.



\*\*Original LoRA:\*\* 92.31%



\*\*Corrective QLoRA v3:\*\* 92.31%



\*\*Improvement:\*\* 0.00 percentage points



Therefore, simply increasing training epochs or continuing corrective training without additional analysis is not justified by this experiment.



The next model-improvement step should instead focus on:



1\. Additional error analysis

2\. Expanding the independent evaluation set

3\. Investigating the Fight-class failure

4\. Improving the corrective dataset

5\. Evaluating future changes on a fresh holdout



\---



\# Reproducibility



The experiments are tracked using MLflow.



\### MLflow Experiment



`SentinelAI - Qwen2.5-VL Experiments`



\### Historical Runs



1\. `baseline-qlora-sft-historical`

2\. `human-ground-truth-temporal-evaluation`

3\. `corrective-qlora-v3-historical`

4\. `fresh-holdout-video-level-comparison`



The MLflow tracking database is stored locally and excluded from Git.



The tracking implementation is version-controlled in:



`mlflow\_tracking/log\_existing\_experiments.py`



Future training and evaluation runs should ideally log parameters and metrics directly from their respective pipelines rather than reconstructing them afterward.

