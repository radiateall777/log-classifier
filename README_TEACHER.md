# UniXcoder Teacher & Robust Distillation Pipeline

This module implements the teacher-model training pipeline for text/code mixed log classification.

After several backbone and training-strategy comparisons, the current strongest teacher baseline is based on:

```text
microsoft/unixcoder-base + Cross-Entropy fine-tuning
```
## Step1:Clean CE-only Teacher Fine-Tuning
Backbone: microsoft/unixcoder-base
Head: single linear 5-class classifier
Loss: Cross-Entropy
Purpose:
establish the strongest clean teacher baseline;
avoid instability from additional auxiliary heads or metric losses;
produce reliable logits for later student distillation.

Recommended command:
```
python -m log_classifier.teacher.train_stage1_ce \
  --config configs/teacher/ce_unixcoder_seed42.yaml

```

The clean teacher checkpoint is saved under:
```
outputs/teacher/ce_unixcoder_seed42/best/
```

Expected artifacts:
```
best/
├── pytorch_model.bin
├── tokenizer files
├── label_mapping.json
├── eval_results.json
├── config_snapshot.json
└── dev_logits.pt
```