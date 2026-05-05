# CodeBERT-Aug-HN-SCL (Teacher Model)

The CodeBERT-Aug-HN-SCL module implements a robust teacher model for text/code mixed classification. 

It is designed to learn robust representations by explicitly addressing noisy labels and varied structural patterns (e.g., markdown, role markers) in dialogue/code data, making it ideal to later distill into a high-throughput student model.

## Training Stages

The training is divided into two stages to ensure stability and robustness:

### Stage 1: Standard Fine-Tuning
- **Model**: `CodeBERT` + Cross-Entropy Loss
- **Purpose**: Establishes a solid baseline and generates reliable initial predictions and feature embeddings.

### Stage 2: Robust Training
- **Data Augmentation**: Applies 7 label-preserving augmentations (e.g., `remove_role_markers`, `rename_code_variables`, `cn_en_term_swap`) to ensure the model doesn't overfit to superficial patterns.
- **Supervised Contrastive Learning (SCL)**: Pulls features of the same class together while pushing apart different classes.
- **Consistency Regularization**: Minimizes the Symmetric KL Divergence between predictions on the original text and its augmented view.
- **Hard Negative Reweighting**: Re-weights Cross-Entropy loss dynamically based on Stage 1 confidences and nearest-negative similarities, focusing the model on hard samples.

## Dataset Format

The project accepts standard `JSONL` or `CSV` files. Each row must contain at minimum:
```json
{
  "id": 1187,
  "text": "user: Title: ... assistant: ```java ...```",
  "label_text": "Java Spring"
}
```

## Commands

**1. Train Stage 1**
```bash
python -m log_classifier.teacher.train_stage1 --config configs/teacher/train_stage1.yaml
```

**2. Train Stage 2**
```bash
python -m log_classifier.teacher.train_stage2 --config configs/teacher/train_stage2.yaml
```

**3. Evaluate Checkpoint**
```bash
python -m log_classifier.teacher.evaluate --config configs/teacher/eval.yaml
```

## Hard Negative Reweighting

During the transition from Stage 1 to Stage 2, the system evaluates the training set using the Stage 1 model. It assigns a high weight to samples that:
1. Are misclassified.
2. Have low prediction confidence.
3. Have a very close nearest-neighbor belonging to a different class.

These weights are saved and then applied to scale the CE loss during Stage 2.

## Robust Evaluation

The evaluation script not only tests the model on the clean test set but explicitly generates perturbed versions of the test set using the 7 augmentation strategies. It outputs a JSON report detailing both clean performance and performance under each specific structural perturbation, ensuring true robustness.
