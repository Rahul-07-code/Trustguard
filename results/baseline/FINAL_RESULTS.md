# Final Results

## Dataset
- Classes: 21
- Features: 46
- Train samples: 5392858
- Validation samples: 1155488
- Test samples: 1155597

## Model
- Algorithm: XGBoost CUDA classifier
- Objective: multi:softprob
- Best iteration: 210
- Model size: 20.71 MB

## Performance
- Accuracy: 0.999682
- Macro precision: 0.996904
- Macro recall: 0.997158
- Macro F1: 0.997025
- Weighted F1: 0.999682
- Training time: 272.86 seconds
- Inference/evaluation time: 3.88 seconds

## Resources
- Peak RAM: 11.83 GB
- RAM limit: 14 GB
- RAM status: PASS
- Peak GPU memory: 2.64 GB
- GPU: NVIDIA GeForce RTX 4060 Laptop GPU
- GPU training: CONFIRMED

## Protocol
- Test data was not used for tuning or early stopping.
- Train data was used for fitting; validation data was used for early stopping; test data was evaluated once after model selection.

Model: `results/model/xgboost_21class_final.json`
Results directory: `results/`
