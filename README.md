# Chest X-Ray Pneumonia Classification

A deep learning-based system for automated pneumonia classification from chest X-ray images using a custom lightweight CNN architecture.

## Overview

This project implements a binary classification model to distinguish between normal and pneumonia cases in chest X-ray images. The model achieves high accuracy while maintaining a small footprint, making it suitable for deployment in resource-constrained environments.

## Features

- **Binary Classification**: Classifies chest X-rays as pneumonia vs. normal
- **Lightweight Architecture**: Custom ResNet variant with ~600K trainable parameters
- **Interpretability**: GradCAM visualization for model decision explanation
- **Comprehensive Metrics**: Accuracy, Precision, Recall, F1-Score, ROC-AUC
- **Threshold Optimization**: Configurable decision threshold (default: 0.25)
- **Data Augmentation**: Advanced augmentation pipeline for robust training

## Dataset

The model is trained on a combined dataset from two sources:

1. **Kaggle Chest X-Ray Dataset**: 
   - 1,000 pneumonia cases (selected)
   - 500 normal cases (selected)

2. **Dhuli Dataset** (custom dataset):
   - Additional pneumonia and normal cases

**Data Split**: The model uses separate unseen test data for final evaluation.

## Requirements

```
torch>=1.9.0
torchvision>=0.10.0
numpy>=1.19.0
matplotlib>=3.3.0
opencv-python>=4.5.0
scikit-learn>=0.24.0
psutil
opendatasets
```

Install dependencies:

```bash
pip install -r requirements.txt
```

## Model Architecture

The model uses a custom **SmallResNet** architecture with the following characteristics:

- **Input**: Grayscale chest X-ray images (224×224×1)
- **Parameters**: ~600K trainable parameters
- **Backbone**: ResNet-inspired with BasicBlocks
- **Output**: Binary classification (Normal/Pneumonia)
- **Regularization**: Dropout (p=0.2)

## Data Preprocessing

### Training Transforms
- Resize to 224×224
- Gaussian blur (σ=0.1-0.5)
- Random sharpness adjustment
- Random rotation (±10°)
- Random affine transformations
- Color jitter (brightness, contrast)
- Random erasing
- Normalization (mean=0.4755, std=0.2145)

### Validation/Test Transforms
- Resize to 224×224
- Grayscale conversion
- Normalization (same as training)

## Training

The model is trained with:

- **Loss Function**: CrossEntropyLoss
- **Optimizer**: Adam (with learning rate scheduling)
- **Early Stopping**: Monitors validation accuracy with patience
- **Best Model Selection**: Saves model with highest validation accuracy

### Training Command

```python
python train.py --epochs 50 --batch-size 32 --lr 0.001
```

## Evaluation

The model provides comprehensive evaluation metrics:

- **Accuracy**: Overall classification accuracy
- **Precision**: Positive predictive value
- **Recall**: Sensitivity
- **F1-Score**: Harmonic mean of precision and recall
- **ROC-AUC**: Area under the receiver operating characteristic curve
- **Confusion Matrix**: True/False positives and negatives
- **Inference Latency**: Average time per image

### Running Evaluation

```python
python evaluate.py --model-path smallresnet_final.pth --test-data /path/to/test/data
```

## Model Interpretability

The project includes **GradCAM** (Gradient-weighted Class Activation Mapping) visualization to understand which regions of the X-ray image the model focuses on when making predictions.

### GradCAM Features
- Highlights important regions in the X-ray
- Separate visualizations for normal and pneumonia cases
- Overlay heatmaps on original images
- Color-coded predictions (green = correct, red = incorrect)

## Usage

### 1. Prepare Your Data

Organize your dataset in the following structure:

```
data/
├── train/
│   ├── normal/
│   └── pneumonia/
├── val/
│   ├── normal/
│   └── pneumonia/
└── test/
    ├── normal/
    └── pneumonia/
```

### 2. Train the Model

```python
# Load and prepare data
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# Initialize model
from model import SmallResNet
model = SmallResNet(num_classes=2, dropout_p=0.2)

# Train
results = train(
    model=model,
    train_dataloader=train_loader,
    val_dataloader=val_loader,
    optimizer=optimizer,
    epochs=50,
    patience=10
)
```

### 3. Make Predictions

```python
import torch
from model import SmallResNet

# Load model
checkpoint = torch.load('smallresnet_final.pth')
model = SmallResNet()
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# Predict
with torch.no_grad():
    outputs = model(image_tensor)
    probability = torch.softmax(outputs, dim=1)[:, 1]
    prediction = (probability >= 0.25).long()
```

## Performance

The model achieves strong performance on the test set:

| Metric | Value |
|--------|-------|
| Accuracy | High performance on unseen data |
| Precision | Measures positive predictive value |
| Recall | Measures sensitivity |
| F1-Score | Balanced precision-recall metric |
| ROC-AUC | Area under ROC curve |
| Inference Time | Low latency per image |

*Note: Specific values are stored in the trained model checkpoint.*

## Model Checkpoint Information

The saved model checkpoint includes:

- Model state dict
- Number of training epochs
- Total and trainable parameters
- FLOPs (floating point operations)
- Training dataset size
- Optimizer configuration

## Visualizations

The project generates the following visualizations:

1. **ROC Curve** (`roc_curve.png`): Shows true positive rate vs. false positive rate
2. **Confusion Matrix** (`confusion_matrix.png`): Displays classification results
3. **GradCAM - Normal** (`gradcam_normal.png`): Attention maps for normal cases
4. **GradCAM - Pneumonia** (`gradcam_pneumonia.png`): Attention maps for pneumonia cases

## File Structure

```
├── model.py                    # SmallResNet architecture
├── train.py                    # Training script
├── evaluate.py                 # Evaluation script
├── utils.py                    # Helper functions
├── data_preparation.py         # Dataset preparation
├── gradcam.py                  # GradCAM implementation
├── requirements.txt            # Dependencies
└── README.md                   # This file
```

## Threshold Tuning

The model uses a configurable decision threshold (default: 0.25) for binary classification. This can be adjusted based on your use case:

- **Lower threshold** (e.g., 0.20): Higher recall, lower precision (fewer missed cases)
- **Higher threshold** (e.g., 0.30): Higher precision, lower recall (fewer false alarms)

## Memory Footprint

The model is designed to be lightweight:

- **Model Size**: ~2-3 MB (.pth file)
- **GPU Memory**: Low memory footprint during inference
- **CPU Compatible**: Can run on CPU for inference

## Citation

If you use this code in your research, please cite:

```bibtex
@misc{chest_xray_pneumonia_classification,
  title={Chest X-Ray Pneumonia Classification using SmallResNet},
  author={Your Name},
  year={2024},
  url={https://github.com/yourusername/chest-xray-pneumonia}
}
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Kaggle Chest X-Ray Dataset: [Chest X-Ray Images (Pneumonia)](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia)
- Custom Dhuli Dataset for additional training data
- PyTorch framework for deep learning implementation

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## Contact

For questions or feedback, please open an issue on GitHub.

## Future Improvements

- [ ] Multi-class classification (viral vs. bacterial pneumonia)
- [ ] Model ensemble for improved accuracy
- [ ] Mobile deployment (TensorFlow Lite, ONNX)
- [ ] Web interface for easy inference
- [ ] Integration with DICOM medical imaging standard
- [ ] Additional explainability methods (LIME, SHAP)

---

**Note**: This is a research/educational project. For medical applications, please consult with healthcare professionals and ensure compliance with relevant medical device regulations.
