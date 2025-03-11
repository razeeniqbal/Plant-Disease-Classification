# Image Classification Model Comparison

![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange)
![Keras](https://img.shields.io/badge/Keras-2.x-red)
![Python](https://img.shields.io/badge/Python-3.x-blue)

A comprehensive comparison of custom CNN architecture versus VGG16 transfer learning for image classification tasks, demonstrating practical implementation of deep learning concepts.

## 📚 Project Overview

This project explores two distinct approaches to image classification:

1. **Custom CNN Architecture**: A hand-crafted convolutional neural network built from scratch with multiple convolutional layers, batch normalization, and dense layers.

2. **VGG16 Transfer Learning**: Leveraging a pre-trained VGG16 model (trained on ImageNet) with frozen base layers and custom classification layers on top.

Each model underwent 5 separate training phases to ensure statistical reliability of the results, with both training and validation metrics captured for comprehensive performance evaluation.

## 🧠 Model Architectures

### Custom CNN
```
Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)),
    BatchNormalization(),
    MaxPooling2D(2, 2),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    Flatten(),
    Dense(256, activation='relu'),
    Dense(num_classes, activation='softmax')
])
```

### VGG16 Transfer Learning
```
base_model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
base_model.trainable = False
x = Flatten()(base_model.output)
x = Dense(256, activation='relu')(x)
predictions = Dense(num_classes, activation='softmax')(x)
model = Model(inputs=base_model.input, outputs=predictions)
```

## 📊 Results & Analysis

### Custom CNN Results
| Phase | Train Accuracy | Validation Accuracy | Train Loss | Validation Loss |
|-------|----------------|---------------------|------------|-----------------|
| 1     | 0.9964         | 0.4883              | 0.0215     | 4.2453          |
| 2     | 0.9891         | 0.4841              | 0.0437     | 4.1427          |
| 3     | 0.9879         | 0.4690              | 0.0418     | 4.1755          |
| 4     | 0.9962         | 0.5201              | 0.0131     | 3.1908          |
| 5     | 0.9897         | 0.4421              | 0.0505     | 4.4042          |
| **Avg**   | **0.9919**         | **0.4807**              | **0.0341**     | **4.0317**          |

### VGG16 Transfer Learning Results
| Phase | Train Accuracy | Validation Accuracy | Train Loss | Validation Loss |
|-------|----------------|---------------------|------------|-----------------|
| 1     | 0.9925         | 0.6779              | 0.0363     | 1.6280          |
| 2     | 0.9883         | 0.6888              | 0.0489     | 1.3832          |
| 3     | 0.9893         | 0.6888              | 0.0445     | 1.6508          |
| 4     | 0.9962         | 0.6795              | 0.0274     | 1.3236          |
| 5     | 0.9646         | 0.6703              | 0.1124     | 1.6252          |
| **Avg**   | **0.9862**         | **0.6811**              | **0.0539**     | **1.5222**          |

## 🔍 Key Findings

1. **Transfer Learning Superiority**: The VGG16 transfer learning model significantly outperformed the custom CNN, achieving an average validation accuracy of 68.11% compared to 48.07% for the custom model.

2. **Overfitting**: Both models exhibited signs of overfitting, with training accuracies near 99% but substantially lower validation accuracies. This suggests that:
   - Additional regularization techniques might be beneficial
   - More diverse training data could improve generalization

3. **Validation Loss Disparity**: The VGG16 model maintained a much lower validation loss (1.52 vs 4.03), indicating better generalization capabilities despite the overfitting.

4. **Training Stability**: The VGG16 model showed more consistent performance across training phases, with lower variance in validation metrics.

## 🛠️ Implementation Details

### Training Configuration
- **Optimizer**: Adam
- **Loss Function**: Categorical Cross-Entropy
- **Batch Size**: 32
- **Epochs**: 15 per phase
- **Image Size**: 224×224×3
- **Data Split**: 80% training, 20% validation

### Data Processing
- Images were rescaled (1/255)
- Training data used a validation split of 0.2
- Class mode was set to 'categorical'

## 💡 Conclusions & Learnings

This experiment demonstrates the power of transfer learning in computer vision tasks. While building custom models from scratch provides greater flexibility and learning opportunities, leveraging pre-trained models like VGG16 offers significant performance advantages, especially with limited datasets.

Key learnings:
- Transfer learning provides a strong baseline with minimal tuning
- Custom architectures require more careful regularization to avoid overfitting
- Multiple training runs (phases) help establish reliable performance metrics
