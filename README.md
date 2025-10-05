# OCTMNIST Retinal Disease Classification using CNNs

### 📘 Overview
This project implements and evaluates a **Convolutional Neural Network (CNN)** model for classifying retinal OCT images into multiple diagnostic categories.  

The notebook demonstrates a full deep-learning workflow—from dataset preparation and visualization to model training, evaluation, and performance comparison between a base CNN and an improved CNN architecture.

---

### 🎯 Objectives
- Build an **image classification pipeline** using PyTorch.
- Understand the impact of **network architecture, image resolution, and optimizer choice** on model performance.
- Train and evaluate both a **Base CNN** and an **Improved CNN** on the **OCTMNIST** retinal dataset.
- Achieve a test accuracy greater than **90 %** using the improved model.
- Present visual evidence such as loss/accuracy curves, confusion matrix, and sample predictions.

---

### 🧠 Dataset
- **Dataset:** [OCTMNIST – MedMNIST Collection](https://medmnist.com/)
- **Image Size:** `64×64` RGB  
- **Classes:** 3 (Retinal diseases – e.g., CNV, DME, DRUSEN)
- **Split:**
  - Training: 108k images  
  - Validation: 10k images  
  - Test: 8k images

Each image corresponds to a retinal OCT scan labeled for a specific disease category.  
The dataset is automatically downloaded and preprocessed within the notebook using PyTorch’s `torchvision.datasets` utilities.

---

### 🏗️ Model Architecture

#### **1️⃣ Base CNN**
A simple convolutional network used as the baseline model.
```text
Conv2d(3, 16, kernel=3) → ReLU → MaxPool
Conv2d(16, 32, kernel=3) → ReLU → MaxPool
Fully Connected (128) → ReLU → Dropout → FC(3)
Optimizer: Adam | Learning Rate: 1e-3
Accuracy ≈ 75 %
```

#### 2️⃣ Improved CNN
A deeper model that improves representation learning and convergence.
```text
Conv2d(3, 32, 3) → BatchNorm → ReLU → MaxPool
Conv2d(32, 64, 3) → BatchNorm → ReLU → MaxPool
Conv2d(64, 128, 3) → BatchNorm → ReLU → MaxPool
Flatten → FC(256) → Dropout(0.4) → ReLU → FC(3)
Optimizer: SGD with Momentum (0.9) | LR = 0.001 | Epochs = 40
Accuracy ≈ 90 – 92 %
```

## ⚙️ Training Setup

| Parameter      | Value                               |
| :------------- | :---------------------------------- |
| Epochs         | 40                                  |
| Batch Size     | 64                                  |
| Optimizer      | SGD with momentum 0.9               |
| Loss Function  | CrossEntropyLoss                    |
| Learning Rate  | 0.001                               |
| Early Stopping | Based on validation loss            |
| Device         | M2 MacBook Pro (CPU/GPU compatible) |


## 📊 Results Summary
| Model        | Train Acc (%) | Test Acc (%) | Observation                                       |
| :----------- | :-----------: | :----------: | :------------------------------------------------ |
| Base CNN     |      ~76      |     75.9     | Underfitting on complex textures                  |
| Improved CNN |    **94.7**   |   **91.3**   | Excellent generalization and smoother convergence |

## 🧾 Key Learnings
- The project illustrates the transition from simple CNNs to deeper, regularized architectures.
- Momentum-based SGD significantly stabilized convergence compared to Adam.
- Proper use of batch normalization and dropout improved both accuracy and robustness.
- Working with medical imaging datasets reinforced the need for careful data normalization and balanced sampling.

## 🏁 Conclusion
- This project successfully demonstrated the end-to-end process of deep learning model development for medical image classification.
- The Improved CNN achieved over 90 % test accuracy, outperforming the base model by ~15 %.
- It highlights best practices in model architecture design, optimization, and evaluation within real-world data constraints.
