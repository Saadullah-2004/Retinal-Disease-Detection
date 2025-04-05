# AI-Driven Retinal Disease Detection Tool

## Team Members
- Tanish Roy
- Dev Shah
- Saadullah Shazad
- Inaam Azeezur-Rahman

## Overview
This project focuses on developing an **AI-driven diagnostic assistant** for **retinal disease detection**. The tool automates the interpretation of **retinal images**, assisting healthcare professionals in diagnosing diseases such as **diabetic retinopathy (DR)** and **age-related macular degeneration (AMD)**. By integrating **deep learning models and explainability mechanisms**, the system provides reliable and transparent medical diagnoses.

## Key Features
### 1️⃣ **Automated Disease Detection**
- Uses **deep learning models** to classify **retinal diseases** and their severity.
- Detects conditions like **diabetic retinopathy, AMD**, and other abnormalities.

### 2️⃣ **Multi-Step Reasoning**
- Integrates multiple AI models for **segmentation, classification, and feature extraction**.
- Dynamically selects the best model for different **detection tasks**.

### 3️⃣ **Explainability & Transparency**
- Utilizes **Grad-CAM heatmaps** to highlight important retinal areas.
- Provides **visual explanations** to support clinical decision-making.

### 4️⃣ **Adaptive Learning**
- Improves accuracy with **continuous learning** from real-world data.
- Can handle more complex and nuanced cases over time.

### 5️⃣ **Interoperability**
- Seamlessly integrates with **Electronic Health Records (EHRs)**.
- Works in **local hospital systems** or cloud-based environments.

### 6️⃣ **Diagnostic Report Generation**
- Generates **automated PDF reports** including diagnosis, severity score, and insights.
- Provides detailed **medical documentation** for clinicians.

### 7️⃣ **Efficiency & Scalability**
- Reduces **diagnostic time** for routine screenings.
- Scales to **large datasets** for hospital and research use.

## Dataset
- **APTOS 2019 Blindness Detection Dataset** ([Kaggle](https://www.kaggle.com/c/aptos2019-blindness-detection))
  - ~3,662 images labeled for **five severity levels**.
- **DRIVE Dataset** ([Grand Challenge](https://drive.grand-challenge.org/))
  - 40 images with annotated **vessel segmentation data**.

## Classification Categories
- **0** - No DR
- **1** - Mild DR
- **2** - Moderate DR
- **3** - Severe DR
- **4** - Proliferative DR

## System Architecture
1. **Preprocessing**
   - Image resizing to **224x224**.
   - **Normalization** for model input consistency.
   - **Data augmentation**: rotation, flips, contrast adjustments.

2. **Model Training**
   - **Modified ResNet50** with additional layers for classification.
   - **Optimizer**: Adam (learning rate = **0.0001**).
   - **Loss function**: Sparse categorical crossentropy.

3. **Model Explainability**
   - **Saliency maps** and **Grad-CAM heatmaps** for interpretability.
   - Enhances trust in **AI-based diagnoses**.



## Team Responsibilities
| Team Member         | Responsibility |
|---------------------|---------------|
| **Tanish Roy**     | Research AI models, implement classification algorithms |
| **Dev Shah**       | Train models, tune hyperparameters, evaluate performance |
| **Saadullah Shazad** | Handle dataset processing, data augmentation, and evaluation metrics |
| **Inaam Azeezur-Rahman** | Develop visualizations, generate reports, deploy Flask API |

## Repository Setup
- **GitHub Repository**: [Retinal-Disease-Detection](https://github.com/Saadullah-2004/Retinal-Disease-Detection)
- **Dataset Processing**: Managed by Saadullah Shazad

## Next Steps
1. **Train ResNet50-based model** for classification.
2. **Implement explainability features** using saliency maps.
3. **Deploy API for real-time diagnostics**.

## References
1. He, K., et al. *Deep Residual Learning for Image Recognition.* CVPR, 2016. [Paper](https://arxiv.org/abs/1512.03385)
2. Selvaraju, R.R., et al. *Grad-CAM: Visual Explanations from Deep Networks.* ICCV, 2017. [Paper](https://arxiv.org/abs/1610.02391)
3. APTOS 2019 Blindness Detection. [Kaggle](https://www.kaggle.com/c/aptos2019-blindness-detection)

---
This project serves as an **augmented intelligence tool**, supporting clinicians with **fast, reliable, and transparent AI-based retinal disease detection**.
