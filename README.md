# Retinal Disease Detection Using Deep Learning

## Team Members
- Tanish Roy
- Dev Shah
- Saadullah Shazad
- Inaam Azeezur-Rahman

## Project Overview
This project focuses on developing an **AI-driven retinal disease detection system**, leveraging deep learning techniques for **diabetic retinopathy (DR) classification**. The system analyzes **retinal images** and classifies them into **five severity levels** while providing **explainability through saliency maps**. The goal is to enhance **early detection and automated diagnosis** in ophthalmology.

## Problem Statement
Diabetic retinopathy (DR) is a leading cause of **vision impairment**, and early detection is critical for **timely treatment**. However, **manual diagnosis is time-consuming and requires expert ophthalmologists**. Our **deep learning-based solution** aims to automate **retinal image analysis**, making **diagnosis more accessible and efficient**.

## Dataset
- **APTOS 2019 Blindness Detection Dataset** ([Kaggle](https://www.kaggle.com/c/aptos2019-blindness-detection))
  - ~3,662 images categorized into **five severity levels** (0-4).
- **DRIVE Dataset** ([Grand Challenge](https://drive.grand-challenge.org/))
  - 40 images with annotated **ground truth for vessel segmentation**.

## Classification Categories
- **0** - No DR
- **1** - Mild DR
- **2** - Moderate DR
- **3** - Severe DR
- **4** - Proliferative DR

## System Architecture
The project consists of **image preprocessing, model training, inference, and visualization**:

1. **Preprocessing**
   - Image resizing to **224x224**.
   - **Normalization** for consistent model input.
   - **Data augmentation**: rotation, flips, contrast adjustments.

2. **Model Training**
   - **Modified ResNet50** with additional dense layers for classification.
   - **Optimizer**: Adam (learning rate = **0.0001**).
   - **Loss function**: Sparse categorical crossentropy.

3. **Model Explainability**
   - **Saliency maps** to highlight regions influencing predictions.
   - Helps interpret model decisions for **transparent AI diagnosis**.

4. **Deployment**
   - Flask-based **API** for real-time inference.
   - **Cloud deployment** via AWS Lambda & API Gateway.

## Team Responsibilities
| Team Member         | Responsibility |
|---------------------|---------------|
| **Tanish Roy**     | Research model architectures, implement ResNet50-based classifier |
| **Dev Shah**       | Conduct training, tune hyperparameters, and evaluate model performance |
| **Saadullah Shazad** | Dataset preprocessing, image augmentation, and evaluation metrics |
| **Inaam Azeezur-Rahman** | Develop visualizations, generate saliency maps, and deploy Flask API |

## Repository Setup
- **GitHub Repository**: [Retinal-Disease-Detection](https://github.com/Saadullah-2004/Retinal-Disease-Detection)
- **Dataset Management**: Preprocessed by Saadullah Shazad

## Next Steps
1. Train **ResNet50-based model** for DR classification.
2. Implement **saliency maps** for explainability.
3. Deploy **Flask-based API** for real-time diagnosis.

## References
1. He, K., et al. *Deep Residual Learning for Image Recognition.* CVPR, 2016. [Paper](https://arxiv.org/abs/1512.03385)
2. Selvaraju, R.R., et al. *Grad-CAM: Visual Explanations from Deep Networks.* ICCV, 2017. [Paper](https://arxiv.org/abs/1610.02391)
3. APTOS 2019 Blindness Detection. [Kaggle](https://www.kaggle.com/c/aptos2019-blindness-detection)

This project aims to bridge the gap between AI and **medical diagnostics**, providing a transparent and efficient solution for **retinal disease detection**.
