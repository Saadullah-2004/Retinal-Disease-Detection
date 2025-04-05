# AI-Driven Retinal Disease Detection Tool

## Team Members
- **Dev Shah**
- **Tanish Roy**
- **Inaam Azeezur-Rahman**
- **Saadullah Shahzad**

## Overview
This project presents an end-to-end **AI-driven diagnostic assistant** specifically designed for **retinal disease detection**. Our tool automates the analysis of retinal images to support clinical decision-making by leveraging cutting-edge deep learning models along with robust explainability mechanisms. It addresses conditions such as **diabetic retinopathy (DR)** and **age-related macular degeneration (AMD)** by providing not only high diagnostic accuracy but also transparent insights into the decision process.

## Key Features

### 1️⃣ Automated Disease Detection
- **Vision Transformer (ViT):** Fine-tuned for robust classification of retinal abnormalities.
- **U-Net Segmentation:** Precisely delineates key retinal structures and pathologies using a U-Net architecture optimized with Cross-Entropy Loss.

### 2️⃣ Multi-Modal Integration
- **Dual-Model Pipeline:** Integrates a global feature extractor (ViT) with a local segmentation model (U-Net) to capture both contextual and fine-grained pathological details.
- **Modular Architecture:** Each module (classification, segmentation, and explainability) operates cohesively to deliver a comprehensive diagnostic outcome.

### 3️⃣ Explainability & Transparency
- **Grad-CAM++ Integration:** Generates heatmaps that highlight critical retinal regions influencing model predictions.
- **Clinically Interpretable Visualizations:** Supports clinicians with intuitive and reliable explanations of the AI decision process.

### 4️⃣ Automated Report Generation
- **LangChain Integration:** Aggregates outputs from all modules to automatically generate standardized PDF diagnostic reports.
- **Seamless EHR Integration:** Facilitates incorporation into existing clinical workflows for efficient patient management.

### 5️⃣ Efficiency & Scalability
- **High Accuracy:** The ViT model achieves a test accuracy of ~95% for classification, while the U-Net delivers precise segmentation (96% accuracy).
- **Optimized Performance:** Designed for real-time clinical applications with minimal computational overhead.

## Datasets

### Classification Dataset
- **APTOS 2019 Blindness Detection Dataset** ([Kaggle](https://www.kaggle.com/c/aptos2019-blindness-detection))  
  - Contains high-quality retinal fundus images with annotations for disease severity (e.g., diabetic retinopathy grades).

### Segmentation Dataset
- A specialized subset of the APTOS 2019 dataset with pixel-level annotations for:
  - **Background**
  - **Haemorrhages**
  - **Hard Exudates**
  - **Microaneurysms**

*Preprocessing steps include resizing, normalization, and augmentation to ensure consistency and robust performance across models.*

## System Architecture

1. **Preprocessing:**
   - Resize and normalize retinal images.
   - Tokenize images into patches for the Vision Transformer.
   - Apply data augmentation techniques to improve model generalizability.

2. **Classification (Vision Transformer):**
   - Decomposes images into patch tokens with positional encodings.
   - Uses multi-head self-attention to capture both local and global dependencies.
   - Outputs a probability distribution over retinal disease classes.

3. **Segmentation (U-Net):**
   - Employs an encoder-decoder structure with skip connections.
   - Performs pixel-level classification to segment retinal structures and lesions.
   - Optimized with Cross-Entropy Loss for precise boundary delineation.

4. **Explainability (Grad-CAM++):**
   - Computes class-specific heatmaps to visualize the model’s focus areas.
   - Enhances clinician trust through intuitive visual explanations.

5. **Report Generation:**
   - Uses LangChain to combine classification results, segmentation maps, and visual explanations into comprehensive PDF reports.
   - Designed for smooth integration with Electronic Health Records (EHRs).

## Team Responsibilities

| Team Member               | Responsibility                                                                                              |
|---------------------------|-------------------------------------------------------------------------------------------------------------|
| **Dev Shah**              | Developed the overall pipeline and implemented the Vision Transformer-based classification module.        |
| **Saadullah Shahzad**     | Designed and implemented the U-Net segmentation model and managed dataset preprocessing.                   |
| **Tanish Roy**            | Integrated and fine-tuned the Grad-CAM++ explainability component for visualizing key decision areas.       |
| **Inaam Azeezur-Rahman**  | Led the automated report generation module and ensured seamless clinical integration via LangChain.          |

## Repository Setup
- **GitHub Repository:** [Retinal-Disease-Detection](https://github.com/Saadullah-2004/Retinal-Disease-Detection)
- **Dataset Management:** Handled within the repository with scripts for preprocessing, augmentation, and evaluation.

## Next Steps
1. **Refinement of Model Training:** Further optimize the Vision Transformer and U-Net models with larger and more diverse datasets.
2. **Enhanced Explainability:** Explore additional interpretability techniques to complement Grad-CAM++.
3. **Clinical Integration:** Develop and deploy a real-time API to support live diagnostics and integrate with hospital EHR systems.
4. **Continuous Learning:** Implement adaptive learning mechanisms to update models with real-world clinical data.

## References
- Gulshan, V., et al. (2016). *Development and validation of a deep learning algorithm for detection of diabetic retinopathy in retinal fundus photographs.* JAMA, 316(22), 2402–2410.
- Dosovitskiy, A., et al. (2020). *An image is worth 16x16 words: Transformers for image recognition at scale.* arXiv preprint arXiv:2010.11929.
- Ronneberger, O., Fischer, P., & Brox, T. (2015). *U-Net: Convolutional networks for biomedical image segmentation.* MICCAI.
- Selvaraju, R. R., et al. (2017). *Grad-CAM: Visual explanations from deep networks via gradient-based localization.* International Journal of Computer Vision.
- Chattopadhay, A., et al. (2018). *Grad-CAM++: Improved visual explanations for deep convolutional networks.* IEEE Winter Conference on Applications of Computer Vision.
- LangChain. *LangChain: A Framework for LLM Applications.* Retrieved from [GitHub](https://github.com/hwchase17/langchain).
- APTOS 2019 Blindness Detection. Retrieved from [Kaggle](https://www.kaggle.com/c/aptos2019-blindness-detection).

---

This project is an **augmented intelligence tool** designed to support clinicians with fast, reliable, and transparent AI-based retinal disease detection, thereby enhancing early intervention and improving patient outcomes.
