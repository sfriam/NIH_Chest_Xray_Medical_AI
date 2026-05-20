# NIH_Chest_Xray_Medical_AI

# Project Overview

This project focuses on building a deep learning pipeline for multi-label chest X-ray disease classification using the NIH Chest X-ray Dataset. The goal was to explore how convolutional neural networks can identify multiple thoracic diseases from grayscale chest radiographs while addressing real-world medical AI challenges, such as:

class imbalance,
patient-level data leakage,
threshold tradeoffs,
explainability in healthcare AI

The project started with a binary disease-vs-no-disease baseline and later evolved into a more realistic multi-label classification system capable of predicting multiple thoracic conditions simultaneously.

# Dataset

This project uses a sampled version of the NIH Chest X-ray Dataset containing:

5,606 chest X-ray images,
14 thoracic disease labels,
Patient metadata,
AP/PA view positions,
Disease Labels,
Atelectasis,
Cardiomegaly,
Consolidation,
Edema,
Effusion,
Emphysema,
Fibrosis,
Hernia,
Infiltration,
Mass,
Nodule,
Pleural Thickening,
Pneumonia,
Pneumothorax

Dataset Source -- NIH Chest X-ray Dataset on Kaggle

The dataset is not included in this repository due to size limitations.

# Objectives

The project was designed to:

Build a baseline binary chest X-ray classifier,
Transition from binary classification to multi-label prediction,
Evaluate disease-specific performance using ROC-AUC,
Handle severe class imbalance using weighted BCE loss,
Visualize model attention using Grad-CAM and to 
Understand limitations of medical imaging datasets

# Technologies & Libraries Used
Deep Learning,
PyTorch,
Torchvision,
Data Processing,
NumPy,
Pandas,
PIL,
Visualization,
Matplotlib,
OpenCV,
Machine Learning Utilities,
scikit-learn

# Model Architecture
DenseNet121 (transfer learning),
Methodology
# 1. Exploratory Data Analysis (EDA)

Performed imaging-specific EDA including:

label distribution analysis,
disease prevalence analysis,
patient-level inspection,
visualization of chest X-rays,
class imbalance analysis
# 2. Patient-Level Train/Validation Split

To prevent data leakage, the dataset was split using:

GroupShuffleSplit , 
grouped by Patient ID

This ensured that images from the same patient never appeared in both training and validation sets.

# 3. Binary Classification Baseline

The project initially started as a binary classification task:

Task,
Healthy vs Diseased,
Model,
DenseNet121 pretrained on ImageNet,
Evaluation,
ROC Curve,
AUC Score,
Threshold analysis,
Sensitivity vs Specificity tradeoffs,
Confusion Matrix

This phase helped establish:

baseline performance,
clinical tradeoff understanding,
model limitations
# 4. Multi-Label Classification

The project was later extended into a multi-label classification pipeline where each X-ray could contain multiple diseases simultaneously.

Label Engineering,
Converted "Finding Labels" into multi-hot encoded vectors,
Removed "No Finding" as a standalone disease class,
Represented healthy scans using all-zero vectors,
Output Representation

Each image produces a 14-dimensional prediction vector.

# 5. Handling Class Imbalance

The dataset exhibited severe imbalance, with diseases like Hernia having extremely few positive samples.

To address this:

calculated disease-specific positive weights,
implemented weighted BCEWithLogitsLoss

This improved:

rare disease learning,
Macro AUC performance,
balanced disease representation

Model Evaluation -- Metrics Used

ROC-AUC,
Macro AUC,
Micro AUC,
Per-disease AUC,
Threshold analysis,

Key Findings
---Stronger Disease Performance

The model achieved stronger performance on:

Emphysema,
Edema,
Effusion,
Cardiomegaly,

These diseases tend to have:

clearer structural abnormalities,
stronger radiographic patterns,
Challenging Diseases

Lower performance was observed for:

Fibrosis,
Nodule,
Infiltration,

Likely reasons:

label noise,
subtle radiographic patterns,
severe imbalance,
visual ambiguity

### Grad-CAM Explainability

Grad-CAM was implemented to visualize:

which regions of the chest X-ray influenced model predictions.

The model demonstrated anatomically meaningful attention patterns primarily focused on:

lung regions,
lower thoracic structures

This helped validate that the model was learning medically relevant spatial features rather than relying on image artifacts.

Results Summary

Baseline Multi-Label Model
Macro AUC: ~0.72,
Micro AUC: ~0.81

Weighted BCE Model

Macro AUC improved to ~0.75
Rare disease performance improved significantly

This demonstrated the impact of imbalance-aware learning in medical imaging tasks.

# Future Improvements

Potential future extensions include:

Focal Loss implementation,
Cross-validation,
Advanced augmentation techniques,
Vision Transformers (ViTs),
Deployment using Streamlit or FastAPI,
Clinical report generation, and 
External dataset validation



# How to Run
1. Clone the Repository
git clone <repository-link>

2. Install Dependencies
pip install -r requirements.txt

4. Download Dataset

Download the NIH Chest X-ray dataset from:

NIH Chest X-ray Dataset on Kaggle

Update dataset paths in the notebook accordingly.
