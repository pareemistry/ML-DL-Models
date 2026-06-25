# ML-DL-Models

# Machine Learning Models Implementation

## Text Dataset
Symptom2Disease Dataset

Problem Type:
Multi-Class Disease Classification (24 Disease Categories)

Models:
1. Naive Bayes - 92.08%
2. Support Vector Machine (SVM) - 95.83%
3. Decision Tree - 68.75%

## Image Dataset
Melanoma Skin Cancer Dataset of 10000 Images

Problem Type:
Binary Image Classification (Benign vs Malignant)

Models:
4. K-Nearest Neighbors (KNN) - 83.17%
5. Random Forest - 89.17%

## Best Models

### Text Classification
Support Vector Machine (SVM) - 95.83%

### Image Classification
Random Forest - 89.17%

## Data Preprocessing

### Text Dataset
- Text Cleaning
- TF-IDF Vectorization
- Train-Test Split (80:20)

### Image Dataset
- Image Loading using OpenCV
- Grayscale Conversion
- Image Resizing (64 × 64)
- Feature Extraction using Pixel Flattening
- Train-Test Split

## Evaluation Metrics
- Accuracy Score
- Precision
- Recall
- F1-Score
- Classification Report

## Tools & Libraries Used
- Python
- Pandas
- NumPy
- Scikit-Learn
- OpenCV
- Matplotlib
- Kaggle Notebook

## Summary

This project demonstrates the implementation of five Machine Learning algorithms on healthcare datasets involving both text and image classification tasks.

For text classification, the Symptom2Disease dataset was used to predict diseases from patient symptom descriptions. TF-IDF vectorization was applied to convert textual data into numerical features, and three machine learning models were trained and evaluated.

For image classification, the Melanoma Skin Cancer Dataset of 10000 Images was used to classify skin lesions as benign or malignant. Images were preprocessed through grayscale conversion, resizing, and feature extraction before training machine learning models.

Among all implemented models, Support Vector Machine (SVM) achieved the highest accuracy for text classification (95.83%), while Random Forest achieved the highest accuracy for image classification (89.17%).

# Deep Learning Models Implementation

## Image Dataset
Fashion-MNIST

Models:
1. ANN - 86.95%
2. Deep ANN - 86.89%
3. CNN - 91.10%

## Text Dataset
IMDB Movie Reviews

Models:
4. Dense Neural Network - 89.02%
5. LSTM - 88.41%

## Best Models
CNN (Image Classification) - 91.10%
Dense Neural Network (Text Classification) - 89.02%

Tools Used:
- Python
- TensorFlow/Keras
- Kaggle Notebook

  # Vision Models Implementation

## Dataset
**Intel Image Classification Dataset**

Classes:
- Buildings
- Forest
- Glacier
- Mountain
- Sea
- Street

## Models Implemented

1. ConvNeXt
2. Vision Transformer (ViT)
3. Swin Transformer
4. EVA
5. DINOv2
6. MAE (Masked Autoencoder)

## Evaluation Metrics

- Accuracy
- Precision
- Recall
- F1 Score
- Confusion Matrix
- Training Loss Curve

## Environment

- Python
- PyTorch
- timm
- Hugging Face Transformers
- Scikit-learn
- Matplotlib
- Seaborn
- Kaggle (Tesla T4 GPU)

## Repository Structure

```
├── ConvNeXt.ipynb
├── ViT.ipynb
├── Swin_Transformer.ipynb
├── EVA.ipynb
├── DINOv2.ipynb
├── MAE.ipynb
└── README.md
```

## Objective

Implement and compare state-of-the-art vision models on the Intel Image Classification dataset using transfer learning and evaluate their performance using standard classification metrics.
