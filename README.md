# Iris Classification Project

## Overview
This project classifies the Iris flower dataset using two approaches:
1. **Support Vector Machine (SVM)** with `scikit-learn`
2. **Neural Network** with `TensorFlow` Keras API.

The goal is to predict Iris species (Setosa, Versicolor, Virginica) based on sepal and petal dimensions.

## Dataset
- **Samples:** 150
- **Features:** Sepal and petal length & width
- **Classes:** 0 (Setosa), 1 (Versicolor), 2 (Virginica)

## Prerequisites
Install required libraries:
```bash
pip install numpy scikit-learn tensorflow
```

## Workflow
1. **Data Preparation:** Split dataset (80:20) and standardize features.
2. **Model Training:**
   - SVM: RBF kernel, `C=1`, gamma=`scale`.
   - Neural Network: Input (4), hidden layers (64, ReLU), output (3, softmax). Adam optimizer, sparse categorical crossentropy, 50 epochs.
3. **Evaluation:** Accuracy scores for both models.

## Results
- **SVM Accuracy:** Calculated using `accuracy_score`.
- **Neural Network Accuracy:** Achieved after 50 epochs.

