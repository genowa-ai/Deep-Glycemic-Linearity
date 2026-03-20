# Deep-Glycemic-Linearity
High-precision Diabetes dataset analysis via a Weight-Optimized Deep Neural Network.

# Diabetes Risk Prediction: Dense Neural Network (DNN) 

This repository contains a high-performance predictive model for diabetes risk assessment. The project focuses on utilizing **Linear Weights** within a Deep Neural Network to classify patient health data with high precision.

## Project Overview
The goal was to build a "Specialist" model capable of identifying diabetes risk based on 7 key physiological and behavioral features. This serves as the numeric baseline before integrating Large Language Model (LLM) reasoning with **Gemma-2B**.

### Key Achievements:
* **Final Accuracy:** 88.26%
* **Weighted F1-Score:** 0.88
* **Data Size:** 60,000 Training samples | 10,000 Test samples

---

## 🛠️ Technical Implementation

### 1. Feature Engineering (The "Pressure" Challenge)
A primary hurdle was the raw `pressure` string (e.g., "120/80"). 
* **Solution:** Implemented a robust split-and-cast pipeline to create independent `systolic` and `diastolic` float features.
* **Integrity:** Used training-set medians to fill missing values in the test set, preventing **Data Leakage**.

### 2. Ordinal Mapping (0-1-2 Strategy)
To increase model "sharpness" over standard binary flags, categorical features (`cholesterol`, `gluc`) were mapped to a 3-step risk scale:
* `low` → 0
* `medium` → 1
* `high` → 2
This allows the **Linear Weights** to learn a mathematical progression of risk.

### 3. Model Architecture
* **Framework:** TensorFlow/Keras
* **Layers:** Dense layers with Dropout for regularization.
* **Scaling:** `StandardScaler` fitted strictly on training data.

---

## Performance Analysis

| Metric | Score |
| :--- | :--- |
| **Accuracy** | 88.26% |
| **Precision (Diabetes)** | 0.86 |
| **Recall (Diabetes)** | 0.65 |

**Observation:** The model is exceptionally strong at identifying healthy patients (96% recall). The next phase of this project involves using **Gemma-2B** to investigate the "False Negatives"—the 900 cases where the numeric model was conservative but a medical LLM might find hidden risk patterns.

---

## 📂 Repository Structure
* `diabetes_analysis.ipynb`: The core Kaggle-based training pipeline.
* `diabetes_predictions_final.csv`: The final 10,000-row prediction output.
* `requirements.txt`: Python environment dependencies.

## 🔗 Project Links
Live Interactive Notebook: Kaggle: https://www.kaggle.com/code/eugenewata/deep-glycemic-linearity

Final Prediction Report: Download diabetes_predictions_final.csv

GitHub Repository: https://github.com/genowa-ai/Deep-Glycemic-Linearity

## 🔜 Future Work: Phase 2
* Integration of **Gemma-2B** for clinical context reasoning.
* Hybrid "Ensemble" approach: Combining Linear Weights with Transformer-based Attention.
