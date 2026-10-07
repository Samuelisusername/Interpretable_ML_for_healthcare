# Interpretable_ML_for_healthcare
Project 2 of ETH course ML for healthcare


Q1: Predicting Heart disease based on Age sex, colesterol levels and other factors with a heavy focus of interpretability and explainability (XAI). 
Architectures used towards a interpretable prediction with reliable explainability: 
* Lasso Logistic Regression with L1 regularization
* 4 Layered MLP with dropout
* Neural Additive Model $$g(E[Y]) = \beta + \sum_{i=1}^d f_i(x_i)$$

Q2: Predicting Pneumonia from Chest X-Rays using CNNs

* **Model:** Binary CNN classifier trained on 384×384 grayscale chest X-rays.
* **Explainability:** Applied **Grad-CAM** post-hoc analysis to generate heatmaps showing which pixel regions drove predictions.
* **Reliability Check:** Tested model integrity by training on **randomly shuffled labels** to ensure genuine feature learning over dataset noise.
* **Metrics:** Evaluated performance via Accuracy, Precision, Recall, F1 Score, AUROC, and AUPRC.

Verification of reliability of models

AI Usage declaration: 
Tool used: Gemini 3.1 Pro
Files affected: all
Purpose: Initial Brainstorming, Code cleanup and comments generation for readability
