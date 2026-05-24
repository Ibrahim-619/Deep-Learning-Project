# Deep-Learning-Project
Deep Learning model to predict player churn risk in online games using TensorFlow &amp; Keras
# 🎮 Player Churn Risk Prediction — Deep Learning

A deep learning project that predicts whether an online game player 
is at risk of churning (quitting) based on their in-game behavior.

## 📊 Dataset
- 100,034 player records (40,034 original + 60,000 simulated)
- Source: Online Gaming Behavior Dataset

## 🧠 Model
- Deep Neural Network built with TensorFlow / Keras
- Architecture: 256 → 128 → 64 → 32 neurons
- Optimizer: Adam (learning rate = 0.001)
- Regularization: BatchNormalization + Dropout
- Accuracy: 86.24% | AUC-ROC: 0.885

## 🔍 Features Selected (via ANOVA F-test)
- Sessions Per Week
- Average Session Duration (minutes)
- Player Level
- Achievements Unlocked

## 🚀 How to Run
1. Install dependencies: pip install gradio tensorflow scikit-learn joblib numpy
2. Run the app: python churn_app.py
3. Open browser at: http://localhost:7860

## 🛠️ Tech Stack
Python | TensorFlow | Keras | Scikit-learn | Gradio | Pandas | NumPy
