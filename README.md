# Audio File Classification using ML

## Overview
**Audio File Classification using ML** is a machine learning-driven project that classifies music genres and offers personalized playlist recommendations. Using the GTZAN dataset—comprising 1000 audio clips across 10 genres such as Blues, Hip-hop, Classical, and Rock—the system leverages feature extraction techniques like Discrete Fourier Transform (DFT) and applies a variety of algorithms, including CNN, DNN, XGBoost, and SVM. Among these, the Convolutional Neural Network (CNN) achieved the highest accuracy of **92.93%**, making it the most effective model for music genre classification. The project also evaluates model performance using key metrics such as accuracy, precision, recall, and F1-score to ensure robust and reliable predictions.

Beyond classification, the system is designed to generate personalized playlist recommendations based on user preferences, offering practical applications for music streaming services. Future enhancements include supporting sub-genre classification, incorporating multi-modal data such as lyrics or metadata, enabling real-time predictions for mobile or streaming platforms, and ensuring ethical considerations like fairness and transparency in recommendations. This project demonstrates the power and versatility of machine learning in audio analysis and user-centric music experiences.

## Features
- 🎵 Music Genre Classification (10 genres)
- 🧠 Model Comparison: CNN, DNN, XGBoost, SVM, Random Forest
- 🎯 High Accuracy: 92.93% with CNN
- 🔊 Feature Extraction: Using DFT and signal processing
- 📊 Evaluation Metrics: Accuracy, F1-score, Precision, Recall
- 🤖 Personalized Playlist Recommendations

## Dataset
**GTZAN Genre Collection**
- 1000 audio clips (30 seconds each, `.wav` format)
- 10 genres: Blues, Classical, Country, Disco, Hip-hop, Jazz, Metal, Pop, Reggae, Rock
- 100 clips per genre
- Publicly available on [Kaggle](https://www.kaggle.com/datasets)

## Model Performance

| Model                             | F1-Score | Accuracy |
|----------------------------------|----------|----------|
| XGBoost                          | 0.740    | 0.745    |
| Stochastic Gradient Descent (SGD)| 0.636    | 0.660    |
| MLP Classifier                   | 0.857    | 0.858    |
| Support Vector Machine (SVM)     | 0.758    | 0.762    |
| Random Forest                    | 0.811    | 0.814    |
| LightGBM                         | 0.903    | 0.903    |
| Deep Neural Network (DNN)        | 0.923    | 0.933    |
| **Convolutional Neural Network (CNN)** | —      | **92.93%** |

## Future Scope
- 🔍 Fine-grained (sub-genre) classification
- 🧾 Multi-modal learning (e.g., lyrics, metadata)
- ⚡ Real-time genre classification for mobile/streaming
- 🧠 Ethical AI: fairness, transparency, and privacy

## Getting Started

### Prerequisites
- Python 3.x
- Install dependencies from `requirements.txt`:
  ```bash
  pip install -r requirements.txt
