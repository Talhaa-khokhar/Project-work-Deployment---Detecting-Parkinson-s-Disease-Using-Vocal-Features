# Project work & Deployment - Detecting Parkinson's Disease Using Vocal Features

## Overview

This Python-based system detects Parkinson's disease using voice analysis. The system consists of the following components:

### Data Preprocessing
- **Feature Extraction:** Extract vocal features like pitch, frequency, and formants from a dataset of voice samples.
- **Data Scaling:** Use StandardScaler to standardize features by removing the mean and scaling to unit variance.
- **Data Balancing:** Apply SMOTE (Synthetic Minority Over-sampling Technique) to address class imbalance issues.

### Machine Learning Models
- **Random Forest**
- **Gradient Boosting**
- **Support Vector Classifier (SVC)**

### Model Evaluation
Evaluate model performance using:
- Accuracy
- F1 Score
- Recall
- Precision
- ROC AUC

This system has the potential to revolutionize the early diagnosis and management of Parkinson's disease. The code is written in Python and utilizes popular libraries like scikit-learn, imbalanced-learn, and joblib.

## Deployment

The system is deployed using Flask, a lightweight Python web framework, and Docker, a containerization platform. This setup allows for easy deployment and scaling of the application.

### Flask API
A RESTful API built using Flask provides a programmatic interface for interacting with the Parkinson's disease detection system.

### Docker Containerization
The Flask API is containerized using Docker, ensuring the application is isolated from the host system and can be easily deployed and scaled.

### Docker Compose
Docker Compose is used to define and run multi-container Docker applications, simplifying the management of the application's dependencies and services.

Deployment using Flask and Docker offers a scalable and reliable solution for deploying the Parkinson's disease detection system.
