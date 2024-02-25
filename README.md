# Credit-Card-Fraud-Detection-with-Tensorflow

This repository is dedicated to the development of a Credit Card Fraud Detection system using TensorFlow, one of the most popular deep learning frameworks. Credit card fraud is a serious concern for both financial institutions and consumers, and machine learning techniques can play a crucial role in identifying fraudulent transactions in real-time.

## Table of Contents

- [Introduction](#introduction)
- [Dataset](#dataset)
- [Data Preprocessing](#data-preprocessing)
- [Model Development](#model-development)
- [Model Evaluation](#model-evaluation)
- [Best Model Selection](#best-model-selection)
- [Threshold Selection](#threshold-selection)
- [Results](#results)
- [Future Work](#future-work)
- [Dependencies](#dependencies)
- [Usage](#usage)

## Introduction

Credit card fraud is a significant problem for financial institutions and cardholders worldwide. Traditional rule-based fraud detection systems often struggle to keep pace with evolving fraud tactics. Machine learning, particularly neural networks, offers a promising approach to detecting fraudulent transactions based on patterns in transaction data.

This project explores the development and evaluation of several neural network models for credit card fraud detection. Various architectures and training strategies are considered to optimize model performance and generalization ability.

## Dataset

The dataset used in this project contains credit card transactions with the following features:

- `id`: Unique identifier for each transaction.
- `V1` to `V28`: Anonymized features representing various transaction attributes (e.g. time, location, etc.).
- `Amount`: Transaction amount.
- `Class`: Target variable indicating whether the transaction is fraudulent (1) or legitimate (0).

## Data Preprocessing

Data preprocessing steps include:

- Loading the dataset and inspecting its structure.
- Exploratory data analysis to understand data distributions and relationships.
- Handling missing values (if any) and outliers.
- Feature engineering and selection.
- Splitting the dataset into training, validation, and test sets.

## Model Development

Several neural network architectures are developed and trained using TensorFlow and Keras. Model architectures include fully connected feedforward networks with various hidden layers, activations, regularization techniques (e.g., dropout, L2 regularization), and batch normalization.

## Model Evaluation

Model performance is evaluated using standard metrics such as accuracy, precision, recall, and F1 score. Evaluation is performed on both the validation and test datasets to assess model generalization ability.

## Best Model Selection

The best-performing model is selected based on validation accuracy. The model with the highest validation accuracy is chosen for further analysis and deployment.

## Threshold Selection

A dynamic thresholding approach is employed to determine the classification threshold for fraud detection. Threshold selection methods include static thresholds and quantile-based thresholds derived from the model's predicted probabilities.

## Results

The selected model's performance on the test dataset is reported, including accuracy, precision, recall, and F1 score. Additional visualizations may be provided to illustrate model performance and decision thresholds.

## Future Work

Future enhancements to the project may include:

- Fine-tuning model hyperparameters to improve performance further.
- Exploring additional feature engineering techniques or alternative neural network architectures.
- Investigating advanced anomaly detection algorithms for credit card fraud detection.
- Deployment of the model in real-world applications and integration with existing fraud detection systems.

## Dependencies

- Python 3.x
- TensorFlow
- Keras
- NumPy
- Pandas
- Matplotlib
- Seaborn
- Scikit-learn

## Usage

1. Clone the repository:
   ```bash
   git clone https://github.com/tapiwachinyerere/credit-card-fraud-detection.git

2. Install dependencies:
   ```bash
   pip install -r requirements.txt

3. Run the Jupyter notebook to reproduce the analysis and results.
