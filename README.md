# Handwritten Digit Recognition Using Random Forest

## Project Overview

This project focuses on recognizing handwritten digits using Machine Learning classification algorithms.

The project uses the Scikit-learn Digits dataset, which contains handwritten digit images represented as numerical features.

Two classification algorithms are implemented and compared:

- Random Forest Classifier
- Decision Tree Classifier

The main objective is to evaluate both algorithms and identify the better-performing model using different classification metrics.

## Dataset Information

The project uses the built-in Digits dataset from Scikit-learn.

- Total Samples: 1,797
- Number of Features: 64
- Number of Classes: 10
- Digit Classes: 0 to 9
- Training Samples: 1,437
- Testing Samples: 360

## Machine Learning Algorithms

### 1. Random Forest Classifier

Random Forest is an ensemble learning algorithm that combines multiple decision trees to improve prediction performance and reduce overfitting.

### 2. Decision Tree Classifier

Decision Tree is a supervised learning algorithm that makes predictions using a tree-like structure of decisions and conditions.

## Technologies Used

- Python
- NumPy
- Pandas
- Scikit-learn
- Matplotlib
- Seaborn
- Joblib

## Model Evaluation

The models are evaluated using the following performance metrics:

- Accuracy
- Precision
- Recall
- F1 Score
- Classification Report
- Confusion Matrix

## Performance Results

| Metric | Random Forest | Decision Tree |
|---|---:|---:|
| Accuracy | 96.11% | 82.50% |
| Precision | 96.20% | 82.41% |
| Recall | 96.11% | 82.50% |
| F1 Score | 96.09% | 82.37% |

**Best Performing Model:** Random Forest Classifier

**Best Accuracy:** 96.11%

## Project Structure

```text
Random-Forest-Digit-Recognition/
│
├── main.py
├── requirements.txt
├── README.md
│
├── src/
│   ├── data_loader.py
│   ├── model_training.py
│   ├── evaluation.py
│   ├── visualization.py
│   └── utils.py
│
└── outputs/
    ├── graphs/
    ├── models/
    └── reports/
```

## How to Run the Project

### Step 1: Clone the Repository

```bash
git clone YOUR_GITHUB_REPOSITORY_URL
```

### Step 2: Navigate to the Project Folder

```bash
cd Random-Forest-Digit-Recognition
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 4: Execute the Project

```bash
python main.py
```

## Generated Outputs

The project automatically generates:

- Trained Random Forest model
- Trained Decision Tree model
- Model comparison graphs
- Confusion matrix visualizations
- Classification reports
- Model performance comparison CSV

## Conclusion

The experimental results show that the Random Forest Classifier performs better than the Decision Tree Classifier on the selected test dataset.

Random Forest achieved an accuracy of 96.11%, making it the best-performing model in this comparison.