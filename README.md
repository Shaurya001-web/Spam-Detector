# 🌳 Decision Tree Iris Classification

This project demonstrates the implementation of a **Decision Tree Classifier** using the famous Iris dataset. It focuses on understanding how decision trees work and how controlling model complexity helps prevent overfitting.

## 📌 Project Overview

* Load Iris dataset from `sklearn`
* Split data into training and testing sets
* Train a Decision Tree model
* Evaluate performance using accuracy
* Visualize the decision tree

## ⚙️ Technologies Used

* Python 🐍
* Pandas
* Scikit-learn
* Matplotlib

## 🧠 How It Works

### 1. Data Loading

```python
from sklearn.datasets import load_iris
```

The dataset contains 150 samples with features like sepal and petal dimensions.

### 2. Train-Test Split

```python
from sklearn.model_selection import train_test_split
```

* 80% training
* 20% testing

### 3. Model Training

```python
from sklearn.tree import DecisionTreeClassifier

model = DecisionTreeClassifier(max_depth=2)
model.fit(X_train, y_train)
```

### 🌳 Why `max_depth = 2`?

#### 🔴 Overfitting Problem

* Too many splits
* Memorizes training data
* Poor performance on new data

#### 🟢 Solution

* Limits tree growth
* Focuses on important patterns
* Improves generalization

## 📊 Result

* Accuracy: ~98%

## 🌿 Visualization

```python
from sklearn.tree import plot_tree
```

Helps understand decision paths and feature importance.

## 📁 Project Structure

```
decision-tree-iris-classification/
│
├── decision_tree_iris.ipynb
├── model.py
├── requirements.txt
└── README.md
```

## 🚀 How to Run

```bash
git clone https://github.com/your-username/decision-tree-iris-classification.git
pip install -r requirements.txt
```

## 🎯 Key Learnings

* Decision Tree fundamentals
* Overfitting vs Generalization
* Importance of `max_depth`
* Model evaluation

## 📌 Future Improvements

* Compare different `max_depth` values
* Add confusion matrix
* Try advanced models
