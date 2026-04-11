# 📩 Spam Message Classifier using Machine Learning

This project is a spam detection system built using **Python, Scikit-learn, and Streamlit**. It classifies SMS or email messages as **Spam (1)** or **Ham (0)** using machine learning techniques.

## 🚀 Project Overview

* Reads dataset from `combined_dataset.csv`
* Normalizes labels into binary format (ham = 0, spam = 1)
* Removes invalid or missing data for better model performance
* Cleans text data (lowercasing, removing special characters, trimming spaces)
* Converts text into numerical features using **TF-IDF Vectorizer**

## ⚙️ Model Workflow

* Splits dataset into training and testing sets using `train_test_split`
* Trains a **Random Forest Classifier** for prediction
* Evaluates performance using accuracy and classification metrics

## 🌐 Streamlit Integration

* Provides a simple UI to input messages
* Predicts whether the message is spam or not in real time

## 📊 Technologies Used

* Python 🐍
* Pandas, NumPy
* Scikit-learn
* Matplotlib, Seaborn
* Streamlit

## 📁 Project Structure

```
spam-classifier/
│
├── app.py              # Streamlit app
├── model.py            # ML model logic
├── combined_dataset.csv
├── requirements.txt
└── README.md
```

## 🎯 Key Features

* End-to-end ML pipeline (data → preprocessing → training → prediction)
* Text classification using NLP techniques
* Interactive web interface using Streamlit

## 🔮 Future Improvements

* Hyperparameter tuning
* Try other models (Naive Bayes, Logistic Regression)
* Deploy on cloud (Render / Streamlit Cloud)
* Improve preprocessing with stopword removal

---

💡 This project demonstrates practical application of NLP and machine learning for real-world spam detection.
