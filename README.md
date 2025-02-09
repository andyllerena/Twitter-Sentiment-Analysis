# Twitter Sentiment Analysis using Machine Learning 🚀

## 📌 Project Overview
This project is a **Twitter Sentiment Analysis** system built using **Python, NLP, and machine learning**. It classifies tweets as **Positive, Neutral, or Negative** using **Logistic Regression and Support Vector Machines (SVM)**. The model was optimized using **GridSearchCV**, achieving an accuracy of **87.58%**.

The dataset consists of **recent tweets about the Pfizer & BioNTech COVID-19 vaccine**, collected using the **Tweepy Python package to access the Twitter API**. This dataset enables the study of public sentiment regarding vaccines through **Natural Language Processing (NLP) techniques**.

---

## 📌 Features
✅ **Preprocesses tweets**: Cleans text by removing stopwords, URLs, and punctuation.  
✅ **Vectorizes text**: Converts tweets into numerical features using **CountVectorizer & TF-IDF**.  
✅ **Trains Machine Learning Models**: Uses **Logistic Regression & SVM** for sentiment classification.  
✅ **Optimizes Hyperparameters**: Implements **GridSearchCV** to fine-tune **C, kernel, and gamma**.  
✅ **Evaluates Performance**: Uses **confusion matrices, precision-recall metrics, and F1-score**.  

---

## 📌 Tech Stack
- **Programming Language**: Python 🐍  
- **Libraries**: `Pandas`, `NumPy`, `Scikit-learn`, `NLTK`, `Matplotlib`, `Seaborn`, `WordCloud`, `Tweepy`  
- **Machine Learning Models**: Logistic Regression, SVM (Support Vector Machine)  
- **Text Processing**: CountVectorizer, TF-IDF  

---

## 📌 Data Collection & Inspiration
- The dataset consists of **recent tweets about the Pfizer & BioNTech COVID-19 vaccine**.  
- The data was collected using the **Tweepy Python package to access the Twitter API**.  
- The goal is to **analyze public sentiment** regarding the vaccine, study trending topics, and perform various **NLP tasks**.  

---

## 📌Model Performance
- **Best Model**: Support Vector Machine (SVM)
- **Final Accuracy**: 87.58%
- **Negative Sentiment Recall Improved**: 40% → 46%
- **Optimized using GridSearchCV**
