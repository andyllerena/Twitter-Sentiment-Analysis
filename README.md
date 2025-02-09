Twitter Sentiment Analysis using Machine Learning 🚀
📌 Project Overview
This project is a Twitter Sentiment Analysis system built using Python, NLP, and machine learning. It classifies tweets as Positive, Neutral, or Negative using Logistic Regression and Support Vector Machines (SVM). The model was optimized using GridSearchCV, achieving an accuracy of 87.58%.

📌 Features
✅ Preprocesses tweets: Cleans text by removing stopwords, URLs, and punctuation.
✅ Vectorizes text: Converts tweets into numerical features using CountVectorizer & TF-IDF.
✅ Trains Machine Learning Models: Uses Logistic Regression & SVM for sentiment classification.
✅ Optimizes Hyperparameters: Implements GridSearchCV to fine-tune C, kernel, and gamma.
✅ Evaluates Performance: Uses confusion matrices, precision-recall metrics, and F1-score.

📌 Tech Stack
Programming Language: Python 🐍
Libraries: Pandas, NumPy, Scikit-learn, NLTK, Matplotlib, Seaborn, WordCloud
Machine Learning Models: Logistic Regression, SVM (Support Vector Machine)
Text Processing: CountVectorizer, TF-IDF
📌 Installation & Setup Instructions
1️⃣ Clone the Repository
bash
Copy
Edit
git clone https://github.com/andyllerena/Twitter-Sentiment-Analysis.git
cd Twitter-Sentiment-Analysis
2️⃣ Install Required Dependencies
Ensure you have Python 3.x installed, then run:

bash
Copy
Edit
pip install -r requirements.txt
3️⃣ Download NLTK Stopwords (First-time setup)
python
Copy
Edit
import nltk
nltk.download('stopwords')
nltk.download('punkt')
4️⃣ Run the Sentiment Analysis
To execute the script:

bash
Copy
Edit
python sentiment_analysis.py
📌 How to Use
Place a dataset of tweets (.csv format) in the project directory.
Run sentiment_analysis.py to classify tweets.
View sentiment distributions via confusion matrices & visualizations.
📌 Model Performance
Best Model: Support Vector Machine (SVM)
Final Accuracy: 87.58%
Negative Sentiment Recall Improved: 40% → 46%
Optimized using GridSearchCV
📌 Future Improvements
🚀 Implement TF-IDF instead of CountVectorizer to improve word representation.
🚀 Balance dataset to reduce Neutral tweet dominance.
🚀 Experiment with Deep Learning models (LSTMs, Transformers).

📌 Contributing
Pull requests are welcome! If you'd like to contribute, fork the repository, create a new branch, and submit a pull request (PR).

📌 License
This project is licensed under the MIT License.
