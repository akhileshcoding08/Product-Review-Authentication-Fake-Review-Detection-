# 🛒 Product Review Authentication - Fake Review Detection

A machine learning project that detects whether an e-commerce product review is **genuine (OR)** or **computer-generated / fake (CG)** using Natural Language Processing and a Multinomial Naive Bayes classifier.

---

## 📌 Overview

Fake reviews mislead customers and distort a product's real reputation. This project builds a **text classification pipeline** that reads a product review and predicts whether it was written by a real customer or auto-generated. It covers the full workflow — data cleaning, exploratory data analysis (EDA), feature engineering, model training, hyperparameter tuning, and performance evaluation.

## 📂 Dataset

- **File:** `fake_reviews.csv`
- **Rows:** 40,432 reviews (40,420 after removing duplicates)
- **Columns:**
  | Column | Description |
  |---|---|
  | `category` | Product category (e.g., Home & Kitchen, Clothing, Shoes & Jewelry) |
  | `rating` | Star rating given by the reviewer (1-5) |
  | `label` | `OR` = Original/Genuine review, `CG` = Computer-Generated/Fake review |
  | `text_` | The raw review text |

## 🎯 Objective

Train a supervised text-classification model that can flag a review as **Fake (CG)** or **Genuine (OR)** based purely on its text content, and compare baseline vs. tuned model performance.

## 🧰 Tech Stack & Libraries

- **Language:** Python 3
- **Data handling:** `pandas`, `numpy`
- **Visualization:** `matplotlib`, `seaborn`, `wordcloud`
- **NLP / ML:** `scikit-learn` (`CountVectorizer`, `MultinomialNB`, `GridSearchCV`, `RandomizedSearchCV`)
- **Environment:** Jupyter Notebook

## 🔄 Project Workflow

1. **Data Loading & Cleaning** : Load `fake_reviews.csv`, inspect structure, remove duplicate rows, check for missing values.
2. **Exploratory Data Analysis (EDA)** : Label distribution, rating distribution, category distribution, review length/word count analysis, word clouds, top frequent words, correlation heatmaps.
3. **Feature Engineering** : Convert review text into numeric vectors using `CountVectorizer` (Bag-of-Words), and encode labels (`OR → 0`, `CG → 1`).
4. **Train/Test Split** : 80/20 split with `random_state=42`.
5. **Baseline Model** : Train a `MultinomialNB` classifier on the vectorized text.
6. **Model Evaluation** : Accuracy, precision/recall/F1 report, confusion matrix, ROC curve, precision-recall curve, 5-fold cross-validation.
7. **Hyperparameter Tuning** : Optimize the Naive Bayes smoothing parameter (`alpha`) using both `GridSearchCV` and `RandomizedSearchCV`.
8. **Model Comparison** : Compare baseline vs. tuned models using accuracy and residual (probability error) analysis.

## 📊 Results

| Model | Accuracy |
|---|---|
| Baseline Multinomial NB | **83.03%** |
| GridSearchCV (best α = 0.1) | **83.83%** |
| RandomizedSearchCV (best α ≈ 0.207) | **83.84%** |

Hyperparameter tuning gave a modest but consistent accuracy improvement over the baseline model, with both search strategies converging to a similar optimal `alpha`.

## 🗂️ Project Structure

```
product-review-authentication/
│
├── Product_Review_Authentication_NB.ipynb   # Main Jupyter notebook (EDA + ML pipeline)
├── fake_reviews.csv                         # Dataset (not included — add your own copy)
├── README.md                                # Project documentation
└── requirements.txt                         # Python dependencies
```

## ⚙️ Installation & Setup

```bash
# 1. Clone the repository
git clone https://github.com/<your-username>/product-review-authentication.git
cd product-review-authentication

# 2. (Optional) Create a virtual environment
python -m venv venv
source venv/bin/activate      # On Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Launch Jupyter Notebook
jupyter notebook Product_Review_Authentication_NB.ipynb
```

### requirements.txt
```
numpy
pandas
matplotlib
seaborn
wordcloud
scikit-learn
scipy
jupyter
```

## ▶️ Usage

1. Place `fake_reviews.csv` in the project root directory.
2. Open `Product_Review_Authentication_NB.ipynb` in Jupyter Notebook / JupyterLab.
3. Run all cells sequentially (`Kernel → Restart & Run All`).
4. Review the EDA plots, model metrics, and the final model comparison table.

## 🚀 Future Improvements

- Use TF-IDF vectorization and compare against Bag-of-Words.
- Try other classifiers (Logistic Regression, SVM, Random Forest, or transformer-based models like BERT).
- Perform text pre-processing such as lemmatization, stopword removal, and n-gram features.
- Deploy the model as a REST API or a simple web app for real-time review checking.
  
