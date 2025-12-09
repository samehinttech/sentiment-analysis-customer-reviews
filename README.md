# sentiment-analysis-customer-reviews

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)
![Python](https://img.shields.io/badge/python-3.12%2B-blue.svg)
[![developed with PyCharm](https://img.shields.io/badge/IDE-PyCharm-green?logo=pycharm&logoColor=white)](https://www.jetbrains.com/pycharm/)
![Jupyter](https://img.shields.io/badge/Notebook-Jupyter-orange.svg)
[![Last Commit](https://img.shields.io/github/last-commit/samehinttech/sentiment-analysis-customer-reviews?color=purple)](https://github.com/samehinttech/sentiment-analysis-customer-reviews/commits/main)

## Project Overview

This repository contains the deliverables for a group project completed by BIT students at the **FHNW University of Applied Sciences and Arts Northwestern Switzerland**.

The project focuses on BI and data analytics solution using a real-world customer feedback dataset. The primary goal is to apply data science and Natural Language Processing (NLP) techniques to extract actionable business insights.

### Stakeholders

- **Customer Service Manager** – Oversees support team; identifies common complaints and service issues
- **Product Manager** – Manages product quality; understands product feedback (quality, packaging, delivery)
- **Marketing Team** – Creates campaigns; finds positive testimonials and areas for improvement
- **Operations Manager** – Manages logistics; monitors delivery and packaging feedback

### Trigger

- **Weekly Review Meetings** – Customer service team reviews sentiment trends and identifies top complaints to prioritize
- **Quarterly Reports** – Management needs sentiment metrics and feature insights for business decisions

### Analytical Questions

1. What is the overall sentiment distribution of customer reviews?
2. Which product features (delivery, quality, packaging, service, price) receive positive vs negative feedback?
3. What are the most common topics/themes in negative reviews?
4. Which areas should we prioritize for improvement based on customer feedback?

### Expected Results

- **Sentiment Dashboard** – Visual overview of sentiment distribution across ratings and categories
- **Model Comparison** – Accuracy metrics for VADER, Naive Bayes, Logistic Regression, and BERT
- **Feature Analysis Report** – Sentiment scores per feature (delivery, quality, service, price, value)
- **Word Clouds** – Visual representation of common terms per sentiment class
- **Processed Dataset** – Exported CSV with predictions for further analysis

---

## Implementation

### Pipeline Overview

```
Raw Reviews → Text Preprocessing → Feature Extraction → Sentiment Classification → Feature Analysis → Export
```

### Notebook Structure

1. **Part 1** – Libraries Import
2. **Part 2** – Exploratory Data Analysis (EDA)
3. **Part 3** – Text Preprocessing (cleaning, tokenization, lemmatization)
4. **Part 4** – Feature Extraction (TF-IDF vectorization)
5. **Part 5** – Sentiment Classification Models (VADER, NB, LR, BERT)
6. **Part 6** – Topic Modeling (LDA) & Feature-Based Sentiment Analysis
7. **Part 7** – Export Processed Data
8. **Part 8** – Conclusion

---

## Technology Stack

### NLP & Text Processing

- **NLTK** – Tokenization, stopword removal, lemmatization
- **TF-IDF** – Feature extraction for ML models
- **WordCloud** – Vocabulary visualization

### Sentiment Analysis Models

- **VADER** – Rule-based baseline (79.64% accuracy)
- **Naive Bayes** – Classical ML (100% accuracy)
- **Logistic Regression** – Classical ML (100% accuracy)
- **BERT** – Transformer model (91.76% accuracy)

### Topic Modeling

- **LDA (Latent Dirichlet Allocation)** – Discover topics in reviews

### Libraries

- **pandas, numpy** – Data manipulation
- **matplotlib, seaborn** – Visualization
- **scikit-learn** – ML models, TF-IDF, evaluation
- **transformers, torch** – BERT model
- **vaderSentiment** – VADER baseline

---

## Quick Start

### Prerequisites

- Python 3.12+
- NVIDIA GPU (optional, for faster BERT inference)

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/samehinttech/sentiment-analysis-customer-reviews.git
   cd sentiment-analysis-customer-reviews
   ```

2. **Create virtual environment**
   ```bash
   python -m venv .venv
   ```

3. **Activate virtual environment**
   ```bash
   .\.venv\Scripts\Activate.ps1
   ```

4. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

5. **GPU Support (Optional)**
   ```bash
   # For NVIDIA RTX 30/40 series (CUDA 12.4)
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
   
   # For NVIDIA RTX 50 series (CUDA 13.0)
   pip install --pre torch torchvision --index-url https://download.pytorch.org/whl/nightly/cu130
   ```

6. **Run the notebook**
   ```bash
   jupyter notebook notebooks/sentiment_analysis.ipynb
   ```

> **Note:** BERT model downloads automatically on first run (~500MB)
> 
> **IMPORTANT NOTE** The notebook is designed to run from start to finish without interruptions.
 Please ensure all cells are executed in order for proper functionality.
> Sorry for that but you need to be patient as some steps (like BERT inference) may take time depending on your hardware.

---

## References

### Tutorials
- Acknowledgements to our great instructor for his support and guidance throughout the learning journey.
  (Greate teaching materials and tutorials provided by him were instrumental in completing this project successfully.)
- [TensorFlow: Basic Text Classification (Sentiment Analysis)](https://www.tensorflow.org/tutorials/keras/text_classification) 
- [TensorFlow: Classify Text with BERT](https://www.tensorflow.org/text/tutorials/classify_text_with_bert)
- [TensorFlow Hub: Text Classification with Movie Reviews](https://www.tensorflow.org/hub/tutorials/tf2_text_classification)
- [Hugging Face: Getting Started with Sentiment Analysis](https://huggingface.co/blog/sentiment-analysis-python)

### Dataset

- [Customer Sentiment Dataset on Kaggle](https://www.kaggle.com/datasets/kundanbedmutha/customer-sentiment-dataset)

### Libraries Documentation

- [Python Documentation](https://docs.python.org/3.13/contents.html)
- [Pandas Documentation](https://pandas.pydata.org/docs/user_guide/index.html)
- [Seaborn Documentation](https://seaborn.pydata.org/tutorial.html)
- [Matplotlib Documentation](https://matplotlib.org/stable/users/index.html)
- [Scikit-learn Documentation](https://scikit-learn.org/stable/user_guide.html)
- [NLTK Documentation](https://www.nltk.org/)
- [spaCy Documentation](https://spacy.io/usage)
- [Transformers Documentation](https://huggingface.co/docs/transformers/index)
- [Hugging Face Models](https://huggingface.co/nlptown/bert-base-multilingual-uncased-sentiment)
- [TextBlob Documentation](https://textblob.readthedocs.io/en/dev/quickstart.html#sentiment-analysis)
- [VADER Sentiment Analysis](https://vadersentiment.readthedocs.io/en/latest/pages/features_and_updates.html)