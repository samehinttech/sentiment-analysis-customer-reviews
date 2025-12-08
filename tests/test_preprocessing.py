# import re
# import string
# import nltk
# from nltk.tokenize import word_tokenize
# from nltk.corpus import stopwords
# from nltk.stem import WordNetLemmatizer
# import spacy
# import pandas as pd
#
# # Download NLTK resources
# nltk_resources = ['punkt', 'punkt_tab', 'stopwords', 'wordnet', 'omw-1.4']
# for resource in nltk_resources:
#     try:
#         nltk.download(resource, quiet=True)
#         print("Finished downloading:", resource)
#     except Exception as e:
#         print(f"Error downloading {resource}: {e}\n")
# # Load dataset
# df = pd.read_csv('../data/customer_sentiment.csv')
# # Initialize tools
# lemmatizer = WordNetLemmatizer()
# nlp = spacy.load('en_core_web_sm')
#
#
# # Text Cleaning Function
# def clean_text(text):
#     """ Clean and Normalize Text Data
#      Steps:
#       1. Lowercase the text
#       2. Remove URLs
#       3. Remove emails addresses, we actually don't have any in the dataset
#       4. Remove punctuation
#       5. Remove extra whitespace
#      """
#     # Convert to lowercase
#     text = text.lower()
#     # Remove URLs
#     text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
#     # Remove email addresses
#     text = re.sub(r'\S+@\S+', '', text)
#     # Remove punctuation
#     text = text.translate(str.maketrans('', '', string.punctuation))
#     # Remove extra whitespace
#     text = re.sub(r'\s+', ' ', text).strip()
#     return text
#
#
# # Tokenization function
# def tokenize_text(text):
#     """ Tokenize text into words """
#     tokens = word_tokenize(text)
#     return tokens
#
#
# # Stopword Removal Function
# def remove_stopwords(tokens):
#     """
#      Remove common English stopwords, but we decided to keep negations.
#      Like:
#         not, no,n't, don’t, doesn’t, didn’t,
#         never, none, nobody, nothing, neither, nor.
#      Negations are crucial for sentiment analysis.
#      """
#     stop_words = set(stopwords.words('english'))
#     negations = {"not", "no", "n't", "don’t", "doesn’t", "didn’t",
#                  "never", "none", "nobody", "nothing", "neither", "nor"}
#     filtered_tokens = [word for word in tokens if word not in stop_words or word in negations]
#     return filtered_tokens
#
#
# # Lemmatization Function
# def lemmatize_tokens_nltk(tokens):
#     """ Lemmatize text using NLTK's WordNetLemmatizer
#     Example:
#         packages -> package
#         delivered -> deliver
#     """
#     lemmatize = WordNetLemmatizer()
#     return [lemmatize.lemmatize(token) for token in tokens]
#
#
# def lemmatize_tokens_spacy(tokens):
#     """ Lemmatize text using spaCy
#     Example:
#       running -> run, better -> good | While NLTK may not handle these well as been found while our research.
#     """
#     # join tokens back to text for spaCy processing
#     text = ' '.join(tokens)
#     doc = nlp(text)
#     return [token.lemma_ for token in doc]
#
#
# # Preprocessing Pipeline
# def preprocess_pipeline(text, method='nltk'):
#     """ Complete Preprocessing Pipeline
#      method: 'nltk' or 'spacy' for lemmatization
#     """
#     text = clean_text(text)
#     tokens = tokenize_text(text)
#     tokens = remove_stopwords(tokens)
#     if method == 'nltk':
#         tokens = lemmatize_tokens_nltk(tokens)
#     elif method == 'spacy':
#         tokens = lemmatize_tokens_spacy(tokens)
#     else:
#         raise ValueError("Method must be 'nltk' or 'spacy'")
#     processed_text = ' '.join(tokens)
#     return processed_text
#
#
# # Apply preprocessing to the 'review_text' column
# df['processed_text_nltk'] = df['review_text'].apply(lambda x: preprocess_pipeline(x, method='nltk'))
# df['processed_text_spacy'] = df['review_text'].apply(
#     lambda x: preprocess_pipeline(x, method='spacy'))
#
#
# if __name__ == "__main__":
#     # Run preprocessing on the dataset and display a few samples
#     for i in range(5):
#         print(f"\nSample {i + 1}:")
#         print("Original:", df.loc[i, 'review_text'])
#         print("Processed (NLTK):", df.loc[i, 'processed_text_nltk'])
#         print("Processed (spaCy):", df.loc[i, 'processed_text_spacy'])
