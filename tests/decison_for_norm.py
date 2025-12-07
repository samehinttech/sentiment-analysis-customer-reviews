# Exam the dataset for normalization decision
import pandas as pd
from collections import Counter

# Load dataset
df_check = pd.read_csv('../data/customer_sentiment.csv')

# Sample review texts
print("\n1. SAMPLE REVIEW TEXTS (first 10):")
for i, review in enumerate(df_check['review_text'].head(10), 1):
    print(f"\n   {i}. {review[:]}")

# Platform distribution
print("\n2. PLATFORM DISTRIBUTION:")
print(df_check['platform'].value_counts())

# Get all words from reviews (lowercased)
all_words = ' '.join(df_check['review_text'].str.lower()).split()
word_counts = Counter(all_words)

print("\n3. TOP 30 MOST COMMON WORDS IN REVIEWS:")
for word, count in word_counts.most_common(50):
    print(f"   {word:20s}: {count:,}")

# Check for retail/delivery terms
retail_terms = ['delivery', 'deliver', 'delivered', 'shipping', 'shipped', 'ship',
                'refund', 'return', 'returned', 'money', 'back',
                'order', 'ordered', 'ordering', 'package', 'packaging',
                'quality', 'product', 'service', 'customer']

print("\n4. FREQUENCY OF RETAIL/DELIVERY TERMS:")
for term in retail_terms:
    count = sum(1 for review in df_check['review_text'].str.lower() if term in review)
    if count > 0:
        print(f"   {term:15s}: appears in {count:,} reviews ({count / len(df_check) * 100:.1f}%)")

# 6. Check platform mentions in text
print("\n5. PLATFORM NAMES MENTIONED IN REVIEWS:")
platforms = df_check['platform'].unique()
for platform in platforms[:]:  # Check first 10 platforms
    count = sum(1 for review in df_check['review_text'].str.lower() if platform.lower() in review)
    if count > 0:
        print(f"   {platform:20s}: mentioned {count} times")
print("not mentioned")

# Conclusion
# No platforms mentioned in text - platform normalization unnecessary
# Terms are already simple - "delivery", "quality", "product" are base forms
# Lemmatization could be considered for words like "delivered" -> "deliver",
# "ordered" -> "order" "packing" -> "packaging" -> "package"
# Reviews are short - complex normalization adds small value
# Decision: (clean + lemmatize + stopwords)
