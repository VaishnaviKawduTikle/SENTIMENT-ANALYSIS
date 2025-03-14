import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from textblob import TextBlob  # For sentiment analysis

# Download stopwords if not already downloaded
nltk.download('stopwords')

# Load CSV file (already working)
file_path = r"C:\Users\vaish\OneDrive\Desktop\vaishu amazon.csv"  # Change this if needed
df = pd.read_csv(r"C:\Users\vaish\OneDrive\Desktop\vaishu amazon.csv")

# Choose the correct column for reviews
TEXT_COLUMN = "review_content"  # Change to "review_title" if needed

# Ensure the column exists
if TEXT_COLUMN not in df.columns:
    raise ValueError(f"Column '{TEXT_COLUMN}' not found! Check column names.")

# Function to clean text
def preprocess_text(text):
    if pd.isna(text):
        return ""  # Handle NaN values
    text = str(text).lower()  # Convert to lowercase
    text = re.sub(r'\W', ' ', text)  # Remove special characters
    text = re.sub(r'\s+', ' ', text).strip()  # Remove extra spaces
    words = text.split()
    words = [word for word in words if word not in stopwords.words('english')]  # Remove stopwords
    return ' '.join(words)

# Apply text preprocessing
df['cleaned_text'] = df[TEXT_COLUMN].apply(preprocess_text)

# Function for sentiment analysis
def get_sentiment(text):
    return TextBlob(text).sentiment.polarity  # Returns sentiment score (-1 to 1)

# Apply sentiment analysis
df['sentiment_score'] = df['cleaned_text'].apply(get_sentiment)

# Classify sentiment
df['sentiment'] = df['sentiment_score'].apply(lambda x: "Positive" if x > 0 else ("Negative" if x < 0 else "Neutral"))

# Display results
print(df[['cleaned_text', 'sentiment_score', 'sentiment']].head())

# Save results to a new CSV file
df.to_csv("sentiment_results.csv", index=False)
print("✅ Sentiment analysis completed! Results saved to 'sentiment_results.csv'.")
df.to_csv(r"C:\Users\vaish\OneDrive\Desktop\sentiment_results.csv", index=False)

