"""
Simple text classifier that predicts whether a message or customer review 
is positive or negative. Uses a simple neural network for classification.
"""


import nltk
#nltk.download('punkt_tab', force=True)
#nltk.download('stopwords')
#nltk.download('wordnet')

import pandas as pd
import string
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer



import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# Initialize stopwords and lemmatizer
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

#Feed uncleaned text to clean it
def preprocess_text(text):
    text = str(text)
    # Lowercase
    text = text.lower()
    
    # Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))
    
    # Tokenize
    tokens = word_tokenize(text)
    
    # Remove stopwords
    tokens = [word for word in tokens if word not in stop_words]
    
    # Lemmatize
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    
    # Join back to string if needed
    #print("Tokens:", tokens)

    return ' '.join(tokens)


def load_data(csv_file):
    """Load and preprocess data"""
    df = pd.read_csv(csv_file)
    df.columns = df.columns.str.strip()
    df['cleaned'] = df['message'].apply(preprocess_text)
    
    # Since your sentiment is already 1/0, just use it directly
    df['label'] = df['sentiment']
    
    # Remove rows with empty cleaned text
    df = df[df['cleaned'].str.len() > 0]
    
    return df




#Converting the cleaner text into a bag of words vectors
def create_bow_vectors(texts, vectorizer=None, fit=True):
    """Convert text to Bag of Words vectors"""
    if vectorizer is None:
        vectorizer = CountVectorizer(max_features=1000)
    
    if fit:
        vectors = vectorizer.fit_transform(texts)
    else:
        vectors = vectorizer.transform(texts)
    
    return vectors.toarray(), vectorizer



def build_model(input_size):
    """Build neural network as per requirements"""
    model = Sequential([
        Dense(16, activation='relu', input_shape=(input_size,)),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

def main():
    # Load and prepare data
    df = load_data('feedback.csv')
    print(df.dtypes)
    print(f"Loaded {len(df)} samples")
    print(f"Positive: {sum(df['label'])}, Negative: {len(df) - sum(df['label'])}")
    
    X_train_text, X_test_text, y_train, y_test = train_test_split(
        df['cleaned'], df['label'], test_size=0.2, random_state=42
    )
    
    # Create BoW vectors
    X_train, vectorizer = create_bow_vectors(X_train_text, fit=True)
    X_test, _ = create_bow_vectors(X_test_text, vectorizer, fit=False)
    
    # Convert labels to numpy arrays to avoid issues
    y_train = np.array(y_train, dtype=np.float32)
    y_test = np.array(y_test, dtype=np.float32)
    
    print(f"Training data shape: {X_train.shape}")
    print(f"Vocabulary size: {X_train.shape[1]}")
    
    # Build and train model
    model = build_model(X_train.shape[1])
    model.fit(X_train, y_train, epochs=20, batch_size=4, verbose=1)
    
    # Evaluate
    y_pred_prob = model.predict(X_test)
    y_pred = (y_pred_prob > 0.5).astype(int).flatten()
    accuracy = accuracy_score(y_test, y_pred)
    print(f"\nAccuracy: {accuracy:.4f}")
    
    # Test with new text
    while True:
        text = input("\nEnter text to classify (or 'quit'): ")
        if text.lower() == 'quit':
            break
        
        cleaned = preprocess_text(text)
        if cleaned.strip() == "":
            print("Text became empty after preprocessing!")
            continue
            
        vector = vectorizer.transform([cleaned]).toarray()
        prediction = model.predict(vector)[0][0]
        sentiment = "Positive" if prediction > 0.5 else "Negative"
        print(f"Prediction: {sentiment} ({prediction:.3f})")

if __name__ == "__main__":
    main()