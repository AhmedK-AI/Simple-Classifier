# Text Sentiment Classifier

## Project Overview

This project creates a sentiment classifier to categorize customer feedback as **positive** or **negative** using NLP and a neural network.

## How It Works

### 1. Text Preprocessing

The raw text undergoes these steps using **NLTK**:

* **Lowercasing**: "GREAT Service!" → "great service!"
* **Punctuation removal**: "great service!" → "great service"
* **Tokenization**: "great service" → \["great", "service"]
* **Stopword removal**: Removes common words like "the", "and".
* **Lemmatization**: Converts words like "running" → "run".

### 2. Bag of Words (BoW) Vectorization

* Converts text into numerical vectors based on word frequency.
* Example: "great service" → `[1, 0, 1, 0, 0, ...]`

### 3. Neural Network

* **Input Layer**: Size equals vocabulary size (277 features in this case)
* **Hidden Layer**: 16 neurons, ReLU activation
* **Output Layer**: Single neuron with sigmoid activation (outputs probability 0-1)
* **Training**: 20 epochs, batch size 4, Adam optimizer

## Files Required

* `text_classifier.py` - Main code
* `feedback.csv` - Dataset with columns: id, message, sentiment (1=positive, 0=negative)

### CSV Format Example

```
id,message,sentiment
1,"The delivery was super fast and the packaging was neat!",1
2,"Terrible service. My order arrived a week late and damaged.",0
```

## How to Run

1. Install required packages:

   ```bash
   pip install nltk pandas numpy scikit-learn tensorflow
   ```

2. Download NLTK data (run once):

   ```python
   import nltk
   nltk.download('punkt')
   nltk.download('stopwords')
   nltk.download('wordnet')
   ```

3. Place `feedback.csv` in the same directory as the script.

4. Run the program:

   ```bash
   python text_classifier.py
   ```

## What You'll See

* Output of text preprocessing steps
* Dataset loading confirmation
* Training progress (20 epochs)
* Final accuracy score
* Interactive mode to test new text

