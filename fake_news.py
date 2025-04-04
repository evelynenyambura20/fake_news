import streamlit as st
import pandas as pd
import numpy as np
import re
import string
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import io
import joblib
import logging
import json
import chardet

# Set up logging
logging.basicConfig(level=logging.INFO, filename='app.log', filemode='a',
                    format='%(asctime)s - %(levelname)s - %(message)s')

# Download NLTK resources
nltk.download('stopwords', quiet=True)
nltk.download('punkt', quiet=True)
nltk.download('wordnet', quiet=True)
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

# Define color palette
LABEL_COLORS = {'fake': '#fc8d62', 'real': '#66c2a5'}

def clean_text(text, remove_stopwords=True, lemmatize=True):
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(f'[{re.escape(string.punctuation)}]', '', text)
    text = re.sub(r'\d+', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    tokens = word_tokenize(text)
    if remove_stopwords:
        tokens = [word for word in tokens if word not in stop_words]
    if lemmatize:
        tokens = [lemmatizer.lemmatize(word) for word in tokens]
    return ' '.join(tokens)

def load_file(uploaded_file):
    try:
        file_extension = uploaded_file.name.split('.')[-1].lower()
        raw_data = uploaded_file.read()
        result = chardet.detect(raw_data)
        encoding = result['encoding'] or 'utf-8'
        
        if file_extension == 'csv':
            return pd.read_csv(io.StringIO(raw_data.decode(encoding)))
        elif file_extension == 'xlsx':
            return pd.read_excel(io.BytesIO(raw_data))
        elif file_extension == 'txt':
            return pd.read_csv(io.StringIO(raw_data.decode(encoding)), delimiter='\t')
        elif file_extension == 'json':
            return pd.read_json(io.StringIO(raw_data.decode(encoding)))
        else:
            st.error("Unsupported file format")
            return None
    except Exception as e:
        st.error(f"Error loading file: {str(e)}")
        return None

def display_label_counts(df, label_column, title="Label Counts"):
    counts = df[label_column].value_counts()
    st.write(f"{title}:")
    for label, count in counts.items():
        st.write(f"{str(label).capitalize()}: {count}")

def plot_confusion_matrix(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')
    ax.set_title('Confusion Matrix')
    st.pyplot(fig)

def generate_wordcloud(text_data, title):
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(' '.join(text_data))
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.imshow(wordcloud, interpolation='bilinear')
    ax.axis('off')
    ax.set_title(title)
    st.pyplot(fig)

def train_and_predict(df, text_column, label_column):
    # Clean text data
    df['cleaned_text'] = df[text_column].apply(clean_text)
    
    # Vectorize text
    tfidf = TfidfVectorizer(max_features=5000)
    X = tfidf.fit_transform(df['cleaned_text']).toarray()
    y = df[label_column]
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train model
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)
    
    # Predict
    y_pred = model.predict(X_test)
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    
    return model, tfidf, accuracy, report, y_test, y_pred

def main():
    st.title("Fake News Detection Dashboard")
    
    # File upload
    uploaded_file = st.file_uploader("Upload CSV, Excel, TXT, or JSON", 
                                   type=["csv", "xlsx", "txt", "json"])
    
    if uploaded_file:
        df = load_file(uploaded_file)
        if df is not None:
            st.subheader("Dataset Preview")
            st.dataframe(df.head())
            
            # Column selection
            text_column = st.selectbox("Select Text Column:", df.columns)
            label_column = st.selectbox("Select Label Column:", df.columns)
            
            # Display initial label counts
            display_label_counts(df, label_column, "Label Distribution")
            
            if st.button("Analyze and Train Model"):
                try:
                    # Train model and get results
                    model, tfidf, accuracy, report, y_test, y_pred = train_and_predict(df, text_column, label_column)
                    
                    # Display results
                    st.subheader("Model Performance")
                    st.write(f"Accuracy: {accuracy:.2%}")
                    st.text("Classification Report:")
                    st.text(report)
                    
                    # Plot confusion matrix
                    st.subheader("Confusion Matrix")
                    plot_confusion_matrix(y_test, y_pred)
                    
                    # Generate word clouds
                    st.subheader("Word Clouds")
                    for label in df[label_column].unique():
                        label_text = df[df[label_column] == label]['cleaned_text']
                        generate_wordcloud(label_text, f"Word Cloud for {label} News")
                    
                    # Save model option
                    if st.button("Save Model"):
                        joblib.dump(model, 'fake_news_model.pkl')
                        joblib.dump(tfidf, 'tfidf_vectorizer.pkl')
                        st.success("Model and vectorizer saved successfully!")
                        
                except Exception as e:
                    st.error(f"Error during analysis: {str(e)}")
                    logging.error(f"Analysis error: {str(e)}")

if __name__ == "__main__":
    main()