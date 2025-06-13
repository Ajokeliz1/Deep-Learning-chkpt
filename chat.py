import random
import string
import nltk
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import streamlit as st
import speech_recognition as sr

# Download NLTK resources
nltk.download('punkt')
nltk.download('wordnet')
# Download NLTK resources
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('punkt_tab') # Add this line
# Load and preprocess data
def load_data(pairs_file):
    with open(pairs_file, 'r', encoding='utf-8', errors='ignore') as file:
        raw_data = file.read().lower()
    sentences = nltk.sent_tokenize(raw_data)
    lemmatizer = WordNetLemmatizer()
    return sentences, lemmatizer

# Generate chatbot response
def chatbot_response(user_input, sentences, lemmatizer):
    if user_input.lower() in ['bye', 'goodbye', 'exit']:
        return "Goodbye! Have a great day."
    
    sentences.append(user_input)
    word_tokens = nltk.word_tokenize(user_input)
    lemmatized_tokens = [lemmatizer.lemmatize(token) for token in word_tokens]
    processed_input = ' '.join(lemmatized_tokens)
    
    tfidf_vectorizer = TfidfVectorizer(tokenizer=nltk.word_tokenize, stop_words='english')
    tfidf_matrix = tfidf_vectorizer.fit_transform(sentences)
    
    cosine_similarities = cosine_similarity(tfidf_matrix[-1], tfidf_matrix)
    similar_idx = cosine_similarities.argsort()[0][-2]
    matched_sentence = sentences[similar_idx]
    
    sentences.remove(user_input)
    return matched_sentence if cosine_similarities[0, similar_idx] > 0 else "I'm sorry, I don't understand that."

# Transcribe speech to text
def transcribe_speech():
    r = sr.Recognizer()
    with sr.Microphone() as source:
        st.info("Listening... Speak now")
        audio = r.listen(source)
        try:
            text = r.recognize_google(audio)
            return text
        except sr.UnknownValueError:
            return "Could not understand audio"
        except sr.RequestError:
            return "API unavailable"
def main():
    st.title("Speech-Enabled Chatbot")
    
    # Initialize session state
    if 'history' not in st.session_state:
        st.session_state.history = []
    
    # Load chatbot data
    sentences, lemmatizer = load_data('pairs.txt')  # Replace with your file path
    
    # Input selection
    input_method = st.radio("Choose input method:", ("Text", "Speech"))
    user_input = ""
    
    if input_method == "Speech":
        if st.button("Start Recording"):
            user_input = transcribe_speech()
    else:
        user_input = st.text_input("Type your message:")
    
    # Process input and generate response
    if user_input:
        response = chatbot_response(user_input, sentences, lemmatizer)
        st.session_state.history.append(("You", user_input))
        st.session_state.history.append(("Chatbot", response))
    
    # Display conversation history
    for speaker, message in st.session_state.history:
        st.write(f"**{speaker}:** {message}")

if __name__ == "__main__":
    main()