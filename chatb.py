import nltk
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import streamlit as st
import speech_recognition as sr
import tempfile

# Download NLTK resources
nltk.download('punkt')
nltk.download('wordnet')

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

# Transcribe from uploaded audio
def transcribe_audio_file(audio_file):
    recognizer = sr.Recognizer()
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
        tmp_file.write(audio_file.read())
        tmp_file_path = tmp_file.name
    
    with sr.AudioFile(tmp_file_path) as source:
        audio = recognizer.record(source)
        try:
            return recognizer.recognize_google(audio)
        except sr.UnknownValueError:
            return "Could not understand audio"
        except sr.RequestError:
            return "API unavailable"

def main():
    st.title("Speech-Enabled Chatbot (via Audio File)")
    
    # Initialize session state
    if 'history' not in st.session_state:
        st.session_state.history = []
    
    # Load chatbot data
    sentences, lemmatizer = load_data('pairs.txt')  # Ensure 'pairs.txt' exists in app folder
    
    # Input method
    input_method = st.radio("Choose input method:", ("Text", "Upload Audio (.wav)"))
    user_input = ""
    
    if input_method == "Upload Audio (.wav)":
        uploaded_audio = st.file_uploader("Upload a WAV file", type=["wav"])
        if uploaded_audio is not None:
            st.info("Transcribing...")
            user_input = transcribe_audio_file(uploaded_audio)
            st.success(f"You said: {user_input}")
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
