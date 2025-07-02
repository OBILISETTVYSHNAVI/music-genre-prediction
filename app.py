import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
import os
import soundfile as sf
from pydub import AudioSegment

# Load model
model = load_model("genre_cnn.h5")
genres = ['blues', 'classical', 'country', 'disco', 'hiphop',
          'jazz', 'metal', 'pop', 'reggae', 'rock']

def extract_features(file_path):
    audio, sr = librosa.load(file_path, duration=30, mono=True)
    mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=40)
    return mfccs[..., np.newaxis]

def predict_genre(audio_file):
    file_path = "temp_audio.wav"
    audio = AudioSegment.from_file(audio_file)
    audio.export(file_path, format="wav")

    features = extract_features(file_path)
    features = np.expand_dims(features, axis=0)
    predictions = model.predict(features)[0]

    return predictions

st.title("🎵 Music Genre Classifier")
audio_file = st.file_uploader("Upload Audio File (.mp3, .wav, .mp4)", type=["mp3", "wav", "mp4"])

if audio_file is not None:
    st.audio(audio_file, format='audio/wav')
    st.write(f"✅ File name: {audio_file.name}")
    st.write(f"🧪 Detected MIME type: {audio_file.type}")

    predictions = predict_genre(audio_file)
    top_genre = genres[np.argmax(predictions)]
    confidence = np.max(predictions) * 100

    st.subheader(f"🎧 Predicted Genre: {top_genre}")
    st.write(f"Confidence: {confidence:.2f}%")

    # Show bar chart
    st.bar_chart(predictions)
