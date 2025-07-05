
import streamlit as st
import librosa
import librosa.display
import numpy as np
import tensorflow as tf
import os
from pydub import AudioSegment
import tempfile
import matplotlib.pyplot as plt

# Genre labels (adjust if your model differs)
GENRES = ['blues', 'classical', 'country', 'disco', 'hiphop',
          'jazz', 'metal', 'pop', 'reggae', 'rock']

# Load model
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("genre_cnn.h5")

model = load_model()

# Audio preprocessing
def preprocess_audio(file):
    # Convert to wav if needed
    ext = os.path.splitext(file.name)[1].lower()
    with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmpfile:
        if ext == '.mp3':
            sound = AudioSegment.from_mp3(file)
            sound.export(tmpfile.name, format='wav')
        elif ext == '.mp4':
            sound = AudioSegment.from_file(file, format="mp4")
            sound.export(tmpfile.name, format='wav')
        else:
            tmpfile.write(file.read())
            tmpfile.flush()

        y, sr = librosa.load(tmpfile.name, duration=30)
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
        mfcc = mfcc.T
        if mfcc.shape[0] < 1300:
            mfcc = np.pad(mfcc, ((0, 1300 - mfcc.shape[0]), (0, 0)))
        else:
            mfcc = mfcc[:1300]

        return mfcc[np.newaxis, ..., np.newaxis]

# 🎧 Streamlit App Interface
st.set_page_config(page_title="🎶 Music Genre Classifier", layout="centered")
st.title("🎧 Music Genre Classification App")
st.markdown("Upload an audio file (.mp3, .mp4, or .wav) to predict its genre using a CNN model.")

audio_file = st.file_uploader("Upload audio", type=["wav", "mp3", "mp4"])

if audio_file is not None:
    st.audio(audio_file, format='audio/wav')
    st.write(f"✅ File name: {audio_file.name}")
    st.write(f"🧪 Detected MIME type: {audio_file.type}")

    with st.spinner("Processing audio..."):
        features = preprocess_audio(audio_file)
        prediction = model.predict(features)[0]
        predicted_index = np.argmax(prediction)
        confidence = prediction[predicted_index] * 100
        predicted_genre = GENRES[predicted_index]

    st.success(f"🎵 Predicted Genre: **{predicted_genre.upper()}** ({confidence:.2f}%)")

    # Show genre probability chart
    st.subheader("📊 Genre Confidence")
    fig, ax = plt.subplots()
    ax.bar(GENRES, prediction, color='skyblue')
    ax.set_ylabel("Confidence")
    ax.set_xticklabels(GENRES, rotation=45)
    st.pyplot(fig)
