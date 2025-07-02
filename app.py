import streamlit as st
import librosa
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
from keras.models import load_model
from sklearn.preprocessing import LabelEncoder

N_MFCC = 40
MAX_LEN = 130
MODEL_PATH = "genre_cnn.h5"
LABELS = ['blues', 'classical', 'country', 'disco', 'hiphop',
          'jazz', 'metal', 'pop', 'reggae', 'rock']

model = load_model(MODEL_PATH)
encoder = LabelEncoder()
encoder.fit(LABELS)

def extract_features(audio_path):
    try:
        audio, sr = librosa.load(audio_path, duration=30)
        mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=N_MFCC)
        if mfcc.shape[1] < MAX_LEN:
            mfcc = np.pad(mfcc, ((0, 0), (0, MAX_LEN - mfcc.shape[1])), mode='constant')
        else:
            mfcc = mfcc[:, :MAX_LEN]
        return mfcc[..., np.newaxis]
    except Exception as e:
        st.error(f"❌ Audio load failed: {e}")
        return None

st.set_page_config(page_title="🎵 Music Genre Classifier", layout="centered")
st.title("🎧 Music Genre Classifier")
st.write("Upload an audio file (`.mp3`, `.wav`, `.mp4`) to predict its genre.")

audio_file = st.file_uploader("Upload your audio file", type=["wav", "mp3", "mp4"])

if audio_file is not None:
    st.success(f"✅ File uploaded: {audio_file.name}")
    with open("temp_audio", "wb") as f:
        f.write(audio_file.read())

    features = extract_features("temp_audio")

    if features is not None:
        features = np.expand_dims(features, axis=0)
        prediction = model.predict(features)[0]
        predicted_index = np.argmax(prediction)
        predicted_label = encoder.inverse_transform([predicted_index])[0]
        confidence = prediction[predicted_index]

        st.markdown(f"### 🎶 Predicted Genre: **{predicted_label.upper()}**")
        st.markdown(f"### 🔮 Confidence: `{confidence:.2%}`")

        st.markdown("#### 🎯 Genre Prediction Probabilities")
        fig, ax = plt.subplots(figsize=(10, 4))
        sns.barplot(x=encoder.classes_, y=prediction, palette="viridis", ax=ax)
        plt.xticks(rotation=45)
        plt.ylabel("Probability")
        st.pyplot(fig)
