

!pip install -q librosa tensorflow scikit-learn matplotlib pydub streamlit pyngrok soundfile

# 🔑 Upload your Kaggle API key
from google.colab import files
files.upload()  # Upload kaggle.json here

# Setup Kaggle
!mkdir -p ~/.kaggle
!cp kaggle.json ~/.kaggle/
!chmod 600 ~/.kaggle/kaggle.json

# ✅ Download GTZAN dataset from working source
!kaggle datasets download -d andradaolteanu/gtzan-dataset-music-genre-classification
!unzip -q gtzan-dataset-music-genre-classification.zip

import os
DATA_DIR = "/content/Data/genres_original"
genres = sorted(os.listdir(DATA_DIR))
print(genres)
# ✅ Import libraries
import os
import librosa
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
from keras.utils import to_categorical
from tqdm import tqdm
import seaborn as sns

# ✅ Dataset path (upload to this location in Colab)
DATA_PATH = "/content/Data/genres_original"

# ✅ Parameters
SAMPLES_PER_TRACK = 660000
MAX_LEN = 130
N_MFCC = 40

# ✅ Feature extractor
def extract_features(file_path):
    try:
        audio, sr = librosa.load(file_path, duration=30)
        mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=N_MFCC)
        if mfcc.shape[1] < MAX_LEN:
            mfcc = np.pad(mfcc, ((0, 0), (0, MAX_LEN - mfcc.shape[1])), mode='constant')
        else:
            mfcc = mfcc[:, :MAX_LEN]
        return mfcc
    except Exception as e:
        print(f"❌ Failed for {file_path}: {e}")
        return None

# ✅ Load data
X, y = [], []
genres = os.listdir(DATA_PATH)
for genre in genres:
    genre_path = os.path.join(DATA_PATH, genre)
    if not os.path.isdir(genre_path): continue
    for file in tqdm(os.listdir(genre_path), desc=f"Processing {genre}"):
        if file.endswith((".wav", ".mp3", ".mp4")):
            file_path = os.path.join(genre_path, file)
            mfcc = extract_features(file_path)
            if mfcc is not None:
                X.append(mfcc)
                y.append(genre)

# ✅ Preprocessing
X = np.array(X)[..., np.newaxis]
y = np.array(y)
encoder = LabelEncoder()
y_encoded = to_categorical(encoder.fit_transform(y))
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

# ✅ CNN model definition
model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(N_MFCC, MAX_LEN, 1)),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(len(np.unique(y)), activation='softmax')
])

# ✅ Compile and train
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
history = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=30, batch_size=32)

# ✅ Save model
model.save("genre_cnn.keras")
print("✅ Model saved as genre_cnn.keras")

# ✅ Plot accuracy & loss
plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train')
plt.plot(history.history['val_accuracy'], label='Validation')
plt.title("Model Accuracy")
plt.legend()
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train')
plt.plot(history.history['val_loss'], label='Validation')
plt.title("Model Loss")
plt.legend()
plt.tight_layout()
plt.show()

# ✅ Evaluate performance
y_true = np.argmax(y_test, axis=1)
y_pred = np.argmax(model.predict(X_test), axis=1)
cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='g', xticklabels=encoder.classes_, yticklabels=encoder.classes_)
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Confusion Matrix")
plt.show()
print(classification_report(y_true, y_pred, target_names=encoder.classes_))


import streamlit as st
import librosa
import numpy as np
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
import librosa.display
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
