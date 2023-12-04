import streamlit as st
import numpy as np
import tensorflow as tf
import librosa
import noisereduce as nr
import os
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()

# Load your pre-trained model
# model = tf.keras.models.load_model(r"C:\Users\N Akhil\bird\bird_species_model.h5")  # Replace with your model path
model = tf.keras.models.load_model(r"birdwebsite/bird_species_model.h5")  # Replace with your model path


# Define a function for bird species prediction
def predict_bird_species(audio_data, sample_rate,le):
    # Extract audio features (e.g., MFCCs or spectrogram) from audio_data
    audio_features = extract_features(audio_data, sample_rate)
    audio_features = audio_features.T  # Transpose to match (num_frames, num_mfcc) shape
    audio_features = np.expand_dims(audio_features, axis=0)
    prediction = model.predict(audio_features)  # Reshape to match the model's input shape
    predicted_class_index = np.argmax(prediction)
    predicted_class = le.inverse_transform([predicted_class_index])[0]
    return predicted_class

# Define a dictionary mapping bird species to image URLs or file paths
bird_images = {
    'Bewick\'s Wren': 'birdwebsite/benwick.jpeg',
    'Northern Mockingbird': 'birdwebsite/northmock.jpeg',
    'American Robin': 'birdwebsite/americanrobin.jpeg',
    'Song Sparrow': 'birdwebsite/songsparrow.jpeg',
    'Northern Cardinal': 'birdwebsite/northcardinal.jpeg',
}


def extract_features(audio_data, sample_rate):
    yt = nr.reduce_noise(y=audio_data, sr=sample_rate)
    mfccs = librosa.feature.mfcc(y=yt, sr=sample_rate, n_mfcc=40)
    mfccs_processed = np.mean(mfccs.T, axis=0)
    return mfccs_processed
    
# Remove the GitHub icon on the top right
st.markdown(
    """
    <link rel="stylesheet" href="style.css">
    """,
    unsafe_allow_html=True,
)

# Streamlit app
st.title("Bird Species Identification")



# Create a file uploader component
audio_file = st.file_uploader("Upload an audio file", type=["wav"])

if audio_file:
    st.audio(audio_file, format="audio/wav")

    # Button to trigger prediction
    if st.button("Identify Bird Species"):
        st.write("Predicting bird species... Please wait.")
        # Read the audio file and sample rate
        audio_data, sample_rate = librosa.load(audio_file, sr=None)
        le = LabelEncoder()
        le.classes_ = np.array(['Bewick\'s Wren','Northern Mockingbird','American Robin','Song Sparrow','Northern Cardinal'])
        predicted_species = predict_bird_species(audio_data, sample_rate, le)
        st.success("Prediction complete!")

        # Display the predicted species
        st.subheader("Bird Species Prediction:")
        st.write(f"The predicted bird species is: {predicted_species}")
        # Display the corresponding bird image
        if predicted_species in bird_images:
            bird_image_path = bird_images[predicted_species]
            st.image(bird_image_path, caption=f"{predicted_species} Image", use_column_width=True)
        else:
            st.warning("No image available for the predicted bird species.")



