import streamlit as st
import whisper
import io
import numpy as np
import torch
from pydub import AudioSegment

# Load the Whisper model
model = whisper.load_model("base")  # You can change this to "tiny", "small", "medium", or "large"

def load_audio_file(audio_file):
    """Load audio file into a format that Whisper can process."""
    audio = AudioSegment.from_file(audio_file)
    audio = audio.set_channels(1).set_frame_rate(16000)  # Whisper works better with 1 channel (mono) and 16kHz sample rate
    audio = np.array(audio.get_array_of_samples())
    return audio

def transcribe_audio(audio):
    """Transcribe audio using Whisper."""
    result = model.transcribe(audio)
    return result['text']

# Streamlit UI
st.title("Whisper Audio Transcription App")
st.write("Upload an audio file to get transcribed text using OpenAI Whisper.")

# File uploader
audio_file = st.file_uploader("Upload an audio file", type=["mp3", "m4a", "wav", "flac"])

if audio_file is not None:
    st.audio(audio_file, format="audio/wav")  # Preview the uploaded audio
    st.write("Processing the file...")

    try:
        # Load and process audio file
        audio = load_audio_file(audio_file)
        
        # Transcribe the audio
        transcription = transcribe_audio(audio)

        # Display the transcription result
        st.subheader("Transcription Result")
        st.write(transcription)

    except Exception as e:
        st.error(f"Error processing the audio: {e}")
