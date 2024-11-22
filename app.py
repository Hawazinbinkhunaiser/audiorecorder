import streamlit as st
import whisper
from pydub import AudioSegment
import os

# Load the Whisper model
model = whisper.load_model("base")

def transcribe_audio(file_path):
    # Convert audio to the format Whisper can process (e.g., wav)
    audio = AudioSegment.from_file(file_path)
    audio = audio.set_channels(1).set_frame_rate(16000)
    temp_file = "temp_audio.wav"
    audio.export(temp_file, format="wav")

    # Use Whisper to transcribe the audio
    result = model.transcribe(temp_file)
    os.remove(temp_file)  # Clean up the temporary file
    return result['text']

# Streamlit UI
st.title("Audio Transcription with Whisper")

st.write("Upload an audio file to transcribe it.")

audio_file = st.file_uploader("Choose an audio file", type=["mp3", "m4a", "wav", "flac"])

if audio_file is not None:
    # Save the uploaded file to a temporary location
    with open("uploaded_audio", "wb") as f:
        f.write(audio_file.getbuffer())

    # Transcribe the audio
    st.write("Transcribing the audio...")
    transcription = transcribe_audio("uploaded_audio")

    # Show the transcription result
    st.subheader("Transcription Result:")
    st.write(transcription)

    # Option to download the transcription
    st.download_button("Download Transcription", transcription, file_name="transcription.txt")

    # Clean up the uploaded audio file
    os.remove("uploaded_audio")
