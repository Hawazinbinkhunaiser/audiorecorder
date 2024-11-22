import streamlit as st
import whisper
import subprocess
from audio_recorder_streamlit import audio_recorder

# Install FFmpeg if needed (for Streamlit Cloud)
# You can skip this if FFmpeg is already installed in your environment
def install_ffmpeg():
    subprocess.run(['apt-get', 'update', '-y'])
    subprocess.run(['apt-get', 'install', 'ffmpeg', '-y'])

# Install FFmpeg (only needed for Streamlit Cloud)
try:
    import ffmpeg
except ImportError:
    install_ffmpeg()

# Load Whisper model (can use "small", "medium", or "large")
model = whisper.load_model("medium")

def transcribe_text_to_voice(audio_location):
    # Transcribe the audio file using Whisper
    result = model.transcribe(audio_location)
    return result["text"]
    
st.title("🧑‍💻 Skolo Online 💬 Talking Assistant")

"""
Hi🤖 just click on the voice recorder and let me know how I can help you today?
"""

audio_bytes = audio_recorder()
if audio_bytes:
    ##Save the Recorded File
    audio_location = "audio_file.wav"
    with open(audio_location, "wb") as f:
        f.write(audio_bytes)

    #Transcribe the saved file to text
    text = transcribe_text_to_voice(audio_location)
    st.write(text)
