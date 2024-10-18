from flask import Flask, request, jsonify
import os
import torch
import numpy as np
import soundfile as sf
import noisereduce as nr
from pydub import AudioSegment
from pydub.silence import detect_nonsilent
import whisper
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from TTS.api import TTS

app = Flask(__name__)

# Ensure CUDA is available, otherwise use CPU
device = "cuda" if torch.cuda.is_available() else "cpu"

# Load Whisper model
whisper_model = whisper.load_model("small").to(device)

# Load TTS model
tts_model = TTS(model_name="tts_models/multilingual/multi-dataset/your_tts", progress_bar=False, gpu=(device == "cuda"))

# Mapping of language codes
LANG_MAP = {
    "en": "en",
    "es": "es",
    "fr": "fr-fr",
    "de": "de",
    "it": "it-it",
}

def extract_audio(input_file_path):
    audio = AudioSegment.from_file(input_file_path)
    audio = audio.set_channels(1).set_frame_rate(16000)
    temp_wav_path = "temp_audio.wav"
    audio.export(temp_wav_path, format="wav")
    return temp_wav_path

def remove_background_noise(audio_file):
    audio, sr = sf.read(audio_file)
    reduced_noise_audio = nr.reduce_noise(y=audio, sr=sr)
    cleaned_audio_file = "cleaned_audio.wav"
    sf.write(cleaned_audio_file, reduced_noise_audio, sr)
    return cleaned_audio_file

def transcribe_audio(audio_file, src_language):
    result = whisper_model.transcribe(audio_file, language=src_language)
    return result["text"]

def clone_voice(audio_file):
    # In a real implementation, you would use a voice cloning model here
    # For this example, we'll just return the path to the cleaned audio file
    return audio_file

def generate_speech(text, voice_file, language):
    output_file = "generated_speech.wav"
    tts_model.tts_to_file(text=text, speaker_wav=voice_file, language=LANG_MAP.get(language, "en"), file_path=output_file)
    return output_file

@app.route('/extract_voice', methods=['POST'])
def extract_voice():
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400
    
    if file:
        file_path = os.path.join("uploads", file.filename)
        file.save(file_path)
        
        audio_path = extract_audio(file_path)
        cleaned_audio = remove_background_noise(audio_path)
        cloned_voice = clone_voice(cleaned_audio)
        
        return jsonify({"message": "Voice extracted successfully", "voice_file": cloned_voice}), 200

@app.route('/generate_speech', methods=['POST'])
def generate_speech_route():
    data = request.json
    text = data.get('text')
    language = data.get('language', 'en')
    voice_file = data.get('voice_file')
    
    if not text or not voice_file:
        return jsonify({"error": "Missing text or voice file"}), 400
    
    output_file = generate_speech(text, voice_file, language)
    
    return jsonify({"message": "Speech generated successfully", "audio_file": output_file}), 200

if __name__ == '__main__':
    os.makedirs("uploads", exist_ok=True)
    app.run(debug=True)