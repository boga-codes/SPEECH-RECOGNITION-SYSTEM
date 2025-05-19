import os
import torch
import librosa
from transformers import Wav2Vec2ForCTC, Wav2Vec2Tokenizer


def load_audio(file_path, target_sr=16000):
    """Load and resample audio file to target sample rate."""
    print(f"Reading audio file from: {file_path}")
    waveform, sample_rate = librosa.load(file_path, sr=target_sr)
    return waveform


def load_model_and_tokenizer(model_name="facebook/wav2vec2-base-960h"):
    """Load the pretrained Wav2Vec2 model and tokenizer."""
    print("Initializing tokenizer and model...")
    tokenizer = Wav2Vec2Tokenizer.from_pretrained(model_name)
    model = Wav2Vec2ForCTC.from_pretrained(model_name)
    return tokenizer, model


def generate_transcription(audio_data, tokenizer, model):
    """Transcribe speech from audio waveform."""
    print("Processing transcription...")
    inputs = tokenizer(audio_data, return_tensors="pt", padding="longest").input_values
    with torch.inference_mode():
        output_logits = model(inputs).logits
    predicted_tokens = torch.argmax(output_logits, dim=-1)
    transcription = tokenizer.batch_decode(predicted_tokens)[0]
    return transcription.strip()


def main(audio_path):
    if not os.path.isfile(audio_path):
        print(f"File not found: {audio_path}")
        return

    tokenizer, model = load_model_and_tokenizer()
    audio_wave = load_audio(audio_path)
    result_text = generate_transcription(audio_wave, tokenizer, model)

    print("\n--- Transcription Result ---")
    print(result_text)


if __name__ == "__main__":
    AUDIO_FILE_PATH = "audio/sample.wav"
    main(AUDIO_FILE_PATH)
