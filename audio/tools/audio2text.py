"""
File: Amit Tzadok
Author: Amit Tzadok <amit.tzadok@icloud.com>
Created: 2025-07-21 19:55:35
Description: This script transcribes a WAV audio file to text.
"""

import speech_recognition as sr

def audio_to_text(wav_path):
    r = sr.Recognizer()
    with sr.AudioFile(wav_path) as source:
        audio = r.record(source)
    try:
        return r.recognize_google(audio)
    except sr.UnknownValueError:
        return "Unintelligible audio"
    except sr.RequestError as e:
        return f"API Error: {e}"

def main():
    input_audio_path = input("Enter the path to your input audio file (.wav): ")

    # Check if the input file is a WAV file
    if not input_audio_path.lower().endswith('.wav'):
        print("Error: Only .wav audio files are supported.")
        return

    # Convert audio to text
    transcription = audio_to_text(input_audio_path)
    print("Transcription:", transcription)

if __name__ == "__main__":
    main()
