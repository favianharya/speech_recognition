import speech_recognition as sr
from transformers import T5ForConditionalGeneration, T5Tokenizer
from pydub import AudioSegment
import sentencepiece
import torch

import warnings
warnings.filterwarnings("ignore")


def get_audio_length(file_path):
    audio = AudioSegment.from_file(file_path)
    length_in_seconds = len(audio) / 1000.0  # pydub calculates in milliseconds
    return length_in_seconds

def speech_to_text(audio_file):
    """
    Opens and listens to an audio file and translates it to text
    Args: audio file
    Returns: text of transcribed audio file
    """
    r = sr.Recognizer()
    text = ""
    
    try:
        with sr.AudioFile(audio_file) as source:
            audio_duration = get_audio_length(audio_file)
            offset = 0
            chunk_size = 60  # chunk size in seconds
            
            while offset < audio_duration:
                audio_text = r.record(source, offset=offset, duration=min(chunk_size, audio_duration - offset))
                try:
                    text_chunk = r.recognize_google(audio_text)
                    text += text_chunk + " "
                except sr.UnknownValueError:
                    print('Google Speech Recognition could not understand the audio.')
                except sr.RequestError as e:
                    print(f'Could not request results from Google Speech Recognition service; {e}')
                    break
                
                offset += chunk_size  # move offset to the next chunk
            
    except Exception as e:
        print(f'An error occurred: {e}')

    return text

audio = '/Users/t-favian.adrian/Documents/data_science_project/speech-to-text/Speaker26_000.wav'
text = speech_to_text(audio)

from transformers import T5ForConditionalGeneration, T5Tokenizer

model_name = "t5-small"
tokenizer = T5Tokenizer.from_pretrained(model_name)
model = T5ForConditionalGeneration.from_pretrained(model_name)

preprocessed_text = "summarize: " + text
inputs = tokenizer.encode(preprocessed_text, return_tensors="pt", max_length=512, truncation=True)

summary_ids = model.generate(inputs, max_length=80, min_length=30, length_penalty=2.0, num_beams=4, early_stopping=True)
summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)

print("\n")
print("Original Text: ", text)
print("Summary: ", summary)
