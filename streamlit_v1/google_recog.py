import streamlit as st
from pydub import AudioSegment
import numpy as np
import scipy.io.wavfile as wavfile
import noisereduce as nr
import speech_recognition as sr
import torch
from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline

from transformers import MarianMTModel, MarianTokenizer
from langdetect import detect
import torch
import warnings

# Ignore all warnings
warnings.filterwarnings('ignore')

def translate_text(text: str, tgt_lang: str, temperature: float, top_k: int, top_p: float) -> str:
    """
    Translates text from the source language to the target language using a pre-trained model.

    Args:
        text (str): The text to translate.
        src_lang (str): The source language code (e.g., 'en' for English).
        tgt_lang (str): The target language code (e.g., 'fr' for French).

    Returns:
        str: The translated text.
    """
    # Find src_lang
    src_lang = detect(text)

    # Load the tokenizer and model
    model_name = f'Helsinki-NLP/opus-mt-{src_lang}-{tgt_lang}'
    tokenizer = MarianTokenizer.from_pretrained(model_name)
    model = MarianMTModel.from_pretrained(model_name)

    # Tokenize the input text
    input_ids = tokenizer(text, return_tensors="pt").input_ids

    # Generate the translation
    translated = model.generate(input_ids, max_length=300, num_return_sequences=1, temperature=temperature, top_k=top_k, top_p=top_p)

    # Decode the translated tokens and return the result
    tgt_text = tokenizer.decode(translated[0], skip_special_tokens=True)
    return tgt_text

def translate_conversation(text: str)-> str:
    lines = text.strip().split('\n')
    translated_lines = []

    for line in lines:
        translated_text = translate_text(line, tgt_lang='id', temperature=0.7, top_k=30, top_p=0.70)
        translated_lines.append(translated_text)

    translated_conversation = '\n'.join(translated_lines)
    return translated_conversation.replace('\n', ' ')

def load_audio(file_path):
    audio = AudioSegment.from_wav(file_path)
    return audio

def preprocess_audio(audio):
    temp_file = "temp.wav"
    audio.export(temp_file, format="wav")
    rate, data = wavfile.read(temp_file)
    reduced_noise = nr.reduce_noise(y=data, sr=rate)
    normalized_data = np.int16((reduced_noise / reduced_noise.max()) * 32767)
    preprocessed_file = "preprocessed_temp.wav"
    wavfile.write(preprocessed_file, rate, normalized_data)
    return preprocessed_file

def recognize_speech(file_path):
    recognizer = sr.Recognizer()
    with sr.AudioFile(file_path) as source:
        audio_data = recognizer.record(source)
    try:
        text = recognizer.recognize_google(audio_data)
        return text
    except sr.RequestError:
        return "API unavailable/unresponsive"
    except sr.UnknownValueError:
        return "Unable to recognize speech"
        
def speech_to_text(audio_file_path):
    audio = load_audio(audio_file_path)
    processed_audio_file = preprocess_audio(audio)
    text = recognize_speech(processed_audio_file)
    return text

def punctuate_text(text):
    tokenizer = AutoTokenizer.from_pretrained("oliverguhr/fullstop-punctuation-multilang-large")
    model = AutoModelForTokenClassification.from_pretrained("oliverguhr/fullstop-punctuation-multilang-large")

    text = recognized_text

    inputs = tokenizer(text, return_tensors="pt")
    tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])

    with torch.no_grad():
        outputs = model(**inputs).logits

    predictions = torch.argmax(outputs, dim=2).squeeze().tolist()

    punctuation_map = {0: "", 1: ".", 2: ",", 3: "?", 4: "!"}

    punctuated_text = ""
    for token, prediction in zip(tokens, predictions):
        if token.startswith("▁"):
            token = token[1:]
            punctuated_text += " "
        punctuated_text += token
        punctuated_text += punctuation_map.get(prediction, "")
    punctuated_text = punctuated_text.strip()

    return punctuated_text

import re

def clean_text(text):
    # Remove special tokens like <s> and </s>
    text = re.sub(r'<.*?>', '', text)

    # Remove extra spaces around punctuation marks
    text = re.sub(r'\s+([?.!,])', r'\1', text)

    # Fix misplaced periods within words
    text = re.sub(r'\b(\w+)\.(\w+)\b', r'\1\2', text)
    text = re.sub(r'\b(\w+),(\w+)\b', r'\1\2', text)

    # Replace multiple spaces with a single space
    text = re.sub(r'\s+', ' ', text).strip()

    # Ensure proper spacing after periods, commas, etc.
    text = re.sub(r'([?.!,])([^\s])', r'\1 \2', text)

    # Correct common punctuation issues
    text = text.replace('..', '.').replace(',,', ',')
    text = text.replace(',.', ',').replace('.,', '.')
    text = text.replace(' !', '!').replace(' ?', '?')

    # Capitalize the first letter of each sentence
    sentences = re.split(r'([.!?]\s*)', text)
    sentences = [s.capitalize() if i % 2 == 0 else s for i, s in enumerate(sentences)]
    text = ''.join(sentences)

    # Handle specific conjunction-like structures
    text = re.sub(r'\band\b', 'and', text)
    text = re.sub(r'\bbut\b', 'but', text)

    # Final correction for double spaces and edge cases
    text = re.sub(r'\s+', ' ', text).strip()

    # # Add '\n' at the end of each sentence
    # text = re.sub(r'([.!?])', r'\1\n', text)

    return text

def summarize_text(text):
    summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
    summary = summarizer(text, max_length=512, min_length=40, do_sample=False)
    summary_text = summary[0]['summary_text'] 

    summary_text = re.sub(r'([.!?])', r'\1\n', summary_text)
    return summary_text

st.title('Speech to Text Application')

uploaded_file = st.file_uploader("Choose an audio file", type="wav")
if uploaded_file is not None:
    audio_file_path = f"temp_audio.wav"
    with open(audio_file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    st.audio(uploaded_file, format='audio/wav')

    recognized_text = speech_to_text(audio_file_path)
    st.write("Recognized Text:", recognized_text)

    # punctuated_text = punctuate_text(recognized_text)
    # cleaned_text = clean_text(punctuated_text)
    # st.text_area("Punctuated Text:", cleaned_text, height=200)

    summary = summarize_text(recognized_text)
    summary_tr=translate_conversation(summary)

    st.text_area("Summary:", summary_tr, height=200)