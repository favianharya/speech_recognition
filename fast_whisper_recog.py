from faster_whisper import WhisperModel
import streamlit as st
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline
from transformers import MarianMTModel, MarianTokenizer
from langdetect import detect
import re
import torch

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import time


def load_model(type:str):
    """
    Translates text from the source language to the target language using a pre-trained model.

    Args:
        type (str): type of model "medium", "medium.en", "small", "small.en", "base", "base.en", "tiny.en", "tiny".

    Returns:
        model: WhisperModel for speech recognition.
    """
    return WhisperModel(type, device="cpu", compute_type="int8")

def transcribe_model(file_path:str, type:str) -> str:
    """
    Transcribe text from speech.

    Args:
        file_path (str): file path of the audio media.

    Returns:
        str: joined str fron the transcribe speech.
    """
    model = load_model(type)
    segments, info = model.transcribe(file_path, beam_size=5, temperature=0.2)

    print("Detected language '%s' with probability %f" % (info.language, info.language_probability))

    text = []
    for segment in segments:
        print("[%.2fs -> %.2fs] %s" % (segment.start, segment.end, segment.text))
        text.append(segment.text)
    
    return ''.join(text)

def summarize_text(text:str) -> str:
    """
    summarize text from speech recognition's transcribe

    Args:
        text (str): string from the speech recognition's transcribe.

    Returns:
        str: summarize text
    """
    summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
    max_length = len(text) // 5  # Use integer division to ensure max_length is an integer
    min_length = max_length // 4  # Use integer division to ensure min_length is an integer

    summary = summarizer(text, max_length=max_length, min_length=min_length, do_sample=False)
    summary_text = summary[0]['summary_text'] 

    # summary_text = re.sub(r'([.!?])', r'\1\n', summary_text)
    return summary_text

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


def main():

    st.title('Speech to Text Application')
    model = st.selectbox("Choose translation language", ["medium", "medium.en", "small", "small.en", "base", "base.en", "tiny.en", "tiny"])
    uploaded_file = st.file_uploader("Choose an audio file", type="wav")
    if uploaded_file is not None:
        audio_file_path = f"temp_audio.wav"
        with open(audio_file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        st.audio(uploaded_file, format='audio/wav')
    
        progress_text = "Operation in progress. Please wait."
        my_bar = st.progress(0, text=progress_text)

        my_bar.progress(10, text="Transcribing audio...")
        recognized_text = transcribe_model(audio_file_path, model)

        st.write("Recognized Text:", recognized_text)

        my_bar.progress(60, text="Translating summary...")
        summary = summarize_text(recognized_text)

        my_bar.progress(100, text="Operation complete.")
        summary_tr=translate_conversation(summary)

        st.text_area("Summary:", summary_tr, height=200)

if __name__ == "__main__":
    main()