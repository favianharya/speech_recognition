from faster_whisper import WhisperModel
import streamlit as st
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline
from transformers import MarianMTModel, MarianTokenizer
from langdetect import detect
import re
import torch
import sentencepiece

from script.download import download_youtube_video_as_mp3
from script.eval_summ import rouge_eval

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

    st.write("Detected language '%s' with probability %f" % (info.language, info.language_probability))

    text = []
    for segment in segments:
        st.write("[%.2fs -> %.2fs] %s" % (segment.start, segment.end, segment.text))
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
    summarizer = pipeline(
            "summarization",
            model="google/pegasus-xsum",
            tokenizer="google/pegasus-xsum",
            use_fast=False
        )
    max_length = len(text) // 5  # Use integer division to ensure max_length is an integer
    min_length = max_length // 4  # Use integer division to ensure min_length is an integer

    summary = summarizer(text, max_length=max_length, min_length=min_length, do_sample=False)
    summary_text = summary[0]['summary_text'] 

    # summary_text = re.sub(r'([.!?])', r'\1\n', summary_text)
    return summary_text

test = 0

def main():

    hide_decoration_bar_style = '''<style>header {visibility: hidden;}</style>'''
    st.markdown(hide_decoration_bar_style, unsafe_allow_html=True)
    # Design hide "made with streamlit" footer menu area
    hide_streamlit_footer = """
    <style>#MainMenu {visibility: hidden;}
    footer {visibility: hidden;}</style>
    """
    st.markdown(hide_streamlit_footer, unsafe_allow_html=True)

    st.title("📺 YouTube video content extraction")
    st.markdown('🚀 **Welcome to the YouTube Video Content Extraction** 📹✨ – Unlock insights from videos like never before! 🔍')



    with st.expander("Criterias", expanded=True):

        model = st.selectbox("Choose model type", ["large","medium", "medium.en", "small", "small.en", "base", "base.en", "tiny.en", "tiny"])
        url = st.text_input("Enter the YouTube video URL")

        download_path = "audio_temp/audio"
        try:
            if st.button('Generate Summarization', icon="🚀", type="primary"):
                with st.spinner():
                    if 'https://youtu.be/' in url:
                        test = download_youtube_video_as_mp3(url, download_path)
                    else:
                        raise ValueError("String must input url")

                    st.video(url) 

                    # Check if the file was downloaded and exists
                    if os.path.exists("audio_temp/audio.wav"):
                        st.audio("audio_temp/audio.wav", format="audio/wav")

                        audio_file_path = "audio_temp/audio.wav"

                        progress_text = "Operation in progress. Please wait."
                        my_bar = st.progress(0, text=progress_text)

                        my_bar.progress(10, text="Transcribing audio...")
                        recognized_text = transcribe_model(audio_file_path, model)

                        st.text_area("Recognized Text:", recognized_text, height=200)

                        my_bar.progress(60, text="summarize...")
                        summary = summarize_text(recognized_text)

                        my_bar.progress(80, text="Summarization result...")
                        st.text_area("Summary:", summary, height=200)

                        my_bar.progress(100, text="Operation complete...")
                        eval = rouge_eval(summary, recognized_text)
                        st.text_area("Evaluation Result:", eval, height=100)
                    else:
                        raise ValueError("unable to download video data")

                    if os.path.exists(audio_file_path):
                        os.remove(audio_file_path)
                        print(f"The file {audio_file_path} has been deleted.")
                    else:
                        print(f"The file {audio_file_path} does not exist.")

        except Exception as e:
                st.error(f"An error occured: {e}", icon="🚨")           

if __name__ == "__main__":
    main()