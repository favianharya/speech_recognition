# Speech-to-Text Pipeline Project

## Introduction

This project focuses on implementing a practical application that integrates speech-to-text, summarization, and translation functionalities. The goal is to create a working solution that effectively processes spoken language from audio files, provides a concise summary, and translates the summarized text into the target language.

## Pipeline Workflow

The pipeline consists of three main stages:

1. **Speech-to-Text (WHISPER)**
   - **Description:** Converts audio files into written text using the WHISPER model, known for its high accuracy in speech recognition.
   - **Input:** Audio file (e.g., `.wav`, `.mp3`)
   - **Output:** Transcribed text
   
   **Process:**
   - The audio file is processed by the WHISPER model.
   - The model outputs a text representation of the spoken content.

2. **Summarization (BART)**
   - **Description:** Generates a concise summary of the transcribed text using the BART model summarization techniques.
   - **Input:** Transcribed text
   - **Output:** Summarized text
   
   **Process:**
   - The transcribed text is fed into the summarization model.
   - The model generates a summary that captures the key points and essential information.

3. **Translation**
   - **Description:** Translates the summarized text into the target language using a translation model.
   - **Input:** Summarized text
   - **Output:** Translated text
   
   **Process:**
   - The summarized text is translated into the desired language.
   - The final translated text is produced for further use.

## How to download Youtube's Audio

1. Put youtube's URL in ```youtube_links.txt```
2. Run the script in terminal ```python download_youtube.py```
3. The audio file will appear in folder ```audio_raw```

## How to chunk Youtube's Audio

1. Specify the ```chunk_length_ms``` (1000 = 1s) in ```chunk_audio.py```
2. Run the script in terminal ```python chunk_audio.py```
3. The raw audio file in ```audio_raw``` will automatically chunk and appear in ```output_chunk``` folder

## Authors

This project was developed by the following students from the Data Science program at Binus University:

- **Favian Harya Nandana Adrian** favian.adrian@binus.ac.id
- **Narendra Nusantara Handradika**  narendra.handradika@binus.ac.id
- **Ravael Daffa Hardjodipuro** ravael.hardjodipuro@binus.ac.id

**Note:** This document provides an overview of the project's objectives and methodology. For technical details or implementation specifics, please refer to the project repository or contact the authors directly.