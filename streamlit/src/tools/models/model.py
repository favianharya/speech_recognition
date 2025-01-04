from faster_whisper import WhisperModel
from tools.utils import logger
from transformers import pipeline

import warnings
warnings.filterwarnings("ignore")

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

class Model:
    def __init__(
            self, 
            model_type: str
            ):
        """
        Initialize the model with a specified type.

        Args:
            model_type (str): Type of Whisper model. Options include "medium", "medium.en", "small", "small.en", 
                              "base", "base.en", "tiny.en", "tiny".
        """
        self.model_type = model_type

    def load_model(self) -> WhisperModel:
        """
        Load the WhisperModel for speech recognition.

        Returns:
            WhisperModel: Loaded model for speech recognition.
        """
        return WhisperModel(self.model_type, device="cpu", compute_type="int8")
    
    def transcribe(self, file_path: str) -> str:
        """
        Transcribe text from speech.

        Args:
            file_path (str): File path of the audio media.

        Returns:
            str: Joined string from the transcribed speech.
        """
        model = self.load_model()
        segments, info = model.transcribe(file_path, beam_size=5, temperature=0.2)

        logger.info(f"🔨 Detected language: {info.language} (Probability: {info.language_probability:.2f})")

        text_segments = []
        for segment in segments:
            logger.info(f"✅ [{segment.start:.2f}s -> {segment.end:.2f}s] {segment.text}")
            text_segments.append(segment.text)
        
        return ''.join(text_segments)
    
    @staticmethod
    def calculate_summary_lengths(text: str) -> tuple:
        """
        Calculate scalable max_length and min_length for text summarization.

        Args:
            text (str): The text to be summarized.

        Returns:
            tuple: A tuple containing max_length and min_length.
        """
        text_length = len(text)

        # Define scaling factors
        max_scaling_factor = 0.5  # max_length will be 20% of the text length
        min_scaling_factor = 0.1  # min_length will be 10% of the text length

        # Calculate lengths
        max_length = int(max(text_length * max_scaling_factor, 50))  # Ensure at least 50 characters
        min_length = int(max(max_length * min_scaling_factor, 20))  # Ensure at least 20 characters

        # Apply upper limit for extremely long texts
        max_length = min(max_length, 200)  # Cap max_length at 200 characters
        min_length = min(min_length, 50)  # Cap min_length at 50 characters

        return max_length, min_length

    def summarize_text(self, text: str) -> str:
        """
        Summarize text from speech recognition's transcription.

        Args:
            text (str): String from the speech recognition's transcription.

        Returns:
            str: Summarized text.
        """
        summarizer = pipeline(
            "summarization",
            model="google/pegasus-xsum",
            tokenizer="google/pegasus-xsum",
            use_fast=False
        )
        
        max_length, min_length = self.calculate_summary_lengths(text)

        logger.info(f"Summarizing text with max_length={max_length}, min_length={min_length}")

        summary = summarizer(text, max_length=max_length, min_length=min_length, do_sample=False)
        return summary[0]['summary_text']
