import streamlit as st
import whisper
import tempfile
from transformers import pipeline

import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

state = dict(result=0)


class Interface:
    def __init__(self):
        super().__init__()

    def get_header(self, title: str, description: str) -> None:
        """
        Display the header of the application.
        """
        st.set_page_config(
            page_title="Speech Recognition",
            page_icon="🗣️",
        )

        hide_decoration_bar_style = """<style>header {visibility: hidden;}</style>"""
        st.markdown(hide_decoration_bar_style, unsafe_allow_html=True)
        hide_streamlit_footer = """
        <style>#MainMenu {visibility: hidden;}
        footer {visibility: hidden;}</style>
        """
        st.markdown(hide_streamlit_footer, unsafe_allow_html=True)

        st.title(title)
        st.markdown(description)
        st.write("\n")

    def input_file(self) -> str:
        """
        Upload an audio file for transcription and summarization.
        """
        uploaded_file = st.file_uploader(
            "Choose an audio file",
            type=["wav", "mp3", "ogg"],
            help="Upload an audio file for transcription and summarization.",
        )
        if uploaded_file is not None:
            st.audio(uploaded_file, format="audio/wav")
        return uploaded_file

    def input_model(self) -> str:
        """
        Load the Whisper model for speech recognition.
        """

        model_type = st.selectbox(
            "Model type",
            (
                "turbo",
            ),
            index=0,
            help="Select the model size for speech recognition.",
        )

        return model_type


class Generation:
    def __init__(self):
        pass

    def _load_model(self, model_type: str) -> whisper:
        """
        Load the Whisper model for speech recognition.
        """
        return whisper.load_model(model_type)

    def _transcribe_model(self, file_path: str, model_type: str) -> str:
        """
        Transcribe speech from an audio file.
        """
        model = self._load_model(model_type)
        result = model.transcribe(file_path, beam_size=5, temperature=0.2)

        text_placeholder = st.empty()
        show_text = "" 

        text = []
        for segment in result["segments"]: 
            line = f"[{segment['start']:.2f}s -> {segment['end']:.2f}s] {segment['text']}"
            text.append(segment["text"])
            show_text += line + "\n"  # Append new line
            text_placeholder.text_area("Transcription:", show_text, height=300)

        return " ".join(text)

    def run_transcribe_model(self, file_path: str, model_type: str, state: dict) -> str:
        """
        Run the transcribe model function.
        """
        transcription = self._transcribe_model(file_path, model_type)
        state["transcription"] = transcription  # Store in state
        st.text_area("Transcription:", transcription, height=300)

    @staticmethod
    def _summarize_text(text: str) -> str:
        """
        summarize text from speech recognition's transcribe
        """
        summarizer = pipeline(
            "summarization",
            model="google/pegasus-xsum",
            tokenizer="google/pegasus-xsum",
            use_fast=False,
        )
        max_length = (
            len(text) // 5
        )  # Use integer division to ensure max_length is an integer
        min_length = (
            max_length // 4
        )  # Use integer division to ensure min_length is an integer

        summary = summarizer(
            text, max_length=max_length, min_length=min_length, do_sample=False
        )

        return summary[0]["summary_text"]

    def run_summarize_text(self, state: dict) -> str:
        """
        Run the summarize text function.
        """
        if "transcription" not in state:
            st.error("No transcription found. Please transcribe first.")
            return

        summary = self._summarize_text(
            state["transcription"]
        )  # Use state transcription
        st.text_area("Summary:", summary, height=300)


class Utils:
    @staticmethod
    def temporary_file(uploaded_file: str) -> str:
        """
        Create a temporary file for the uploaded audio file.
        """
        if uploaded_file is not None:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as temp_file:
                temp_file.write(uploaded_file.read())
                temp_file_path = temp_file.name
            return temp_file_path

    @staticmethod
    def process_step(state, step, title, process_function):
        """
        Handles the transcription and summarization steps dynamically.
        """
        if state.get("result") == step:
            with st.expander(title, expanded=True):
                st.subheader(f"{title} Result")
                with st.spinner("Processing... ⏳"):
                    generation = Generation()
                    process_function(generation)
                    state["result"] = step + 1  # Move to the next step


def main():
    interface = Interface()
    utils = Utils()
    interface.get_header(
        "🗣️ Speech Recognition", "Upload an audio file and transcribe it using Whisper."
    )

    with st.expander("Input file", expanded=True):
        uploaded_file = interface.input_file()

    with st.expander("Generation", expanded=True):
        model_type = interface.input_model()
        if (
            st.button("Generate Result !!", icon="🚀", type="primary")
            and uploaded_file is not None
        ):
            temp_file_path = utils.temporary_file(uploaded_file)
            state["temp_file_path"] = temp_file_path
            state["model_type"] = model_type
            state["result"] = 1

    utils.process_step(
        state,
        1,
        "Transcription",
        lambda gen: gen.run_transcribe_model(
            state["temp_file_path"], state["model_type"], state
        ),
    )
    utils.process_step(
        state, 2, "Summarization", lambda gen: gen.run_summarize_text(state)
    )


if __name__ == "__main__":
    main()
