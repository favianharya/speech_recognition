from tools.models import Model
from tools.utils import logger

import streamlit as st

def main():
    st.set_page_config(page_title="Extraction", page_icon="📺")
    hide_decoration_bar_style = '''<style>header {visibility: hidden;}</style>'''
    st.markdown(hide_decoration_bar_style, unsafe_allow_html=True)
    # Design hide "made with streamlit" footer menu area
    hide_streamlit_footer = """
        <style>#MainMenu {visibility: hidden;}
        footer {visibility: hidden;}</style>
    """
    st.markdown(hide_streamlit_footer, unsafe_allow_html=True)
    st.image('img/logo_socs.png', width=200)
    st.title("Finance Audio Content Extraction")
    st.markdown('🚀 **Welcome to the Finance Audio Content Extraction** ✨ – Unlock financial insights from financial audio like never before! 🔍')

    with st.expander("Criterias", expanded=False):
        model = st.selectbox("Choose transcribing model type", ["large","medium", "medium.en", "small", "small.en", "base", "base.en", "tiny.en", "tiny"])
        models = Model(model_type=model)
        audio_files = {
            "audios/supply_and_demand_explained_in_one_minute.wav": "Supply and Demand Explained in One Minute",
            "audios/subsidies_explained_in_one_minute.wav": "Subsidies Explained in One Minute"
        }

        # Streamlit selectbox for choosing an audio file
        selected_name = st.selectbox("Choose an audio file:", list(audio_files.values()))

        # Map the selected name back to the file path
        selected_audio = [path for path, name in audio_files.items() if name == selected_name][0]

        if st.button('Generate Extraction', icon="🚀", type="primary"):
                with st.spinner():
                    transcription = models.transcribe(selected_audio)
                    summarization = models.summarize_text(transcription)
                    st.text_area("Transcription", value=summarization, height=200)

if __name__ ==  "__main__":
    main()