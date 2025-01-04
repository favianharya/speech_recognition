from tools.utils import Downloader

link_input = "https://youtu.be/vlm09G2mAg4?feature=shared"
link_saved = "/Users/t-favian.adrian/Documents/speech_recognition/streamlit/src/audios"

def main(link_input:str, link_saved:str) -> str:
    download = Downloader()

    result = download.download_audio(link_input,link_saved)

    return result

if __name__ == "__main__":
    main(link_input,link_saved)