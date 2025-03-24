from tools.utils import logger

import yt_dlp
import os
import re


class Downloader:
    def __init__(self):
        pass

    @staticmethod
    def format_filename(input_string: str) -> str:
        """
        Converts the input string into a safe, lowercase filename with underscores.

        Args:
            input_string (str): The original title or string to format.

        Returns:
            str: The formatted filename.
        """
        return re.sub(r'[\s_]+', '_', re.sub(r'[^a-zA-Z0-9\s]', '_', input_string.strip())).lower()

    @staticmethod
    def download_audio(youtube_url: str, output_path: str) -> None:
        """
        Downloads the audio from a YouTube video and saves it as a WAV file with a formatted filename.

        Args:
            youtube_url (str): URL of the YouTube video.
            output_path (str): Directory to save the output file.

        Returns:
            None
        """
        try:
            # Extract video information
            with yt_dlp.YoutubeDL({'quiet': True}) as ydl:
                video_info = ydl.extract_info(youtube_url, download=False)
                original_title = video_info.get('title', 'audio')
                formatted_title = Downloader.format_filename(original_title)

            # Set download options
            ydl_opts = {
                'format': 'bestaudio/best',
                'postprocessors': [{
                    'key': 'FFmpegExtractAudio',
                    'preferredcodec': 'wav',
                    'preferredquality': '192',
                }],
                'outtmpl': os.path.join(output_path, f'{formatted_title}.%(ext)s'),
            }

            # Download the audio
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                ydl.download([youtube_url])
                logger.info(f"✅ Downloaded and saved as WAV: {formatted_title}.wav in {output_path}")

        except Exception as e:
            logger.error(f"🚨 Error downloading audio: {e}")
