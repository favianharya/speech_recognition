import yt_dlp
import os

def download_youtube_video_as_mp3(youtube_url, output_path):
    """
    Download the audio from youtube.

    Args:
        youtube_url (str): youtube's video URL.
        output_path (str): path for output to be save.

    Returns:
        Audio: Audio from youtube's video.
    """
    try:
        ydl_opts = {
            'format': 'bestaudio/best',
            'postprocessors': [{
                'key': 'FFmpegExtractAudio',
                'preferredcodec': 'wav',
                'preferredquality': '192',
            }],
            'outtmpl': output_path,
        }

        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.cache.remove()
            ydl.download([youtube_url])
            print(f"Downloaded and converted to wav: {youtube_url}")

    except yt_dlp.DownloadError as error:
        pass