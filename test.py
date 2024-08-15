import yt_dlp
import os

def download_youtube_video_as_mp3(youtube_url):
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
        }

        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([youtube_url])
            print(f"Downloaded and converted to wav: {youtube_url}")

    except Exception as e:
        print(f"Failed to download {youtube_url}: {e}")

download_youtube_video_as_mp3('https://youtu.be/e68E-4UncnU?feature=shared')