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
            'outtmpl': os.path.join(output_path, '%(title)s.%(ext)s'),
        }

        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([youtube_url])
            print(f"Downloaded and converted to wav: {youtube_url}")

    except Exception as e:
        print(f"Failed to download {youtube_url}: {e}")

def download_mp3_from_file(file_path, output_path):
    """
    To read TXT file filled with youtube's URL and download it with 
    download_youtube_video_as_mp3()

    Args:
        file_path (str): TXT's path.
        output_path (str): path for output to be save.

    Returns:
        Audio: Audio from youtube's video.
    """
    with open(file_path, 'r') as file:
        urls = file.readlines()

    for url in urls:
        url = url.strip()  # Remove any leading/trailing whitespace or newline characters
        if url:  # Check if the line is not empty
            download_youtube_video_as_mp3(url, output_path)

def main():
    download_mp3_from_file("youtube_links.txt", "audio_raw")


if __name__ == "__main__":
    main()