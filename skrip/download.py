import yt_dlp
import os

def download_youtube_video_as_mp3(youtube_url, output_path):
    try:
        ydl_opts = {
            'format': 'bestaudio/best',
            'postprocessors': [{
                'key': 'FFmpegExtractAudio',
                'preferredcodec': 'mp3',
                'preferredquality': '192',
            }],
            'outtmpl': os.path.join(output_path, '%(title)s.%(ext)s'),
        }

        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([youtube_url])
            print(f"Downloaded and converted to MP3: {youtube_url}")

    except Exception as e:
        print(f"Failed to download {youtube_url}: {e}")

def download_mp3_from_file(file_path, output_path):
    with open(file_path, 'r') as file:
        urls = file.readlines()

    for url in urls:
        url = url.strip()  # Remove any leading/trailing whitespace or newline characters
        if url:  # Check if the line is not empty
            download_youtube_video_as_mp3(url, output_path)

# Example usage
download_mp3_from_file("/Users/t-favian.adrian/Documents/speech_recognition/youtube_links.txt", "/Users/t-favian.adrian/Documents/speech_recognition/output")
