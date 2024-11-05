import yt_dlp
import os
import re

def format_filename(input_string, chunk_number=0):
    """
    Format the input string to a safe filename with lowercase and underscores.

    Args:
        input_string (str): The original title or string to format.
        chunk_number (int): Optional chunk number to append to the filename.

    Returns:
        str: The formatted filename.
    """
    # Remove leading and trailing whitespace
    input_string = input_string.strip()
    # Replace special characters with underscores
    formatted_string = re.sub(r'[^a-zA-Z0-9\s]', '_', input_string)
    # Replace multiple spaces or underscores with a single underscore
    formatted_string = re.sub(r'[\s_]+', '_', formatted_string)
    # Convert the string to lowercase
    formatted_string = formatted_string.lower()
    # Append the chunk identifier
    formatted_string += f'_chunk_{chunk_number}'
    return formatted_string

def download_youtube_video_as_mp3(youtube_url, output_path):
    """
    Download the audio from YouTube and save it with a formatted filename.

    Args:
        youtube_url (str): YouTube's video URL.
        output_path (str): Path for output to be saved.

    Returns:
         Audio: Audio from youtube's video.
    """
    try:
        # Fetch video info to get the title
        with yt_dlp.YoutubeDL({'quiet': True}) as ydl:
            info_dict = ydl.extract_info(youtube_url, download=False)
            original_title = info_dict.get('title', 'audio')
            formatted_title = format_filename(original_title)

        ydl_opts = {
            'format': 'bestaudio/best',
            'postprocessors': [{
                'key': 'FFmpegExtractAudio',
                'preferredcodec': 'wav',
                'preferredquality': '192',
            }],
            'outtmpl': os.path.join(output_path, f'{formatted_title}.%(ext)s'),
        }

        # Download and save with formatted title
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([youtube_url])
            print(f"Downloaded and converted to wav: {youtube_url} with title '{formatted_title}'")

    except Exception as e:
        print(f"Failed to download {youtube_url}: {e}")

def download_mp3_from_file(file_path, output_path):
    """
    To read a TXT file filled with YouTube URLs and download each as a formatted title WAV file.

    Args:
        file_path (str): Path to TXT file.
        output_path (str): Path for output to be saved.

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
