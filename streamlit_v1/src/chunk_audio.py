import os
from pydub import AudioSegment
import re

def format_filename(input_string, chunk_number=0):
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

def chunk_audio(file_path, output_folder):
    # Load the audio file
    audio = AudioSegment.from_file(file_path)

    chunk_length_ms =  30000 # 1000 = 1s
    audio_length_ms = len(audio)

    base_filename = format_filename(os.path.splitext(os.path.basename(file_path))[0])

    # Calculate the number of chunks
    num_chunks = audio_length_ms // chunk_length_ms
    remainder_ms = audio_length_ms % chunk_length_ms

    # Create output directory if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)

    # Create chunks
    chunks = [audio[i * chunk_length_ms:(i + 1) * chunk_length_ms] for i in range(num_chunks)]

    # Export chunks to the new folder
    for i, chunk in enumerate(chunks):
        chunk.export(os.path.join(output_folder, f"{base_filename}_chunk_{i}.wav"), format="wav")
        
    if remainder_ms > 0:
        remainder_chunk = audio[num_chunks * chunk_length_ms:]
        remainder_chunk.export(os.path.join(output_folder, f"{base_filename}_chunk_{num_chunks}.wav"), format="wav")

def process_all_audios(input_folder, output_folder):
    # Ensure output folder exists
    os.makedirs(output_folder, exist_ok=True)

    # Process each audio file in the input folder
    for filename in os.listdir(input_folder):
        if filename.endswith(".wav"):  # Modify if you have other audio formats
            file_path = os.path.join(input_folder, filename)

            # Create a folder for the current audio's chunks
            audio_name = os.path.splitext(filename)[0]
            audio_output_folder = os.path.join(output_folder, audio_name)
            os.makedirs(audio_output_folder, exist_ok=True)

            # Chunk the audio and save to the corresponding output folder
            chunk_audio(file_path, audio_output_folder)

# Example usage
process_all_audios("audio_raw", "output_chunk" )
