import os
from pydub import AudioSegment

def to_camel_case(s):
    parts = s.split()
    return parts[0].lower() + ''.join(word.capitalize() for word in parts[1:])

def chunk_audio(file_path, output_folder):
    # Load the audio file
    audio = AudioSegment.from_file(file_path)

    chunk_length_ms =  30000 # 1000 = 1ms

    audio_length_ms = len(audio)

    base_filename = to_camel_case(os.path.splitext(os.path.basename(file_path))[0])

    # Calculate the number of chunks
    num_chunks = audio_length_ms // chunk_length_ms
    remainder_ms = audio_length_ms % chunk_length_ms

    # Create output directory if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)

    # Create chunks
    chunks = [audio[i * chunk_length_ms:(i + 1) * chunk_length_ms] for i in range(num_chunks)]

    # Export chunks to the new folder
    for i, chunk in enumerate(chunks):
        chunk.export(os.path.join(output_folder, f"{base_filename}_chunk_{i}.mp3"), format="mp3")
        
    if remainder_ms > 0:
        remainder_chunk = audio[num_chunks * chunk_length_ms:]
        remainder_chunk.export(os.path.join(output_folder, f"{base_filename}_chunk_{num_chunks}.mp3"), format="mp3")

def process_all_audios(input_folder, output_folder):
    # Ensure output folder exists
    os.makedirs(output_folder, exist_ok=True)

    # Process each audio file in the input folder
    for filename in os.listdir(input_folder):
        if filename.endswith(".mp3"):  # Modify if you have other audio formats
            file_path = os.path.join(input_folder, filename)

            # Create a folder for the current audio's chunks
            audio_name = os.path.splitext(filename)[0]
            audio_output_folder = os.path.join(output_folder, audio_name)
            os.makedirs(audio_output_folder, exist_ok=True)

            # Chunk the audio and save to the corresponding output folder
            chunk_audio(file_path, audio_output_folder)

# Example usage
input_folder = "/Users/t-favian.adrian/Documents/speech_recognition/audio_raw"        
output_folder = "/Users/t-favian.adrian/Documents/speech_recognition/output_chunk"  
process_all_audios(input_folder, output_folder)
