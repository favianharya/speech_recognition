import whisper
import librosa
import soundfile as sf
import tempfile
from pydub import AudioSegment
from pydub.silence import split_on_silence

import warnings
warnings.simplefilter("ignore", category=UserWarning)


def chunk_audio_smart(file_path, min_silence_len=500, silence_thresh=-40):
    """
    Splits an audio file into chunks using silence detection.
    
    :param file_path: Path to the audio file
    :param min_silence_len: Minimum silence duration (ms) to consider as a split point
    :param silence_thresh: Silence threshold in dB (lower means more aggressive silence detection)
    :return: List of chunk file paths
    """
    audio = AudioSegment.from_file(file_path, format="wav")
    
    # Split audio where silence is detected
    chunks = split_on_silence(
        audio, 
        min_silence_len=min_silence_len, 
        silence_thresh=silence_thresh,
        keep_silence=200  # Keep some silence at the edges to avoid harsh cuts
    )

    return chunks

def transcribe_audio_chunks(file_path, model_type="turbo", min_silence_len=500, silence_thresh=-40):
    """
    Transcribes an audio file in dynamically split chunks using temporary files.

    :param file_path: Path to the audio file
    :param model_type: Whisper model type (tiny, base, small, medium, large)
    :param min_silence_len: Minimum silence duration (ms) for splitting
    :param silence_thresh: Silence threshold in dB
    :return: Tuple (formatted transcript with timestamps, concatenated full text)
    """
    model = whisper.load_model(model_type)  # Load Whisper model
    chunks = chunk_audio_smart(file_path, min_silence_len, silence_thresh)  # Get dynamically chunked audio
    
    transcript = []
    full_text = []
    start_time = 0.0  # Track the timestamp
    
    for idx, chunk in enumerate(chunks):
        # Create a temporary file to store the audio chunk
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=True) as temp_file:
            chunk.export(temp_file.name, format="wav")  # Save chunk to temp file
            
            result = model.transcribe(temp_file.name)  # Transcribe the chunk
            text = result["text"]
            end_time = start_time + len(chunk) / 1000.0  # Convert ms to seconds

            transcript.append(f"[{start_time:.2f}s -> {end_time:.2f}s] {text}")
            print(f"[{start_time:.2f}s -> {end_time:.2f}s] {text}")  # Add timestamped text
            full_text.append(text)  # Store transcribed text for concatenation

        start_time = end_time  # Update timestamp
    
    return "\n".join(transcript), " ".join(full_text)  # Return both formatted and concatenated text

# Usage
file_path = "supply_and_demand_explained_in_one_minute.wav"
model = whisper.load_model('turbo')  # Load Whisper model
audio = AudioSegment.from_file(file_path)  # Load audio
# If the audio is short, transcribe it as a whole
if len(audio) / 1000 < 10:
    result = model.transcribe(file_path)
    print(result["text"])

formatted_transcript, concatenated_text = transcribe_audio_chunks(file_path)

print("Formatted Transcript:\n", formatted_transcript)
print("\nConcatenated Text:\n", concatenated_text)
