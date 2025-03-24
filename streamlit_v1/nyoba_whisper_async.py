import asyncio
import time  # For timing each segment
import statistics  # For calculating the median
from concurrent.futures import ThreadPoolExecutor
from faster_whisper import WhisperModel
from tqdm import tqdm  # Importing tqdm for the progress bar

# Function to transcribe audio file (synchronous task to be run asynchronously)
def transcribe_audio(model, file_path, beam_size, temperature):
    start_time = time.time()
    
    # Transcribing the entire audio at once
    segments, info = model.transcribe(file_path, beam_size=beam_size, temperature=temperature)
    
    segments = list(segments)
    
    end_time = time.time()
    total_transcription_time = end_time - start_time
    
    return segments, info, total_transcription_time

async def process_segments(segments, total_transcription_time):
    text = []
    proportional_times = []
    total_audio_duration = segments[-1].end  # The end time of the last segment gives total audio duration

    # Display tqdm progress bar initially but only update it as segments are processed
    with tqdm(total=len(segments), desc="Processing segments", unit="segment") as pbar:
        for segment in segments:
            segment_duration = segment.end - segment.start
            proportional_time = (segment_duration / total_audio_duration) * total_transcription_time
            proportional_times.append(proportional_time)

            # Append the segment's text
            text.append(segment.text)

            # Print the segment and its proportional time
            print(f"[{segment.start:.2f}s -> {segment.end:.2f}s] {segment.text} (Generated in {proportional_time:.2f} seconds)")
            
            # Update the progress bar
            pbar.update(1)
            await asyncio.sleep(0)  # Allow for other asyncio tasks

    # Join all segment text into one string
    joined_string = ''.join(text)
    return joined_string, proportional_times

# Asynchronous wrapper to execute the transcription in a separate thread
async def async_transcribe(model, file_path, beam_size=5, temperature=0.2):
    print("Starting transcription... Please wait.")  # Loading message at the start
    
    loop = asyncio.get_event_loop()
    with ThreadPoolExecutor() as pool:
        segments, info, total_transcription_time = await loop.run_in_executor(
            pool, transcribe_audio, model, file_path, beam_size, temperature
        )
    
    print("Detected language '%s' with probability %f" % (info.language, info.language_probability))
    
    # Process each segment and estimate the time for each based on total transcription time
    transcription, proportional_times = await process_segments(segments, total_transcription_time)
    
    # Print the total transcription time
    print(f"\nTotal transcription time: {total_transcription_time:.2f} seconds")

    # Calculate and print the median time
    median_time = statistics.median(proportional_times)
    print(f"Median time for segment generation: {median_time:.2f} seconds")

    return transcription

# Main asynchronous function
async def main():
    # Initialize the Whisper model
    model = WhisperModel("medium", device="cpu", compute_type="int8")
    
    # Transcribe audio asynchronously and time each segment generation
    transcription = await async_transcribe(model, "/Users/t-favian.adrian/Documents/data_science_project/speech-to-text/Speaker26_000.wav")
    
    # Print the final transcription
    print("\nFull Transcription:\n", transcription)

# Run the async main function
if __name__ == "__main__":
    asyncio.run(main())
