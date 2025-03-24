from faster_whisper import WhisperModel
from tqdm import tqdm  # Import tqdm for the progress bar

# Initialize the Whisper model
model = WhisperModel("medium", device="cpu", compute_type="int8")

# Transcribe the audio file
segments, info = model.transcribe("src/audio_raw/the_federal_funds_rate_explained_in_one_minute_federal_reserve_interest_rate_superpower_or_threat__chunk_0.wav", beam_size=5, temperature=0.2)

print("Detected language '%s' with probability %f" % (info.language, info.language_probability))

# Collect the transcribed text
text = []
# Wrap the segment processing loop with tqdm for a progress bar
for segment in tqdm(segments, desc="Processing segments", unit="segment"):
    print("[%.2fs -> %.2fs] %s" % (segment.start, segment.end, segment.text))
    text.append(segment.text)

# Join all segment text into one string
print('\n')
joined_string = ''.join(text)
print(joined_string)
