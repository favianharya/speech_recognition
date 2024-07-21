from faster_whisper import WhisperModel
model = WhisperModel("medium", device="cpu", compute_type="int8")
segments, info = model.transcribe("/Users/t-favian.adrian/Documents/data_science_project/speech-to-text/Speaker26_000.wav", beam_size=5, temperature=0.2)

print("Detected language '%s' with probability %f" % (info.language, info.language_probability))

text = []
for segment in segments:
    print("[%.2fs -> %.2fs] %s" % (segment.start, segment.end, segment.text))
    text.append(segment.text)

print('\n')
joined_string = ''.join(text)
print(joined_string)