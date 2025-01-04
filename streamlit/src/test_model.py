from tools.models import Model

speech_1 = "/Users/t-favian.adrian/Documents/speech_recognition/src/audio_raw/the_federal_funds_rate_explained_in_one_minute_federal_reserve_interest_rate_superpower_or_threat__chunk_0.wav"

def main(speech: str) -> str:
    # Initialize the Model class
    models = Model(model_type="base")

    # Transcribe the audio
    gas = models.transcribe(speech)

    test = models.summarize_text(gas)

    return test

if __name__ == "__main__":
    result = main(speech_1)
    print(result)
