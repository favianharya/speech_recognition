import tempfile
import os
import pandas as pd
import re
import whisper
from tqdm import tqdm 

import warnings  
warnings.filterwarnings("ignore")

def link_to_dataframe(folder_path: str) -> pd.DataFrame:
    """
    This function reads all the files in the folder and returns a dataframe with the file paths."
    """
    file_paths = [os.path.join(folder_path, file) for file in os.listdir(folder_path)]
    with tempfile.NamedTemporaryFile(
        mode="w", delete=False, suffix=".txt"
    ) as temp_file:
        for path in file_paths:
            temp_file.write(path + "\n")
        temp_path = temp_file.name  # Store the temporary file path

    df = pd.read_csv(temp_path, header=None, names=["audio_file_path"])
    os.remove(temp_path)

    return df

def transcribe(df:pd.DataFrame):
    model = whisper.load_model('turbo')

    results = []  # Store transcriptions

    for path in tqdm(df["audio_file_path"], desc="Transcribing", unit="file"):
        print(f"Transcribing: {path}")  # Optional: Show progress
        result = model.transcribe(path)
        results.append({"audio_file_path": path, "transcription": result["text"]}),
    
    return pd.DataFrame(results)

def main():
    df = link_to_dataframe('00a36b039cb49e6b99bb6d17635a3c25')  # Get DataFrame with file paths
    df['filename'] = df['audio_file_path'].str.extract(r"/(\d+)\.wav$").astype(int)
    df = df.sort_values(by='filename')
    print(df)

    transcriptions_df = transcribe(df)  # Transcribe audio files
    transcriptions_df.to_csv("text.csv")

if __name__ == "__main__":  
    main()