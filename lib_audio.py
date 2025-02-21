#####################################################################################################################################################
################################################################### PREPARE STUFF ###################################################################
#####################################################################################################################################################

# External imports
import os
import sys
import pandas
import torchaudio
import demucs.separate
from pathlib import Path
import yt_dlp
import torch.serialization
import shutil
from demucs.hdemucs import HDemucs

# Project imports
from arguments import args

#####################################################################################################################################################
##################################################################### FUNCTIONS #####################################################################
#####################################################################################################################################################

def list_from_dataset (dataset=None):

    # List subdirectories of the dataset
    dataset_path = os.path.join(args().datasets_path, "audio") if dataset is None else os.path.join(args().datasets_path, "audio", dataset)
    file_names = [str(file.relative_to(os.path.join(args().datasets_path, "audio"))) for file in Path(dataset_path).rglob("*") if file.is_file()]
    actual_datasets = list(set(file_name[:file_name.rfind(os.path.sep)] for file_name in file_names))
    return {s: [file_name[file_name.rfind(os.path.sep)+1:file_name.rfind(".")] for file_name in file_names if file_name.startswith(s)] for s in actual_datasets}
    
#####################################################################################################################################################

def get_audio_path (dataset, file_name_no_extension):

    # Search for the audio file in the dataset
    for file in os.listdir(os.path.join(args().datasets_path, "audio", dataset)):
        if file.startswith(file_name_no_extension):
            return os.path.join(args().datasets_path, "audio", dataset, file)
    
    # Raise exception if the audio file is not found
    raise Exception(f"Audio file not found for {file_name_no_extension}")

#####################################################################################################################################################

def load_audio (audio_path, resample=None, to_mono=False):

    # Get audio file
    audio, sampling_rate = torchaudio.load(audio_path, format="wav")
    
    # Resample if needed
    if resample:
        audio = torchaudio.functional.resample(audio, orig_freq=sampling_rate, new_freq=resample)
    
    # Convert to mono if needed
    if to_mono:
        audio = audio.mean(dim=0, keepdim=True)
        
    # Return audio
    return audio

#####################################################################################################################################################

def normalize_lyrics (lyrics):

    # Remove capitals and special characters
    lyrics = lyrics.lower()
    lyrics = "".join([char for char in lyrics if char.isalnum() or char.isspace() or char in ["'", "-"]])

    # Remove extra spaces
    words = lyrics.split()

    # Remove simple artifacts such as having twice the same word consecutively
    lyrics = [words[i] for i in range(len(words)) if i == 0 or words[i] != words[i-1]]
    lyrics = " ".join(lyrics)
    return lyrics

#####################################################################################################################################################

def get_lyrics (lyrics_file, dataset, file_name_no_extension, memoize=True):

    # Factorize stuff to do when loading a file
    def _load_file ():
        file = pandas.read_excel(lyrics_file, engine="odf", sheet_name=None, dtype=str)
        for sheet in file:
            file[sheet] = file[sheet].fillna("")
        return file

    # Check if the file is already in global memory to avoid reloading
    if memoize:
        memoization_key = lyrics_file
        if "loaded_files" not in globals():
            globals()["loaded_files"] = {}
        if lyrics_file not in globals()["loaded_files"]:
            globals()["loaded_files"][memoization_key] = _load_file()
        file = globals()["loaded_files"][memoization_key]
    else:
        file = _load_file()
    
    # Get row containing sheet name
    sheet_name = dataset.replace(os.path.sep, "___")
    row = file[sheet_name].loc[file[sheet_name]["File"] == file_name_no_extension]
    if not row.empty:
        return {key: normalize_lyrics(row[key].values[0]) for key in row.keys() if key.startswith("Lyrics")}

    # Raise exception if the lyrics are not found
    raise Exception(f"Lyrics not found for {file_name_no_extension}")
 
#####################################################################################################################################################

def download_audio (url, style, file_name, force_dl=False):

    # Create directory if it does not exist
    target_directory = os.path.join(args().datasets_path, "audio", "songs", style)
    os.makedirs(target_directory, exist_ok=True)
    os.chmod(target_directory, 0o777)

    # Ignore if the file already exists
    target_file = os.path.join(target_directory, file_name)
    if not os.path.exists(target_file + ".wav") or force_dl:

        # Download as .m4a and then convert to .wav file
        ydl_opts = {"format": "m4a/bestaudio/best",
                    "outtmpl": target_file,
                    "force_keyframes_at_cuts": True,
                    "postprocessors": [{"key": "FFmpegExtractAudio", "preferredcodec": "wav"}]}

        # Get the file
        print(f"Downloading \"{file_name}\" in dataset \"songs/{style}\"", file=sys.stderr, flush=True)
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([url])
        os.chmod(target_file + ".wav", 0o777)

#####################################################################################################################################################

def extract_vocals (style, file_name, force_extract=False):

    # Create directory if it does not exist
    target_directory = os.path.join(args().datasets_path, "audio", "vocals", style)
    os.makedirs(target_directory, exist_ok=True)
    os.chmod(target_directory, 0o777)

    # Check if the file is already in global memory to avoid reloading
    target_file = os.path.join(target_directory, file_name) + ".wav"
    if not os.path.exists(target_file) or force_extract:
        
        # Extract vocals from the audio file
        # https://github.com/adefossez/demucs
        print(f"Extracting singer from \"{file_name}\" into dataset \"vocals/{style}\"", file=sys.stderr, flush=True)
        demucs_model = "mdx_extra"
        source_file = os.path.join(args().datasets_path, "audio", "songs", style, file_name) + ".wav"
        torch.serialization.add_safe_globals([HDemucs])
        demucs.separate.main(["--two-stems", "vocals", "--name", demucs_model, "--repo", os.path.join(args().models_directory, "demucs"), "-o", "separated", source_file])

        # Reorganize the output
        os.rename(os.path.join("separated", demucs_model, file_name, "vocals.wav"), target_file)
        shutil.rmtree("separated", ignore_errors=True)
        os.chmod(target_file, 0o777)

#####################################################################################################################################################
#####################################################################################################################################################