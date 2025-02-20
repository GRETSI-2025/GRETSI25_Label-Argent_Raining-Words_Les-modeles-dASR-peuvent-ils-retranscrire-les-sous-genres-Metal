#####################################################################################################################################################
################################################################### PREPARE STUFF ###################################################################
#####################################################################################################################################################

# External imports
import os
import pandas
import yt_dlp
import itertools
import demucs.separate
from pathlib import Path

# Project imports
from arguments import args

#####################################################################################################################################################
##################################################################### FUNCTIONS #####################################################################
#####################################################################################################################################################

def list_from_source (source=None):

    # List required sources
    source_path = os.path.join(args().dataset, "audio") if source is None else os.path.join(args().dataset, "audio", source)
    file_names = [str(file.relative_to(os.path.join(args().dataset, "audio"))) for file in Path(source_path).rglob("*") if file.is_file()]
    actual_sources = list(set(file_name[:file_name.rfind(os.path.sep)] for file_name in file_names))
    return {s: [file_name[file_name.rfind(os.path.sep)+1:file_name.rfind(".")] for file_name in file_names if file_name.startswith(s)] for s in actual_sources}
    
#####################################################################################################################################################

def get_audio_file (source, file_name_no_extension):

    # Search for the audio file in the source
    for file in os.listdir(os.path.join(args().dataset, "audio", source)):
        if file.startswith(file_name_no_extension):
            return os.path.join(args().dataset, "audio", source, file)
    
    # Raise exception if the audio file is not found
    raise Exception(f"Audio file not found for {file_name_no_extension}")

#####################################################################################################################################################

def normalize_lyrics (lyrics):

    # Remove capitals and special characters
    lyrics = lyrics.lower()
    lyrics = [char for char in lyrics if char.isalnum() or char.isspace()]

    # Remove simple artifacts such as having twice the same word consecutively
    lyrics = [key for key, _group in itertools.groupby(lyrics)]
    return "".join(lyrics)

#####################################################################################################################################################

def get_lyrics (lyrics_file, source, file_name_no_extension, memoize=True):

    # Check if the file is already in global memory to avoid reloading
    if memoize:
        memoization_key = lyrics_file
        if "loaded_files" not in globals():
            globals()["loaded_files"] = {}
        if lyrics_file not in globals()["loaded_files"]:
            globals()["loaded_files"][memoization_key] = pandas.read_excel(lyrics_file, engine="odf", sheet_name=None)
        file = globals()["loaded_files"][memoization_key]
    else:
        file = pandas.read_excel(lyrics_file, engine="odf", sheet_name=None)
    
    # Get row containing sheet name
    sheet_name = source.replace(os.path.sep, "___")
    row = file[sheet_name].loc[file[sheet_name]["File"] == file_name_no_extension]
    if not row.empty:
        return {key: normalize_lyrics(row[key].values[0]) for key in row.keys() if key.startswith("Lyrics")}

    # Raise exception if the lyrics are not found
    raise Exception(f"Lyrics not found for {file_name_no_extension}")
 
#####################################################################################################################################################

def download_audio (url, source, file_name, force_dl=False):

    # Create directory if it does not exist
    target_directory = os.path.join(args().dataset, "audio", "songs", source)
    if not os.path.exists(target_directory):
        os.makedirs(target_directory)

    # Ignore if the file already exists
    print(f"Downloading \"{file_name}\" in source \"songs/{source}\"")
    target_file = os.path.join(target_directory, file_name)
    if not os.path.exists(target_file + ".wav") or force_dl:

        # Download as .m4a and then convert to .wav file
        ydl_opts = {"format": "m4a/bestaudio/best",
                    "outtmpl": target_file,
                    "force_keyframes_at_cuts": True,
                    "postprocessors": [{"key": "FFmpegExtractAudio", "preferredcodec": "wav"}]}

        # Get the file
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([url])

#####################################################################################################################################################

def extract_singer (source, file_name, force_extract=False):

    # Create directory if it does not exist
    target_directory = os.path.join(args().dataset, "audio", "demucs", source)
    if not os.path.exists(target_directory):
        os.makedirs(target_directory)

    # Check if the file is already in global memory to avoid reloading
    print(f"Extracting singer from \"{file_name}\" into source \"demucs/{source}\"")
    target_file = os.path.join(target_directory, file_name) + ".wav"
    if not os.path.exists(target_file) or force_extract:
        
        # Extract singer from the audio file
        source_file = os.path.join(args().dataset, "audio", "songs", source, file_name) + ".wav"
        demucs.separate.main(["--two-stems", "vocals", "-n", "htdemucs_ft", source_file, "-o", target_file])

#####################################################################################################################################################
#####################################################################################################################################################