#####################################################################################################################################################
###################################################################### LICENSE ######################################################################
#####################################################################################################################################################
#
#    Copyright (C) 2025  Bastien Pasdeloup & Axel Marmoret
#
#    This program is free software: you can redistribute it and/or modify
#    it under the terms of the GNU General Public License as published by
#    the Free Software Foundation, either version 3 of the License, or
#    (at your option) any later version.
#
#    This program is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#    GNU General Public License for more details.
#
#    You should have received a copy of the GNU General Public License
#    along with this program.  If not, see <https://www.gnu.org/licenses/>.
#
#####################################################################################################################################################
################################################################### DOCUMENTATION ###################################################################
#####################################################################################################################################################

"""
    This module contains the functions to handle audio files, lyrics, paths to audio files, downloading them, etc.
"""

#####################################################################################################################################################
################################################################### PREPARE STUFF ###################################################################
#####################################################################################################################################################

# External imports
import os
import sys
import pandas
import torchaudio
import torch
from pathlib import Path
from typing import *
import yt_dlp
import soundfile

# Project imports
from lib.arguments import script_args
import lib.models.loader

#####################################################################################################################################################
##################################################################### FUNCTIONS #####################################################################
#####################################################################################################################################################

def list_from_dataset ( dataset: str = None
                      ) ->       dict[str, list[str]]:

    """
        Function to list the files in a dataset.
        A dataset is a subdirectory of the "audio" directory.
        Files will be organized by subdirectories in the output.
        E.g., if dataset="songs": {"songs/death": [...], "songs/black": [...], ...}.
        In:
            * dataset: The dataset to list (list all datasets if None).
        Out:
            * A dictionary containing the files in the dataset (without their extension).
    """

    # Find actual datasets
    dataset_path = os.path.join(script_args().datasets_path, "audio") if dataset is None else os.path.join(script_args().datasets_path, "audio", dataset)
    file_names = [str(file.relative_to(os.path.join(script_args().datasets_path, "audio"))) for file in Path(dataset_path).rglob("*") if file.is_file()]
    actual_datasets = list(set(file_name[:file_name.rfind(os.path.sep)] for file_name in file_names))

    # List files per dataset
    return {s: [file_name[file_name.rfind(os.path.sep)+1:file_name.rfind(".")] for file_name in file_names if file_name[:file_name.rfind(os.path.sep)] == s] for s in actual_datasets}
    
#####################################################################################################################################################

def get_audio_path ( dataset:                str,
                     file_name_no_extension: str
                   ) ->                      str:

    """
        Function to get the path to an audio file.
        This function allows not to specify the extension of the file in case extensions vary.
        In:
            * dataset:                The dataset containing the audio file.
            * file_name_no_extension: The name of the audio file without extension.
        Out:
            * The path to the audio file.
    """

    # Search for the audio file in the dataset
    for file in os.listdir(os.path.join(script_args().datasets_path, "audio", dataset)):
        if file.startswith(file_name_no_extension):
            return os.path.join(script_args().datasets_path, "audio", dataset, file)
    
    # Raise exception if the audio file is not found
    raise Exception(f"Audio file not found for {file_name_no_extension}")

#####################################################################################################################################################

def load_audio ( audio_path: str,
                 resample:   Optional[int] = None,
                 to_mono:    bool = False,
                 engine:     str = "torchaudio"
               ) ->          torch.Tensor:

    """
        Function to load an audio file.
        In:
            * audio_path: The path to the audio file.
            * resample:   The frequency to resample the audio file to.
            * to_mono:    Whether to convert the audio file to mono.
            * engine:     The engine to use to load the audio file.
        Out:
            * The audio file as a tensor, and its sampling rate.
    """

    # If using Torchaudio
    if engine == "torchaudio":
        
        # Get audio file
        format = audio_path.split(".")[-1]
        audio, sampling_rate = torchaudio.load(audio_path, format=format)

        # Resample and convert to mono if needed
        if resample:
            audio = torchaudio.functional.resample(audio, orig_freq=sampling_rate, new_freq=resample)
            sampling_rate = resample
        if to_mono:
            audio = audio.mean(dim=0, keepdim=True)
        return audio, sampling_rate

    # If using Soundfile
    elif engine == "soundfile":

        # Get audio file, already resampled and converted to mono if needed
        channels = 1 if to_mono else None
        audio, sampling_rate = soundfile.read(audio_path, samplerate=resample, channels=channels)
        audio = torch.tensor(audio)
        return audio, sampling_rate
    
    # Raise exception if the engine is not recognized
    raise Exception(f"Engine {engine} not recognized")

#####################################################################################################################################################

def normalize_lyrics ( lyrics: str
                     ) ->      str:

    """
        Function to normalize lyrics.
        In:
            * lyrics: The lyrics to normalize.
        Out:
            * The normalized lyrics.
    """

    # Remove capitals and special characters
    lyrics = lyrics.lower()
    lyrics = "".join([char for char in lyrics if char.isalnum() or char.isspace() or char in ["'", "-"]])

    # Remove extra spaces
    words = lyrics.split()

    # Remove words too long to be actual words
    words = [word for word in words if len(word) < 25]

    # Remove duplicate words
    lyrics = [words[i] for i in range(len(words)) if i == 0 or words[i] != words[i-1]]
    lyrics = " ".join(lyrics)
    return lyrics

#####################################################################################################################################################

def get_lyrics ( lyrics_file:            str,
                 dataset:                str,
                 file_name_no_extension: str,
                 memoize:                bool = True
               ) ->                      dict[str, str]:

    """
        Function to get the lyrics of a song.
        In:
            * lyrics_file:            The path to the file containing the lyrics.
            * dataset:                The dataset containing the lyrics.
            * file_name_no_extension: The name of the file without extension.
            * memoize:                Whether to memoize the file in memory.
        Out:
            * The lyrics of the song.
    """

    # Factorize stuff to do when loading a file
    def _load_file ():
        file = pandas.read_excel(lyrics_file, engine="odf", sheet_name=None, dtype=str)
        for sheet in file:
            file[sheet] = file[sheet].fillna("")
        return file

    # Check if the file is already in global memory to avoid reloading
    if memoize:
        memoization_key = str(lyrics_file)
        if "loaded_files" not in globals():
            globals()["loaded_files"] = {}
        if lyrics_file not in globals()["loaded_files"]:
            globals()["loaded_files"][memoization_key] = _load_file()
        file = globals()["loaded_files"][memoization_key]
    else:
        file = _load_file()
    
    # Get row containing sheet name
    sheet_name = dataset.replace(os.path.sep, "___")
    row_key = file_name_no_extension.replace("–", "-").lower()
    row = file[sheet_name].loc[file[sheet_name]["File"].str.replace("–", "-").str.lower() == row_key]
    if not row.empty:
        return {key: normalize_lyrics(row[key].values[0]) for key in row.keys() if key.startswith("Lyrics")}

    # Raise exception if the lyrics are not found
    raise Exception(f"Lyrics not found for {file_name_no_extension}")
 
#####################################################################################################################################################

def download_audio ( url:              str,
                     target_file_path: str
                   ) ->                None:
    """
        Function to download an audio file.
        In:
            * url:              The URL of the audio file.
            * target_file_name: The name of the target file.
        Out:
            * None.
    """

    # Ignore if the file already exists
    if not os.path.exists(target_file_path):

        # Download as .m4a and then convert to .wav file
        ydl_opts = {"format": "m4a/bestaudio/best",
                    "outtmpl": target_file_path[:target_file_path.rfind(".")],
                    "force_keyframes_at_cuts": True,
                    "postprocessors": [{"key": "FFmpegExtractAudio", "preferredcodec": "wav"}]}

        # Get the file
        print(f"Downloading \"{target_file_path}\"", file=sys.stderr, flush=True)
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([url])
        os.chmod(target_file_path, 0o777)

#####################################################################################################################################################

def extract_vocals ( source_file_path: str,
                     target_file_path: str,
                     model_name:       Union[str, list[str], tuple[str]] = ("Demucs", "mdx_extra")
                   ) ->                None:

    """
        Function to extract vocals from an audio file.
        In:
            * source_file_path: The path to the source audio file.
            * target_file_path: The path to the target audio file.
            * model_name:       The name of the model to use.
        Out:
            * None.
    """

    # Check if the file is already in global memory to avoid reloading
    if not os.path.exists(target_file_path):
        
        # Extract vocals from the audio file
        print(f"Extracting vocals from \"{source_file_path}\"", file=sys.stderr, flush=True)
        model = lib.models.loader.get_model(model_name)
        model.run(source_file_path, target_file_path)
        os.chmod(target_file_path, 0o777)

#####################################################################################################################################################
#####################################################################################################################################################