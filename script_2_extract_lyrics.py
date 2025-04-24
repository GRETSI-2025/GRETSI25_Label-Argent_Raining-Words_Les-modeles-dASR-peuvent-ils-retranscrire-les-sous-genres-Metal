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
    This script extracts lyrics from audio files using ASR models.
    The lyrics are saved in an ODS file under separate sheets for each dataset.
    Each sheet contains two columns: "File" and "Lyrics".
"""

#####################################################################################################################################################
################################################################### PREPARE STUFF ###################################################################
#####################################################################################################################################################

# External imports
import os
import sys
import pandas

# Project imports
from lib.arguments import script_args
import lib.audio
import lib.models.loader

#####################################################################################################################################################
######################################################################### GO ########################################################################
#####################################################################################################################################################

# Create output directory if it does not exist
data_directory = os.path.join(script_args().output_directory, "data")
os.makedirs(data_directory, exist_ok=True)
os.chmod(data_directory, 0o777)

# Get the list of all files to work on
all_files = lib.audio.list_from_dataset()

# Extract lyrics from all audio files using the models
for asr_model in set(script_args().asr_models_emvd + script_args().asr_models_songs):

    # Load existing lyrics from file
    print(f"Model: \"{asr_model}\"", file=sys.stdout, flush=True)
    lyrics_file_path = os.path.join(data_directory, asr_model.replace(os.path.sep, "-") + ".ods")
    if not os.path.exists(lyrics_file_path):
        all_lyrics = {}
    else:
        dataframes = pandas.read_excel(lyrics_file_path, engine="odf", sheet_name=None)
        all_lyrics = {sheet: dataframes[sheet].to_dict(orient="list") for sheet in dataframes}

    # One sheet per dataset
    for dataset in all_files:
        if (dataset.startswith("emvd") and asr_model not in script_args().asr_models_emvd) \
        or ((not dataset.startswith("emvd")) and asr_model not in script_args().asr_models_songs):
            continue
        print(f"    Dataset: \"{dataset}\"", file=sys.stdout, flush=True)
        dataset_sheet = dataset.replace(os.path.sep, "___")
        if dataset_sheet not in all_lyrics:
            all_lyrics[dataset_sheet] = {"File": [], "Lyrics": []}
        
        # One line per file
        for file_name in all_files[dataset]:
            if file_name not in all_lyrics[dataset_sheet]["File"]:

                # Go through ASR pipeline
                print(f"        File: \"{file_name}\"", file=sys.stdout, flush=True)
                model = lib.models.loader.get_model(asr_model)
                transcription = model.run(lib.audio.get_audio_path(dataset, file_name))
                all_lyrics[dataset_sheet]["File"].append(file_name)
                all_lyrics[dataset_sheet]["Lyrics"].append(transcription)
                print(f"            {transcription}", file=sys.stdout, flush=True)
            
    # Save results to file
    if len(all_lyrics) > 0:
        with pandas.ExcelWriter(lyrics_file_path, engine="odf") as writer:
            for sheet_name, sheet in all_lyrics.items():
                pandas.DataFrame(sheet).to_excel(writer, sheet_name=sheet_name, index=False)
        os.chmod(lyrics_file_path, 0o777)
    
    # Free up memory
    lib.models.loader.free_model(asr_model)

#####################################################################################################################################################
#####################################################################################################################################################