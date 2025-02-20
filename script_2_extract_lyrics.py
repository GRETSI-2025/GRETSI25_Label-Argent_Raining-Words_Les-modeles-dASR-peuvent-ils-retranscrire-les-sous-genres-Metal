#####################################################################################################################################################
################################################################### PREPARE STUFF ###################################################################
#####################################################################################################################################################

# External imports
import os
import sys
import pandas

# Project imports
from arguments import args
from lib_models import get_model
from lib_audio import list_from_source, get_audio_path

#####################################################################################################################################################
######################################################################### GO ########################################################################
#####################################################################################################################################################

# Get the list of all file names to work on
all_file_names = list_from_source(args().source)

# Make sure output subdirectory exists
output_directory = os.path.join(args().output_directory, "data")
os.makedirs(output_directory, exist_ok=True)

# Extract lyrics from all audio files using the models
for asr_model in args().asr_models:

    # Load existing lyrics from file
    lyrics_file_name = os.path.join(output_directory, asr_model.replace(os.path.sep, "-") + ".ods")
    if args().force_lyrics_extraction or not os.path.exists(lyrics_file_name):
        all_lyrics = {}
    else:
        dataframes = pandas.read_excel(lyrics_file_name, engine="odf", sheet_name=None)
        all_lyrics = {sheet: dataframes[sheet].to_dict(orient="list") for sheet in dataframes}

    # Load ASR model
    model = get_model(asr_model)

    # One sheet per source
    for source in all_file_names:
        source_sheet = source.replace(os.path.sep, "___")
        if source_sheet not in all_lyrics:
            all_lyrics[source_sheet] = {"File": [], "Lyrics": []}
        
        # One line per file
        for file_name in all_file_names[source]:
            if file_name not in all_lyrics[source_sheet]["File"]:

                # Go through ASR pipeline
                print(f"Extracting lyrics for \"{file_name}\" with model \"{asr_model}\"", file=sys.stderr, flush=True)
                transcription = model.transcribe(get_audio_path(source, file_name))
                all_lyrics[source_sheet]["File"].append(file_name)
                all_lyrics[source_sheet]["Lyrics"].append(transcription)
                print(file_name, "-", transcription, file=sys.stderr, flush=True)
            
    # Save results to file
    with pandas.ExcelWriter(lyrics_file_name, engine="odf") as writer:
        for sheet_name, sheet in all_lyrics.items():
            pandas.DataFrame(sheet).to_excel(writer, sheet_name=sheet_name, index=False)

    # Set permissions for shared use
    os.chmod(lyrics_file_name, 0o777)
    
#####################################################################################################################################################
#####################################################################################################################################################