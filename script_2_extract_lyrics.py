#####################################################################################################################################################
################################################################### PREPARE STUFF ###################################################################
#####################################################################################################################################################

# External imports
import os
import sys
import pandas

# Project imports
from arguments import args
import lib_audio
import lib_models

#####################################################################################################################################################
######################################################################### GO ########################################################################
#####################################################################################################################################################

# Get the list of all file names to work on
all_file_names = lib_audio.list_from_dataset()

# Extract lyrics from all audio files using the models
for asr_model in args().asr_models:

    # Load existing lyrics from file
    lyrics_file_name = os.path.join(args().output_directory, "data", asr_model.replace(os.path.sep, "-") + ".ods")
    if args().force_lyrics_extraction or not os.path.exists(lyrics_file_name):
        all_lyrics = {}
    else:
        dataframes = pandas.read_excel(lyrics_file_name, engine="odf", sheet_name=None)
        all_lyrics = {sheet: dataframes[sheet].to_dict(orient="list") for sheet in dataframes}

    # One sheet per dataset
    for dataset in all_file_names:
        dataset_sheet = dataset.replace(os.path.sep, "___")
        if dataset_sheet not in all_lyrics:
            all_lyrics[dataset_sheet] = {"File": [], "Lyrics": []}
        
        # One line per file
        for file_name in all_file_names[dataset]:
            if file_name not in all_lyrics[dataset_sheet]["File"]:

                # Go through ASR pipeline
                print(f"Extracting lyrics for \"{file_name}\" with model \"{asr_model}\"", file=sys.stderr, flush=True)
                model = lib_models.get_model(asr_model)
                transcription = model.transcribe(lib_audio.get_audio_path(dataset, file_name))
                all_lyrics[dataset_sheet]["File"].append(file_name)
                all_lyrics[dataset_sheet]["Lyrics"].append(transcription)
                print(f"Transcription: {transcription}", file=sys.stderr, flush=True)
            
    # Save results to file
    with pandas.ExcelWriter(lyrics_file_name, engine="odf") as writer:
        for sheet_name, sheet in all_lyrics.items():
            pandas.DataFrame(sheet).to_excel(writer, sheet_name=sheet_name, index=False)
    os.chmod(lyrics_file_name, 0o777)
    
#####################################################################################################################################################
#####################################################################################################################################################