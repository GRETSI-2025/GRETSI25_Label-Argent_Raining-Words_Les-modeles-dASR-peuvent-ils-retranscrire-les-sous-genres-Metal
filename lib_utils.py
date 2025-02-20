#####################################################################################################################################################
################################################################### PREPARE STUFF ###################################################################
#####################################################################################################################################################

# External imports
import os
from pathlib import Path

# Project imports
from arguments import args

#####################################################################################################################################################
##################################################################### FUNCTIONS #####################################################################
#####################################################################################################################################################

def print_title (title, size=100, character="#"):

    # Print centered title
    print("", flush=True)
    print("", flush=True)
    print(character * size, flush=True)
    print(character + " " * ((size - len(title) - 1) // 2) + title + " " * ((size - len(title) - 2) // 2) + character, flush=True)
    print(character * size, flush=True)
    print("", flush=True)

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
#####################################################################################################################################################