#####################################################################################################################################################
################################################################### PREPARE STUFF ###################################################################
#####################################################################################################################################################

# External imports
import os
import pandas

#####################################################################################################################################################
##################################################################### FUNCTIONS #####################################################################
#####################################################################################################################################################

def normalize_lyrics (lyrics):

    # Remove capitals and special characters
    lyrics = lyrics.lower()
    return "".join([char for char in lyrics if char.isalnum() or char.isspace()])

#####################################################################################################################################################

def get_lyrics (lyrics_file, source, file_name_no_extension, memoize=True):

    # Check if the file is already in global memory to avoid reloading
    if memoize:
        if "loaded_files" not in globals():
            globals()["loaded_files"] = {}
        if lyrics_file not in globals()["loaded_files"]:
            globals()["loaded_files"][lyrics_file] = pandas.read_excel(lyrics_file, engine="odf", sheet_name=None)
        file = globals()["loaded_files"][lyrics_file]
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
#####################################################################################################################################################