#####################################################################################################################################################
################################################################### PREPARE STUFF ###################################################################
#####################################################################################################################################################

# Imports
import os
import numpy
import argparse
from pathlib import Path
from transformers import pipeline
from sentence_transformers import SentenceTransformer, util
import pandas

# Parse aguments
parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, help='Path to the dataset', default="/Brain/public/datasets/metal/data")
parser.add_argument('--output_directory', type=str, help='Path to the output directory', default="/Brain/public/datasets/metal/output")
parser.add_argument('--pipeline_arguments', type=dict, help='Extra arguments for the pipeline', default={"language": "english"})
parser.add_argument('--models', type=list, help='List of models to evaluate', default=["openai/whisper-large-v2"])
parser.add_argument('--extract_lyrics', type=bool, help='Extract lyrics or use precomputed ones', default=True)
args = parser.parse_args()

#####################################################################################################################################################
################################################################## USEFUL FUNCTIONS #################################################################
#####################################################################################################################################################

def list_from_source (source=None):

    """
        List all the files in a given source directory.
        :param source: The source to list the files from (None to list all available sources).
        :return: A dictionary of file names with key being the source and values a list of files in the source with extension removed.
    """

    # List required sources
    source_path = os.path.join(args.dataset, "audio") if source is None else os.path.join(args.dataset, "audio", source)
    file_names = [str(file.relative_to(os.path.join(args.dataset, "audio"))) for file in Path(source_path).rglob("*") if file.is_file()]
    actual_sources = list(set(file_name[:file_name.rfind(os.path.sep)] for file_name in file_names))
    return {s: [file_name[file_name.rfind(os.path.sep)+1:file_name.rfind(".")] for file_name in file_names if file_name.startswith(s)] for s in actual_sources}
    
#####################################################################################################################################################

def get_audio (source, file_name_no_extension):

    """
        Get the full path of the audio file corresponding to a given file name.
        :param source: The source of the audio file.
        :param file_name_no_extension: The file name without extension.
        :return: The full path of the audio file.
    """

    # Search for the audio file in the source
    for file in os.listdir(os.path.join(args.dataset, "audio", source)):
        if file.startswith(file_name_no_extension):
            return os.path.join(args.dataset, "audio", source, file)
    
    # Raise exception if the audio file is not found
    raise Exception(f"Audio file not found for {file_name_no_extension}")

#####################################################################################################################################################

def normalize_lyrics (lyrics):

    """
        Normalize the lyrics of a song by removing capitals and special characters.
        :param lyrics: The lyrics to normalize.
        :return: The normalized lyrics.
    """

    # Remove capitals and special characters
    lyrics = lyrics.lower()
    return "".join([char for char in lyrics if char.isalnum() or char.isspace()])

#####################################################################################################################################################

def get_lyrics (lyrics_file, source, file_name_no_extension):

    """
        Return the lyrics of a song given its file name.
        :param lyrics_file: The file containing the lyrics.
        :param file_name_no_extension: The file name of the song without extension.
        :return: A dictionary of lyrics, with each value is a retranscription of the lyrics with column name as key.
    """

    # Lyrics are stored in .odt files
    sheet = pandas.read_excel(lyrics_file, engine="odf", sheet_name=source.replace(os.path.sep, "___"))
    
    # Search for the lyrics in the sheet
    lyrics = {}
    for index, row in sheet.iterrows():
        if row["File"] == file_name_no_extension:
            for column in sheet.columns:
                if column.startswith("Lyrics"):
                    lyrics[column] = normalize_lyrics(str(row[column]))
            return lyrics

    # Raise exception if the lyrics are not found
    raise Exception(f"Lyrics not found for {file_name_no_extension}")

#####################################################################################################################################################
####################################################################### SCRIPT ######################################################################
#####################################################################################################################################################

###################################
########## PREPARE STUFF ##########
###################################

# Get the list of all file names to work on
all_file_names = list_from_source()
print(f"Found files in sources: {all_file_names}")

###################################
########## EXTRACT LYRICS #########
###################################

# We may run the section or load precomputed results
if args.extract_lyrics:

    # Extract lyrics from all audio files using the models
    for model in args.models:

        # Remove previous results if any
        model_path = os.path.join(args.output_directory, model.replace(os.path.sep, "-") + ".ods")
        if os.path.exists(model_path):
            os.remove(model_path)

        # One sheet per source
        for source in all_file_names:
            
            # Model pipeline for ASR
            data = {"File": [], "Lyrics": []}
            pipe = pipeline(model=model, torch_dtype="auto", return_timestamps=True)
            for file_name in all_file_names[source]:
                out = pipe(get_audio(source, file_name), generate_kwargs=args.pipeline_arguments)
                data["File"].append(file_name)
                data["Lyrics"].append(out["text"])

            # Save results to file
            file = pandas.read_excel(model_path, engine="odf", sheet_name=None) if os.path.exists(model_path) else {}
            file[source.replace(os.path.sep, "___")] = pandas.DataFrame(data)
            with pandas.ExcelWriter(model_path, engine="odf") as writer:
                for sheet_name, sheet in file.items():
                    sheet.to_excel(writer, sheet_name=sheet_name, index=False)

###################################
###### COMPUTE SIMILARITIES #######
###################################

# Load model for sentence embeddings
sentence_transformer = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

# Compute similarity between found and ground truth lyrics
for source in all_file_names:
    for model in args.models:
        for file_name in all_file_names[source]:

            # Load lyrics
            actual_lyrics = get_lyrics(os.path.join(args.dataset, "lyrics.ods"), source, file_name)
            found_lyrics = get_lyrics(os.path.join(args.output_directory, model.replace(os.path.sep, "-") + ".ods"), source, file_name)["Lyrics"]

            # Compute embeddings
            embedding_found = sentence_transformer.encode(found_lyrics, convert_to_tensor=True)
            similarities = []
            for key in actual_lyrics:
                embedding_actual = sentence_transformer.encode(actual_lyrics[key], convert_to_tensor=True)

                # Compute similarity
                similarity = util.pytorch_cos_sim(embedding_found, embedding_actual).item()
                similarities.append(similarity)
                
            # Report results
            print(f"{model} \t\t {source} \t\t {file_name} \t\t {similarities} (mean {numpy.mean(similarities)}, std {numpy.std(similarities)})")