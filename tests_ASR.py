#####################################################################################################################################################
################################################################### PREPARE STUFF ###################################################################
#####################################################################################################################################################

# Imports
import os
import sys
import argparse
from pathlib import Path
from transformers import pipeline
from huggingface_hub import snapshot_download
import pandas
from evaluate import load

# Prepare parser
parser = argparse.ArgumentParser()

# Path arguments
parser.add_argument('--dataset', type=str, help='Path to the dataset', default="/Brain/public/datasets/metal/data")
parser.add_argument('--output_directory', type=str, help='Path to the output directory', default="/Brain/public/datasets/metal/output")
parser.add_argument('--models_directory', type=str, help='Path to where models are downloaded', default="/Brain/public/models")

# Models to use
parser.add_argument('--asr_models', type=list, help='List of models to evaluate', default=["openai/whisper-large-v2",
                                                                                           "openai/whisper-large-v3"])
parser.add_argument('--similarity_metrics', type=list, help='Metrics or models used for computing similarity', default=["WER",
                                                                                                                        "Alibaba-NLP/gte-Qwen2-1.5B-instruct",
                                                                                                                        "sentence-transformers/all-MiniLM-L6-v2",
                                                                                                                        "sentence-transformers/all-mpnet-base-v2"])

# Control parts of the script to run or not
parser.add_argument('--extract_lyrics', type=bool, help='Extract lyrics or use precomputed ones', default=True)

# Go
args = parser.parse_args()

#####################################################################################################################################################
################################################################## USEFUL FUNCTIONS #################################################################
#####################################################################################################################################################

def get_pipeline (task, model_name):

    """
        Load a model pipeline from the Hugging Face Hub.
        The model is downloaded if not already present in the models directory.
        The pipeline is stored in global memory to avoid reloading if calling the function multiple times.
        :param model_name: The name of the model to load.
        :param task: The task of the pipeline.
        :return: The pipeline for the given task with the given model.
    """

    # Check if the pipeline is already in global memory to avoid reloading
    if "loaded_models" not in globals():
        globals()["loaded_models"] = {}
    global_model_key = task + "_" + model_name
    if global_model_key in globals()["loaded_models"]:
        return globals()["loaded_models"][global_model_key]

    # Download the model if not already downloaded
    model_path = os.path.join(args.models_directory, model_name)
    if not os.path.exists(model_path):
        print(f"Downloading model {model_name} to {model_path}", file=sys.stderr, flush=True)
        snapshot_download(repo_id=model_name, local_dir=model_path)
        os.system(f"chmod 777 -R {model_path}")

    # Load the pipeline
    print(f"Loading pipeline with model {model_name} for {task}", file=sys.stderr, flush=True)
    globals()["loaded_models"][global_model_key] = pipeline(task, model=model_path)
    return globals()["loaded_models"][global_model_key]

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

def compute_similarities (predicted_lyrics, actual_lyrics):

    """
        Compute various similarities between two sets of lyrics.
        :param predicted_lyrics: The predicted lyrics.
        :param actual_lyrics: The reference lyrics.
        :return: The similarities between the two sets of lyrics.
    """

    # Function to compute similarity with Sentence Transformers
    def _compute_similarity_with_sentence_transformer (st_model):
        pipe = get_pipeline("feature-extraction", st_model)
        embedding_actual = pipe(actual_lyrics, return_tensors=True)[0].mean(dim=0)
        embedding_predicted = pipe(predicted_lyrics, return_tensors=True)[0].mean(dim=0)
        return float(embedding_actual @ embedding_predicted / (embedding_actual.norm() * embedding_predicted.norm()))
    
    # Function to compute similarity with Word Error Rate
    def _compute_similarity_with_word_error_rate ():
        wer = load("wer")
        error = wer.compute(predictions=[actual_lyrics], references=[predicted_lyrics])
        return 1 - error
    
    # Return a dictionary of similarities
    similarities = {}
    for metric in args.similarity_metrics:
        if "/" in metric:
            similarities[metric] = _compute_similarity_with_sentence_transformer(metric)
        elif metric == "WER":
            similarities["WER"] = _compute_similarity_with_word_error_rate()
    return similarities

#####################################################################################################################################################
####################################################################### SCRIPT ######################################################################
#####################################################################################################################################################

###################################
########## PREPARE STUFF ##########
###################################

# Get the list of all file names to work on
all_file_names = list_from_source()
print(f"Loaded files {all_file_names}", file=sys.stderr, flush=True)

###################################
########## EXTRACT LYRICS #########
###################################

# We may run the section or load precomputed results
if args.extract_lyrics:

    # Extract lyrics from all audio files using the models
    for asr_model in args.asr_models:

        # Remove previous results if any
        results_path = os.path.join(args.output_directory, asr_model.replace(os.path.sep, "-") + ".ods")
        if os.path.exists(results_path):
            os.remove(results_path)

        # Load ASR pipeline
        pipe = get_pipeline("automatic-speech-recognition", asr_model)

        # One sheet per source
        for source in all_file_names:
            
            # Model pipeline for ASR
            data = {"File": [], "Lyrics": []}
            for file_name in all_file_names[source]:
                out = pipe(get_audio(source, file_name), return_timestamps=True, generate_kwargs={"language": "english"})
                data["File"].append(file_name)
                data["Lyrics"].append(out["text"])

            # Save results to file
            file = pandas.read_excel(results_path, engine="odf", sheet_name=None) if os.path.exists(results_path) else {}
            file[source.replace(os.path.sep, "___")] = pandas.DataFrame(data)
            with pandas.ExcelWriter(results_path, engine="odf") as writer:
                for sheet_name, sheet in file.items():
                    sheet.to_excel(writer, sheet_name=sheet_name, index=False)

###################################
###### COMPUTE SIMILARITIES #######
###################################

# Write results to output file
output_file_name = os.path.join(args.output_directory, "similarities.txt")
with open(output_file_name, "w") as output_file:

    # First group by source
    for source in sorted(all_file_names):
        print(f"[SOURCE] {source}", file=output_file, flush=True)

        # Then by file
        for file_name in sorted(all_file_names[source]):
            print(f"|__ [FILE] {file_name}", file=output_file, flush=True)

            # Then by ASR model
            for asr_model in args.asr_models:
                print(f"|   |__ [MODEL] {asr_model}", file=output_file, flush=True)

                # Load lyrics
                actual_lyrics = get_lyrics(os.path.join(args.dataset, "lyrics.ods"), source, file_name)
                found_lyrics = get_lyrics(os.path.join(args.output_directory, asr_model.replace(os.path.sep, "-") + ".ods"), source, file_name)["Lyrics"]

                # Compute similarities
                similarities = {}
                for key in actual_lyrics:
                    similarities[key] = compute_similarities(actual_lyrics[key], found_lyrics)
                    
                # Report results
                for sim_metric in list(similarities.values())[0]:
                    all_sim_measures = [similarities[key][sim_metric] for key in similarities]
                    print(f"|   |   |__ [METRIC] {sim_metric} -- max({all_sim_measures}) = {max(all_sim_measures)}", file=output_file, flush=True)