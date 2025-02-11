#####################################################################################################################################################
################################################################### PREPARE STUFF ###################################################################
#####################################################################################################################################################

# Imports
import os
import argparse
from transformers import pipeline
from sentence_transformers import SentenceTransformer, util

# Parse aguments
parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, help='Path to the dataset', default="/Brain/public/datasets/metal/data")
parser.add_argument('--output_directory', type=str, help='Path to the output directory', default="/Brain/public/datasets/metal/output")
parser.add_argument('--pipeline_arguments', type=dict, help='Extra arguments for the pipeline', default={"language": "english"})
parser.add_argument('--models', type=list, help='List of models to evaluate', default=["openai/whisper-large-v2"])
parser.add_argument('--extract_lyrics', type=bool, help='Extract lyrics or use precomputed ones', default=False)
args = parser.parse_args()

# List available files
all_file_names = [os.path.join(source, file[:file.rfind('.')]) for source in os.listdir(os.path.join(args.dataset, "audio")) for file in os.listdir(os.path.join(os.path.join(args.dataset, "audio"), source))]

# To set manually to a portion of the dataset
#all_file_names = ["songs/bloodbath___like_fire"]

# Functions to get full path to audio/lyrics file
def get_audio(file_name_no_extension):
    for file in os.listdir(os.path.join(args.dataset, "audio", *file_name_no_extension.split(os.path.sep)[:-1])):
        if file.startswith(file_name_no_extension.split(os.path.sep)[-1]):
            return os.path.join(args.dataset, "audio", *file_name_no_extension.split(os.path.sep)[:-1], file)
    raise Exception(f"Audio file not found for {file_name_no_extension}")
def get_lyrics(file_name_no_extension):
    for file in os.listdir(os.path.join(args.dataset, "lyrics", *file_name_no_extension.split(os.path.sep)[:-1])):
        if file.startswith(file_name_no_extension.split(os.path.sep)[-1]):
            return os.path.join(args.dataset, "lyrics", *file_name_no_extension.split(os.path.sep)[:-1], file)
    raise Exception(f"Lyrics file not found for {file_name_no_extension}")

#####################################################################################################################################################
################################################################# LYRICS EXTRACTION #################################################################
#####################################################################################################################################################

# We may run the section or load precomputed results
if args.extract_lyrics:

    # Extract lyrics from all audio files using the models
    for model in args.models:

        # Model pipeline for ASR
        pipe = pipeline(model=model, torch_dtype="auto", return_timestamps=True)
        for file_name in all_file_names:
            os.makedirs(os.path.join(args.output_directory, model, *file_name.split(os.path.sep)[:-1]), exist_ok=True)
            out = pipe(get_audio(file_name), generate_kwargs=args.pipeline_arguments)
            with open(os.path.join(args.output_directory, model, file_name + ".txt"), "w") as f:
                f.write(out["text"])

#####################################################################################################################################################
############################################################### TRANSCRIPTS SIMILARITY ##############################################################
#####################################################################################################################################################

# Load results from files
found_lyrics = {}
for model in args.models:
    found_lyrics[model] = {}
    for file_name in all_file_names:
        with open(os.path.join(args.output_directory, model, file_name + ".txt"), "r") as f:
            found_lyrics[model][file_name] = f.read()

# Load ground truth lyrics
ground_truth_lyrics = {}
for file_name in all_file_names:
    with open(get_lyrics(file_name), "r") as f:
        ground_truth_lyrics[file_name] = f.read()

# Load model
sentence_transformer = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

# Compute similarity between found and ground truth lyrics
for model in args.models:
    print(f"Similarity between lyrics found by {model} and actual lyrics:")
    for file_name in all_file_names:
        embedding_1 = sentence_transformer.encode(found_lyrics[model][file_name], convert_to_tensor=True)
        embedding_2 = sentence_transformer.encode(ground_truth_lyrics[file_name], convert_to_tensor=True)
        similarity = util.pytorch_cos_sim(embedding_1, embedding_2).item()
        print(f"  - {file_name}: {similarity}")