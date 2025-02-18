#####################################################################################################################################################
################################################################### PREPARE STUFF ###################################################################
#####################################################################################################################################################

# Imports
import os
import sys
import argparse
from pathlib import Path
from transformers import pipeline
import huggingface_hub
import pandas
import evaluate
import pickle
import torch
import matplotlib.pyplot as pyplot

# Prepare parser
parser = argparse.ArgumentParser()

# Path arguments
parser.add_argument("--dataset", type=str, help="Path to the dataset", default="/Brain/public/datasets/metal/data")
parser.add_argument("--output_directory", type=str, help="Path to the output directory", default="/Brain/public/datasets/metal/output")
parser.add_argument("--models_directory", type=str, help="Path to where models are downloaded", default="/Brain/public/models")

# Credentials
parser.add_argument("--hf_key", type=str, help="Path to the Hugging Face token file", default=f"/Brain/private/{os.environ["LOGNAME"]}/misc/hugging_face.key")

# Models to use
parser.add_argument("--asr_models", type=list, help="List of models to evaluate", default=["openai/whisper-large-v2",
                                                                                           "openai/whisper-large-v3"])
parser.add_argument("--metrics", type=list, help="Metrics or models used for computing similarity/error", default=["WER",
                                                                                                                   "Alibaba-NLP/gte-Qwen2-1.5B-instruct",
                                                                                                                   "sentence-transformers/all-MiniLM-L6-v2",
                                                                                                                   "sentence-transformers/all-mpnet-base-v2"])

# Control parts of the script to run or not
parser.add_argument("--extract_lyrics", type=bool, help="Force re-extraction of lyrics from audio files", default=False)
parser.add_argument("--compute_metrics", type=bool, help="Force re-computation of metrics", default=False)

# Go
args = parser.parse_args()

#####################################################################################################################################################
################################################################## USEFUL FUNCTIONS #################################################################
#####################################################################################################################################################

def get_pipeline (task, model_name, memoize=True, **kwargs):

    """
        Load a model pipeline from the Hugging Face Hub.
        The model is downloaded if not already present in the models directory.
        If asked, the pipeline is stored in global memory to avoid reloading if calling the function multiple times.
        :param model_name: The name of the model to load.
        :param task: The task of the pipeline.
        :param memoize: Whether to store the pipeline in global memory.
        :param kwargs: Additional arguments to pass to the pipeline.
        :return: The pipeline for the given task with the given model.
    """

    # Check if the pipeline is already in global memory to avoid reloading
    if memoize:
        if "loaded_models" not in globals():
            globals()["loaded_models"] = {}
        global_model_key = f"{task}_{model_name}"
        if global_model_key in globals()["loaded_models"]:
            return globals()["loaded_models"][global_model_key]

    # Download the model if not already downloaded
    model_path = os.path.join(args.models_directory, model_name)
    if not os.path.exists(model_path):
        print(f"Downloading model {model_name} to {model_path}", file=sys.stderr, flush=True)
        huggingface_hub.login(token=open(args.hf_key, "r").read().strip())
        huggingface_hub.snapshot_download(repo_id=model_name, local_dir=model_path)
        os.chmod(f"/Brain/public/models/{model_path}", 0o777)

    # Load the pipeline
    print(f"Loading pipeline with model {model_name} for {task}", file=sys.stderr, flush=True)
    pipe = pipeline(task, model=model_path, **kwargs)

    # Memoize if needed
    if memoize:
        globals()["loaded_models"][global_model_key] = pipe
    return pipe

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

def get_lyrics (lyrics_file, source, file_name_no_extension, memoize=True):

    """
        Return the lyrics of a song given its file name.
        :param lyrics_file: The file containing the lyrics.
        :param file_name_no_extension: The file name of the song without extension.
        :param memoize: Whether to store the lyrics in global memory to avoid reloading if calling the function multiple times.
        :return: A dictionary of lyrics, with each value is a retranscription of the lyrics with column name as key.
    """

    # Check if the lyrics are already in global memory to avoid reloading
    if memoize:
        if "loaded_lyrics" not in globals():
            globals()["loaded_lyrics"] = {}
        global_lyrics_key = f"{lyrics_file}_{source}_{file_name_no_extension}"
        if global_lyrics_key in globals()["loaded_lyrics"]:
            return globals()["loaded_lyrics"][global_lyrics_key]

    # Lyrics are stored in .odt files
    sheet = pandas.read_excel(lyrics_file, engine="odf", sheet_name=source.replace(os.path.sep, "___"))
    
    # Search for the lyrics in the sheet
    lyrics = {}
    for i, row in sheet.iterrows():
        if row["File"] == file_name_no_extension:

            # Extract all candidate lyrics
            for column in sheet.columns:
                if column.startswith("Lyrics"):
                    lyrics[column] = normalize_lyrics(str(row[column]))
            
            # Memoize if needed
            if memoize:
                globals()["loaded_lyrics"][global_lyrics_key] = lyrics
            return lyrics

    # Raise exception if the lyrics are not found
    raise Exception(f"Lyrics not found for {file_name_no_extension}")
 
#####################################################################################################################################################

def compute_metrics (predicted_lyrics, actual_lyrics):

    """
        Compute various metrics between two sets of lyrics.
        :param predicted_lyrics: The predicted lyrics.
        :param actual_lyrics: The reference lyrics.
        :return: The similarities/errors between the two sets of lyrics.
    """

    # Function to compute similarity in an embedding
    def _embedding_similarity (st_model):
        pipe = get_pipeline("feature-extraction", st_model)
        embedding_actual = pipe(actual_lyrics, return_tensors=True)[0].mean(dim=0)
        embedding_predicted = pipe(predicted_lyrics, return_tensors=True)[0].mean(dim=0)
        return float(embedding_actual @ embedding_predicted / (embedding_actual.norm() * embedding_predicted.norm()))
    
    # Function to compute error with Word Error Rate
    def _word_error_rate ():
        wer = evaluate.load("wer")
        error = wer.compute(predictions=[actual_lyrics], references=[predicted_lyrics])
        return error

    # Return a dictionary of metrics
    metrics = {}
    for metric in args.metrics:
        if "/" in metric:
            metrics[metric] = _embedding_similarity(metric)
        elif metric == "WER":
            metrics["WER"] = _word_error_rate()
    return metrics

#####################################################################################################################################################

def print_title (title, size=100, character="#"):

    """
        Print a title with a given message.
        :param title: The message to print.
        :param size: The size of the separator.
        :param character: The character to use for the separator.
    """

    # Print centered title
    print("", flush=True)
    print("", flush=True)
    print("#" * size, flush=True)
    print("#" + " " * ((size - len(title) - 1) // 2) + title + " " * ((size - len(title) - 2) // 2) + "#", flush=True)
    print("#" * size, flush=True)
    print("", flush=True)

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

# Title
print_title("EXTRACT LYRICS")

# Extract lyrics from all audio files using the models
for asr_model in args.asr_models:

    # Load lyrics from file
    lyrics_file_name = os.path.join(args.output_directory, asr_model.replace(os.path.sep, "-") + ".ods")
    if args.extract_lyrics or not os.path.exists(lyrics_file_name):
        all_lyrics = {}
    else:
        dataframes = pandas.read_excel(lyrics_file_name, engine="odf", sheet_name=None)
        all_lyrics = {sheet: dataframes[sheet].to_dict(orient="list") for sheet in dataframes}

    # Load ASR pipeline
    pipe = get_pipeline("automatic-speech-recognition", asr_model)

    # One sheet per source
    for source in all_file_names:
        source_sheet = source.replace(os.path.sep, "___")
        if source_sheet not in all_lyrics:
            all_lyrics[source_sheet] = {"File": [], "Lyrics": []}
        
        # One line per file
        for file_name in all_file_names[source]:
            if file_name not in all_lyrics[source_sheet]["File"]:

                # Go through ASR pipeline
                print(f"Extracting lyrics for {file_name} with model {asr_model}", flush=True)
                out = pipe(get_audio(source, file_name), return_timestamps=True, generate_kwargs={"language": "english"})
                all_lyrics[source_sheet]["File"].append(file_name)
                all_lyrics[source_sheet]["Lyrics"].append(out["text"])
            
    # Save results to file
    with pandas.ExcelWriter(lyrics_file_name, engine="odf") as writer:
        for sheet_name, sheet in all_lyrics.items():
            pandas.DataFrame(sheet).to_excel(writer, sheet_name=sheet_name, index=False)

###################################
######### COMPUTE METRICS #########
###################################

# Title
print_title("COMPUTE METRICS")

# Load metrics from file
metrics_file_name = os.path.join(args.output_directory, "metrics.pt")
all_metrics = {}
if os.path.exists(metrics_file_name):
    with open(metrics_file_name, "rb") as file:
        all_metrics = pickle.load(file)

# First group by source
for source in sorted(all_file_names):

    # Initialize entry
    if source not in all_metrics or args.compute_metrics:
        all_metrics[source] = {}

    # Then group by file
    for file_name in sorted(all_file_names[source]):

        # Initialize entry
        if file_name not in all_metrics[source] or args.compute_metrics:
            all_metrics[source][file_name] = {}

        # Then group by ASR model
        for asr_model in args.asr_models:

            # Initialize entry
            if asr_model not in all_metrics[source][file_name] or args.compute_metrics:
                all_metrics[source][file_name][asr_model] = {}

            # Load lyrics
            actual_lyrics = get_lyrics(os.path.join(args.dataset, "lyrics.ods"), source, file_name)
            found_lyrics = get_lyrics(os.path.join(args.output_directory, asr_model.replace(os.path.sep, "-") + ".ods"), source, file_name)["Lyrics"]

            # Compute metrics if asked
            for key in actual_lyrics:
                if key not in all_metrics[source][file_name][asr_model] or args.compute_metrics:
                    all_metrics[source][file_name][asr_model][key] = compute_metrics(actual_lyrics[key], found_lyrics)

# Save results to file
with open(metrics_file_name, "wb") as file:
    pickle.dump(all_metrics, file)

# Print results
for source in sorted(all_file_names):
    print(f"[SOURCE] {source}", flush=True)
    for file_name in sorted(all_file_names[source]):
        print(f"|__ [FILE] {file_name}", flush=True)
        for asr_model in args.asr_models:
            print(f"|   |__ [MODEL] {asr_model}", flush=True)
            all_metrics_names = list(all_metrics[source][file_name][asr_model].values())[0]
            for metric in all_metrics_names:
                best = min if metric == "WER" else max
                all_values = [all_metrics[source][file_name][asr_model][key][metric] for key in all_metrics[source][file_name][asr_model]]
                print(f"|   |   |__ [METRIC] {metric} -- {best.__name__}({all_values}) = {best(all_values)}", flush=True)

###################################
######### ANALYZE METRICS #########
###################################

# Title
print_title("ANALYZE METRICS")

# Check recognition per style on EMVD
if "emvd" in all_metrics:

    # Group by style, model and metric (best value among candidate lyrics)
    all_styles = list(set(file_name.split("_")[1] for file_name in all_metrics["emvd"]))
    all_models = list(list(all_metrics["emvd"].values())[0].keys())
    all_metrics_names = list(list(list(all_metrics["emvd"].values())[0][all_models[0]].values())[0].keys())
    grouped_metrics = {style: {model: {metric: [] for metric in all_metrics_names} for model in all_models} for style in all_styles}
    for file_name in all_metrics["emvd"]:
        style = file_name.split("_")[1]
        for model in all_models:
            for metric in all_metrics_names:
                best = min if metric == "WER" else max
                grouped_metrics[style][model][metric].append(best([all_metrics["emvd"][file_name][model][key][metric] for key in all_metrics["emvd"][file_name][model]]))

    # Show average results per style
    for style in grouped_metrics:
        print(f"[STYLE] {style}")
        for model in grouped_metrics[style]:
            print(f"|__ [MODEL] {model}")
            for metric in grouped_metrics[style][model]:
                values = torch.tensor(grouped_metrics[style][model][metric])
                mean_metric = torch.mean(values)
                std_metric = torch.std(values)
                print(f"|   |__ [METRIC] {metric} -- mean = {mean_metric} -- std = {std_metric}")

    # Plot results (one subplot per metric pair)
    colors = ["blue", "red", "green", "orange", "purple", "brown", "pink", "gray", "cyan", "magenta"]
    symbols = ["o", "s", "D", "v", "^", "<", ">", "p", "h", "H"]
    fig, axs = pyplot.subplots(len(all_metrics_names)-1, len(all_metrics_names)-1, figsize=(len(all_metrics_names) * 5, len(all_metrics_names) * 5))
    for i_metric_1, metric_1 in enumerate(all_metrics_names):
        for i_metric_2, metric_2 in enumerate(all_metrics_names):
            if i_metric_1 < i_metric_2:
                for i_style, style in enumerate(grouped_metrics):
                    for i_model, model in enumerate(grouped_metrics[style]):
                        values_1 = torch.tensor(grouped_metrics[style][model][metric_1])
                        values_2 = torch.tensor(grouped_metrics[style][model][metric_2])
                        axs[i_metric_1, i_metric_2 - 1].scatter(torch.mean(values_1), torch.mean(values_2), label=f"{style} / {model}", color=colors[i_style], marker=symbols[i_model])
                axs[i_metric_1, i_metric_2 - 1].set_xlabel(metric_1)
                axs[i_metric_1, i_metric_2 - 1].set_ylabel(metric_2)
            if i_metric_1 == 0 and i_metric_2 == 1:
                fig.legend(loc="lower left", bbox_to_anchor=(0.1, 0.1))
            if 0 < i_metric_1 < len(all_metrics_names)-1 and i_metric_2 < i_metric_1:
                axs[i_metric_1, i_metric_2].axis("off")
    fig.tight_layout()
    fig.savefig(os.path.join(args.output_directory, "emvd_metrics_per_style.png"))

#####################################################################################################################################################