#####################################################################################################################################################
################################################################### PREPARE STUFF ###################################################################
#####################################################################################################################################################

# External imports
import os
import sys
from pathlib import Path
import pandas
import pickle
import torch
import matplotlib.pyplot as pyplot

# Project imports
from arguments import args
from lib_models import *
from lib_metrics import *

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

def get_audio (source, file_name_no_extension):

    # Search for the audio file in the source
    for file in os.listdir(os.path.join(args().dataset, "audio", source)):
        if file.startswith(file_name_no_extension):
            return os.path.join(args().dataset, "audio", source, file)
    
    # Raise exception if the audio file is not found
    raise Exception(f"Audio file not found for {file_name_no_extension}")

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
for asr_model in args().asr_models:

    # Load existing lyrics from file
    lyrics_file_name = os.path.join(args().output_directory, asr_model.replace(os.path.sep, "-") + ".ods")
    if args().extract_lyrics or not os.path.exists(lyrics_file_name):
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
                print(f"Extracting lyrics for \"{file_name}\" with model \"{asr_model}\"", flush=True)
                transcription = model.transcribe(get_audio(source, file_name))
                all_lyrics[source_sheet]["File"].append(file_name)
                all_lyrics[source_sheet]["Lyrics"].append(transcription)
            
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
metrics_file_name = os.path.join(args().output_directory, "metrics.pt")
all_metrics = {}
if not args().compute_metrics and os.path.exists(metrics_file_name):
    with open(metrics_file_name, "rb") as file:
        all_metrics = pickle.load(file)

# First group by source
for source in sorted(all_file_names):
    if source not in all_metrics:
        all_metrics[source] = {}

    # Then group by file
    for file_name in sorted(all_file_names[source]):
        if file_name not in all_metrics[source]:
            all_metrics[source][file_name] = {}

        # Then group by ASR model
        for asr_model in args().asr_models:
            if asr_model not in all_metrics[source][file_name]:
                all_metrics[source][file_name][asr_model] = {}

            # Get lyrics
            actual_lyrics = get_lyrics(os.path.join(args().dataset, "lyrics.ods"), source, file_name)
            found_lyrics = get_lyrics(os.path.join(args().output_directory, asr_model.replace(os.path.sep, "-") + ".ods"), source, file_name)["Lyrics"]

            # Then group by lyrics version
            for lyrics_version in actual_lyrics:
                if lyrics_version not in all_metrics[source][file_name][asr_model]:
                    all_metrics[source][file_name][asr_model][lyrics_version] = {}
                    
                # Then group by metric
                for metric_name in args().metrics:
                    if metric_name not in all_metrics[source][file_name][asr_model][lyrics_version]:

                        # Compute metric
                        print(f"Computing metric \"{metric_name}\" for \"{file_name}\" with model \"{asr_model}\" and lyrics version \"{lyrics_version}\"", flush=True)
                        metric = get_metric(metric_name)
                        all_metrics[source][file_name][asr_model][lyrics_version][metric_name] = metric.compute(actual_lyrics[lyrics_version], found_lyrics)

# Save results to file
with open(metrics_file_name, "wb") as file:
    pickle.dump(all_metrics, file)

# Print results
for source in sorted(all_file_names):
    print(f"[SOURCE] {source}", flush=True)
    for file_name in sorted(all_file_names[source]):
        print(f"|__ [FILE] {file_name}", flush=True)
        for asr_model in args().asr_models:
            print(f"|   |__ [MODEL] {asr_model}", flush=True)
            for metric_name in args().metrics:
                metric = get_metric(metric_name)
                all_values = [all_metrics[source][file_name][asr_model][lyrics_version][metric_name] for lyrics_version in all_metrics[source][file_name][asr_model]]
                print(f"|   |   |__ [METRIC] {metric_name} -- {metric.best.__name__}({all_values}) = {metric.best(all_values)}", flush=True)

###################################
########### ANALYZE EMVD ##########
###################################

# Title
print_title("ANALYZE EMVD")

# Check recognition per style on EMVD
if "emvd" in all_metrics:

    # Groups songs by style
    all_styles = list(set(file_name.split("_")[1] for file_name in all_metrics["emvd"]))
    songs_by_style = {style: [file_name for file_name in all_metrics["emvd"] if file_name.split("_")[1] == style] for style in all_styles}

    # Group by style, model and metric
    # In case of multiple lyrics versions for the same song, take the best one
    grouped_metrics = {style: {model: {metric: [] for metric in args().metrics} for model in args().asr_models} for style in all_styles}
    for style in songs_by_style:
        for file_name in songs_by_style[style]:
            for asr_model in args().asr_models:
                for metric_name in args().metrics:
                    metric = get_metric(metric_name)
                    grouped_metrics[style][asr_model][metric_name].append(metric.best([all_metrics["emvd"][file_name][asr_model][lyrics_version][metric_name] for lyrics_version in all_metrics["emvd"][file_name][asr_model]]))
    
    # Show average results per style
    for style in songs_by_style:
        print(f"[STYLE] {style}")
        for asr_model in args().asr_models:
            print(f"|__ [MODEL] {asr_model}")
            for metric_name in args().metrics:
                all_values = torch.tensor(grouped_metrics[style][asr_model][metric_name])
                mean_metric = torch.mean(all_values)
                std_metric = torch.std(all_values)
                print(f"|   |__ [METRIC] {metric_name} -- mean = {mean_metric} -- std = {std_metric}")

    # Plot results (one subplot per metric pair)
    colors = ["blue", "red", "green", "orange", "purple", "brown", "pink", "gray", "cyan", "magenta"]
    symbols = ["o", "s", "D", "v", "^", "<", ">", "p", "h", "H"]
    subplots_size = len(args().metrics) - 1
    fig, axs = pyplot.subplots(subplots_size, subplots_size, figsize=(subplots_size * 5, subplots_size * 5))
    for i_metric_name_1, metric_name_1 in enumerate(args().metrics):
        for i_metric_name_2, metric_name_2 in enumerate(args().metrics):
            if i_metric_name_1 < i_metric_name_2:
                for i_style, style in enumerate(grouped_metrics):
                    for i_model, model in enumerate(grouped_metrics[style]):
                        values_1 = torch.tensor(grouped_metrics[style][model][metric_name_1])
                        values_2 = torch.tensor(grouped_metrics[style][model][metric_name_2])
                        axs[i_metric_name_1, i_metric_name_2 - 1].scatter(torch.mean(values_1), torch.mean(values_2), label=f"{style} / {model}", color=colors[i_style], marker=symbols[i_model])
                axs[i_metric_name_1, i_metric_name_2 - 1].set_xlabel(metric_name_1)
                axs[i_metric_name_1, i_metric_name_2 - 1].set_ylabel(metric_name_2)
            if i_metric_name_1 == 0 and i_metric_name_2 == 1:
                fig.legend(loc="lower left", bbox_to_anchor=(0.1, 0.1))
            if 0 < i_metric_name_1 < subplots_size and i_metric_name_2 < i_metric_name_1:
                axs[i_metric_name_1, i_metric_name_2].axis("off")
    fig.tight_layout()
    fig.savefig(os.path.join(args().output_directory, "emvd_metrics_per_style.png"))

#####################################################################################################################################################
###################################################################### CLEANUP ######################################################################
#####################################################################################################################################################

# Set permissions for shared use
os.chmod(args().output_directory, 0o777)
    
#####################################################################################################################################################
#####################################################################################################################################################