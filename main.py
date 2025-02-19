#####################################################################################################################################################
################################################################### PREPARE STUFF ###################################################################
#####################################################################################################################################################

# External imports
import os
import sys
from pathlib import Path
import pandas
import evaluate
import pickle
import torch
import matplotlib.pyplot as pyplot

# Project imports
from arguments import args
from model_loaders import *
from text_metrics import *

#####################################################################################################################################################
##################################################################### FUNCTIONS #####################################################################
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
    for i, row in sheet.iterrows():
        if row["File"] == file_name_no_extension:

            # Extract all candidate lyrics
            for column in sheet.columns:
                if column.startswith("Lyrics"):
                    lyrics[column] = normalize_lyrics(str(row[column]))
            return lyrics

    # Raise exception if the lyrics are not found
    raise Exception(f"Lyrics not found for {file_name_no_extension}")
 
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

    # Load existing lyrics from file
    lyrics_file_name = os.path.join(args.output_directory, asr_model.replace(os.path.sep, "-") + ".ods")
    if args.extract_lyrics or not os.path.exists(lyrics_file_name):
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
                print(f"Extracting lyrics for {file_name} with model {asr_model}", flush=True)
                out = model.transcribe(get_audio(source, file_name), return_timestamps=True, generate_kwargs={"language": "english"})
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

            # There can be multiple candidate actual lyrics
            actual_lyrics = get_lyrics(os.path.join(args.dataset, "lyrics.ods"), source, file_name)
            found_lyrics = get_lyrics(os.path.join(args.output_directory, asr_model.replace(os.path.sep, "-") + ".ods"), source, file_name)["Lyrics"]
            for actual_lyrics_version in actual_lyrics:
                if actual_lyrics_version not in all_metrics[source][file_name][asr_model] or args.compute_metrics:
                    
                    # Compute metrics
                    for metric_name in args.metrics:
                        metric = get_metric(metric_name)
                        all_metrics[source][file_name][asr_model][actual_lyrics_version] = metric.compute(actual_lyrics[actual_lyrics_version], found_lyrics)

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
            for metric_name in all_metrics_names:
                metric = get_metric(metric_name)
                all_values = [all_metrics[source][file_name][asr_model][key][metric_name] for key in all_metrics[source][file_name][asr_model]]
                print(f"|   |   |__ [METRIC] {metric_name} -- {metric.best.__name__}({all_values}) = {metric.best(all_values)}", flush=True)

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
            for metric_name in all_metrics_names:
                metric = get_metric(metric_name)
                grouped_metrics[style][model][metric].append(metric.best([all_metrics["emvd"][file_name][model][key][metric] for key in all_metrics["emvd"][file_name][model]]))

    # Show average results per style
    for style in grouped_metrics:
        print(f"[STYLE] {style}")
        for model in grouped_metrics[style]:
            print(f"|__ [MODEL] {model}")
            for metric_name in grouped_metrics[style][model]:
                values = torch.tensor(grouped_metrics[style][model][metric_name])
                mean_metric = torch.mean(values)
                std_metric = torch.std(values)
                print(f"|   |__ [METRIC] {metric} -- mean = {mean_metric} -- std = {std_metric}")

    # Plot results (one subplot per metric pair)
    colors = ["blue", "red", "green", "orange", "purple", "brown", "pink", "gray", "cyan", "magenta"]
    symbols = ["o", "s", "D", "v", "^", "<", ">", "p", "h", "H"]
    fig, axs = pyplot.subplots(len(all_metrics_names)-1, len(all_metrics_names)-1, figsize=(len(all_metrics_names) * 5, len(all_metrics_names) * 5))
    for i_metric_name_1, metric_name_1 in enumerate(all_metrics_names):
        for i_metric_name_2, metric_name_2 in enumerate(all_metrics_names):
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
            if 0 < i_metric_name_1 < len(all_metrics_names)-1 and i_metric_name_2 < i_metric_name_1:
                axs[i_metric_name_1, i_metric_name_2].axis("off")
    fig.tight_layout()
    fig.savefig(os.path.join(args.output_directory, "emvd_metrics_per_style.png"))

#####################################################################################################################################################