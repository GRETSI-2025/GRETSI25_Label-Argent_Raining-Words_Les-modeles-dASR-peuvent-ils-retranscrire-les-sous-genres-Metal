#####################################################################################################################################################
################################################################### PREPARE STUFF ###################################################################
#####################################################################################################################################################

# External imports
import os
import sys
import pickle
import torch
import matplotlib.pyplot as plt
import plotly.express as px
import pandas

# Project imports
from arguments import args
from lib_metrics import get_metric
from lib_audio import list_from_source

#####################################################################################################################################################
######################################################################### GO ########################################################################
#####################################################################################################################################################

# Get the list of all file names to work on
all_file_names = list_from_source("emvd")

# Make sure output subdirectory exists
output_directory = os.path.join(args().output_directory, "figures")
os.makedirs(output_directory, exist_ok=True)

# Load metrics from file
metrics_file_name = os.path.join(args().output_directory, "data", "metrics.pt")
with open(metrics_file_name, "rb") as file:
    all_metrics = pickle.load(file)

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
        print(f"[STYLE] {style}", file=sys.stdout, flush=True)
        for asr_model in args().asr_models:
            print(f"|__ [MODEL] {asr_model}", file=sys.stdout, flush=True)
            for metric_name in args().metrics:
                all_values = torch.tensor(grouped_metrics[style][asr_model][metric_name])
                mean_metric = torch.mean(all_values)
                std_metric = torch.std(all_values)
                print(f"|   |__ [METRIC] {metric_name} -- mean = {mean_metric} -- std = {std_metric}", file=sys.stdout, flush=True)

    # Scatter plot of styles per metric/model
    #colors = ["blue", "red", "green", "orange", "purple", "brown", "pink", "gray", "cyan", "magenta"]
    #symbols = ["o", "s", "D", "v", "^", "<", ">", "p", "h", "H"]
    #subplots_size = len(args().metrics) - 1
    #fig, axs = plt.subplots(subplots_size, subplots_size, figsize=(subplots_size * 5, subplots_size * 5))
    #for i_metric_name_1, metric_name_1 in enumerate(args().metrics):
    #    for i_metric_name_2, metric_name_2 in enumerate(args().metrics):
    #        if i_metric_name_1 < i_metric_name_2:
    #            for i_style, style in enumerate(grouped_metrics):
    #                for i_model, model in enumerate(grouped_metrics[style]):
    #                    values_1 = torch.tensor(grouped_metrics[style][model][metric_name_1])
    #                    values_2 = torch.tensor(grouped_metrics[style][model][metric_name_2])
    #                    axs[i_metric_name_1, i_metric_name_2 - 1].scatter(torch.mean(values_1), torch.mean(values_2), label=f"{style} / {model}", color=colors[i_style], marker=symbols[i_model])
    #            axs[i_metric_name_1, i_metric_name_2 - 1].set_xlabel(metric_name_1)
    #            axs[i_metric_name_1, i_metric_name_2 - 1].set_ylabel(metric_name_2)
    #        if i_metric_name_1 == 0 and i_metric_name_2 == 1:
    #            fig.legend(loc="lower left", bbox_to_anchor=(0.1, 0.1))
    #        if 0 < i_metric_name_1 < subplots_size and i_metric_name_2 < i_metric_name_1:
    #            axs[i_metric_name_1, i_metric_name_2].axis("off")
    #fig.tight_layout()
    #fig.savefig(os.path.join(args().output_directory, "figures", "emvd_metrics_per_style.png"))

    # Spider plot of styles per model using plotly.express
    for metric_name in args().metrics:
        data = []
        for style in grouped_metrics:
            for asr_model in args().asr_models:
                mean_value = torch.mean(torch.tensor(grouped_metrics[style][asr_model][metric_name])).item()
                data.append({"Style": style, "Model": asr_model, "Metric": metric_name, "Value": mean_value})
        df = pandas.DataFrame(data)
        fig = px.line_polar(df, r='Value', theta='Style', color='Model', line_close=True)
        fig.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0, 1])), showlegend=True)
        fig.write_image(os.path.join(output_directory, f"emvd - {metric_name}.png"))
        
#####################################################################################################################################################
#####################################################################################################################################################