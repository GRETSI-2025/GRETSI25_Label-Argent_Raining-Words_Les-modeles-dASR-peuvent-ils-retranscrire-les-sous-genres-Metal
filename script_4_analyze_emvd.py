#####################################################################################################################################################
################################################################### PREPARE STUFF ###################################################################
#####################################################################################################################################################

# External imports
import os
import pickle
import torch
import plotly.express as px
import pandas

# Project imports
from arguments import args
import lib_audio
import lib_metrics

#####################################################################################################################################################
######################################################################### GO ########################################################################
#####################################################################################################################################################

# Get the list of all file names to work on
all_file_names = lib_audio.list_from_dataset("emvd")

# Load metrics from file
metrics_file_name = os.path.join(args().output_directory, "data", "metrics.pt")
with open(metrics_file_name, "rb") as file:
    all_metrics = pickle.load(file)

# Get styles in the dataset
all_styles = list(set(file_name.split("_")[1] for file_name in all_metrics["emvd"]))
songs_by_style = {style: [file_name for file_name in all_metrics["emvd"] if file_name.split("_")[1] == style] for style in all_styles}

# Build a plot per metric
for metric_name in args().metrics:
    
    # Create dataframe for the figure
    metric = lib_metrics.get_metric(metric_name)
    data = []
    for style in all_styles:
        for asr_model in args().asr_models:
            best_per_song = [metric.best(all_metrics["emvd"][file_name][asr_model][lyrics_version][metric_name] for lyrics_version in all_metrics["emvd"][file_name][asr_model]) for file_name in songs_by_style[style]]
            mean_value = torch.mean(torch.tensor(best_per_song)).item()
            data.append({"Style": style, "Model": asr_model, "Metric": metric_name, "Value": mean_value})
    
    # Build polar figure
    fig = px.line_polar(pandas.DataFrame(data), r="Value", theta="Style", color="Model", line_close=True)
    fig.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0, 1])), showlegend=True)

    # Save figure
    figure_file_name = os.path.join(args().output_directory, "figures", f"emvd - {metric_name}.png")
    fig.write_image(figure_file_name)
    os.chmod(figure_file_name, 0o777)

#####################################################################################################################################################
#####################################################################################################################################################