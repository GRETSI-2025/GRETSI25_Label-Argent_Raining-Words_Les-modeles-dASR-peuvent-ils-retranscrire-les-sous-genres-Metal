#####################################################################################################################################################
###################################################################### LICENSE ######################################################################
#####################################################################################################################################################
#
#    Copyright (C) 2025  Bastien Pasdeloup & Axel Marmoret
#
#    This program is free software: you can redistribute it and/or modify
#    it under the terms of the GNU General Public License as published by
#    the Free Software Foundation, either version 3 of the License, or
#    (at your option) any later version.
#
#    This program is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#    GNU General Public License for more details.
#
#    You should have received a copy of the GNU General Public License
#    along with this program.  If not, see <https://www.gnu.org/licenses/>.
#
#####################################################################################################################################################
################################################################### DOCUMENTATION ###################################################################
#####################################################################################################################################################

"""
    This script performs analyzes on the songs and source-separated datasets.
    It produces figures for each metric, to compare the performance of the ASR models on the datasets, per style.
"""

#####################################################################################################################################################
################################################################### PREPARE STUFF ###################################################################
#####################################################################################################################################################

# External imports
import os
import pickle
import torch
import plotly.express as px
import matplotlib.pyplot as plt
import numpy as np
import pandas

# Project imports
from lib.arguments import script_args
import lib.audio
import lib.metrics

#####################################################################################################################################################
######################################################################### GO ########################################################################
#####################################################################################################################################################

# Create output directory if it does not exist
figures_directory = os.path.join(script_args().output_directory, "figures")
os.makedirs(figures_directory, exist_ok=True)
os.chmod(figures_directory, 0o777)

# Load metrics from file
metrics_file_path = os.path.join(script_args().output_directory, "data", "metrics.pt")
with open(metrics_file_path, "rb") as file:
    all_metrics = pickle.load(file)
    
# Produce a set of figures per dataset
datasets = [f for f in os.listdir(os.path.join(script_args().datasets_path, "audio")) if f != "emvd"]
for dataset in datasets:

    # Get the list of all files to work on
    all_file_paths = lib.audio.list_from_dataset(dataset)

    # Build a plot per metric
    for metric_name in script_args().metrics:
        
        # Create dataframe for the figure
        metric = lib.metrics.get_metric(metric_name)
        data = []
        for sub_dataset in sorted(all_file_paths):
            style = sub_dataset.split(os.path.sep)[-1]
            for asr_model in script_args().asr_models_songs:
                best_per_song = [metric.best(all_metrics[sub_dataset][file_name][asr_model][lyrics_version][metric_name] for lyrics_version in all_metrics[sub_dataset][file_name][asr_model]) for file_name in all_file_paths[sub_dataset]]
                mean_value = torch.mean(torch.tensor(best_per_song)).item()
                data.append({"Style": style, "Model": asr_model, "Metric": metric_name, "Value": mean_value})
        
        # Build polar figure
        fig = px.line_polar(pandas.DataFrame(data), r="Value", theta="Style", color="Model", line_close=True)
        fig.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0, 1])), showlegend=True, legend=dict(yanchor="bottom", xanchor="center", x=1))

        # Save figure
        figure_file_path = os.path.join(figures_directory, f"{dataset} - {metric_name}.png")
        fig.write_image(figure_file_path)
        os.chmod(figure_file_path, 0o777)

# Compare results per file version (song/sourced-separated)
colors = ["blue", "red", "green", "orange", "purple", "brown", "pink", "gray", "cyan", "magenta", "yellow", "black", "lightblue", "lightgreen", "lightred", "lightorange", "lightpurple", "lightbrown", "lightpink", "lightgray", "lightcyan", "lightmagenta", "lightyellow"]
for i_version_1, version_1 in enumerate(datasets):
    for version_2 in datasets[i_version_1 + 1:]:

        # Get the list of all files to work on
        all_file_paths = lib.audio.list_from_dataset(version_1)

        # Build a plot per metric and model
        for metric_name in script_args().metrics:
            metric = lib.metrics.get_metric(metric_name)
            for asr_model in script_args().asr_models_songs:

                # Get data (performances) per subdataset (for colors)
                data = {sub_dataset.split(os.path.sep)[-1]: [] for sub_dataset in all_file_paths}
                for style in data:
                    for file_name in sorted(all_file_paths[os.path.join(version_1, style)]):
                        perf_version_1 = metric.best(all_metrics[os.path.join(version_1, style)][file_name][asr_model][lyrics_version][metric_name] for lyrics_version in all_metrics[os.path.join(version_1, style)][file_name][asr_model])
                        perf_version_2 = metric.best(all_metrics[os.path.join(version_2, style)][file_name][asr_model][lyrics_version][metric_name] for lyrics_version in all_metrics[os.path.join(version_2, style)][file_name][asr_model])
                        data[style].append((perf_version_1, perf_version_2))
                
                # Compute linear regression of all data
                x, y = zip(*[point for style in data for point in data[style]])
                a, b = np.polyfit(x, y, 1)
                correlation = np.corrcoef(x, y)[0, 1]
                print(f"Version 1: {version_1}, Version 2: {version_2}, Model: {asr_model}, Metric: {metric_name}, Linear regression: y = {a}x + {b}, Correlation: {correlation}")

                # Build scatter figure
                fig, ax = plt.subplots()
                for i_style, style in enumerate(data):
                    x, y = zip(*data[style])
                    ax.scatter(x, y, label=style, color=colors[i_style])
                ax.set_xlabel(f"{version_1} - {metric_name}")
                ax.set_ylabel(f"{version_2} - {metric_name}")
                ax.legend()
                # Automatically adjust figure size to show legend
                box = ax.get_position()
                fig.set_size_inches(fig.get_size_inches()[0] * 1.5, fig.get_size_inches()[1])
                plt.legend(loc="center left", bbox_to_anchor=(1, 0.5))
                

                # Draw diagonal in dashed line
                ax.plot([0, 1], [0, 1], "--", color="black")

                # Draw linear regression in solid line
                x = np.linspace(0, 1, 100)
                y = a * x + b
                ax.plot(x, y, color="black")

                # Save figure
                figure_file_path = os.path.join(figures_directory, f"{version_1} - {version_2} - {metric_name} - {asr_model}.png")
                plt.savefig(figure_file_path)
                os.chmod(figure_file_path, 0o777)

#####################################################################################################################################################
#####################################################################################################################################################: