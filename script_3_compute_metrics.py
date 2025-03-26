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
    This script computes metrics to evaluate the performance of the ASR models.
    When multiple ground truth lyrics are available (as provided by multiple listeners, or from various sources), the best value is selected.
    Metrics are stored in a file to avoid recomputing them.
"""

#####################################################################################################################################################
################################################################### PREPARE STUFF ###################################################################
#####################################################################################################################################################

# External imports
import os
import sys
import pickle

# Project imports
from lib.arguments import script_args
import lib.audio
import lib.metrics

#####################################################################################################################################################
######################################################################### GO ########################################################################
#####################################################################################################################################################

# Create output directory if it does not exist
data_directory = os.path.join(script_args().output_directory, "data")
os.makedirs(data_directory, exist_ok=True)
os.chmod(data_directory, 0o777)

# Get the list of all files to work on
all_files = lib.audio.list_from_dataset()

# Load metrics from file
metrics_file_path = os.path.join(data_directory, "metrics.pt")
all_metrics = {}
if os.path.exists(metrics_file_path):
    with open(metrics_file_path, "rb") as file:
        all_metrics = pickle.load(file)

# First group by dataset
for dataset in sorted(all_files):
    if dataset not in all_metrics:
        all_metrics[dataset] = {}

    # Then group by file
    for file_name in sorted(all_files[dataset]):
        if file_name not in all_metrics[dataset]:
            all_metrics[dataset][file_name] = {}

        # Then group by ASR model
        asr_models = script_args().asr_models if dataset.startswith("emvd") else script_args().asr_models_songs
        for asr_model in asr_models:
            if asr_model not in all_metrics[dataset][file_name]:
                all_metrics[dataset][file_name][asr_model] = {}

            # Get lyrics
            actual_lyrics = lib.audio.get_lyrics(os.path.join(script_args().datasets_path, "lyrics.ods"), dataset.split(os.path.sep)[-1], file_name)
            found_lyrics = lib.audio.get_lyrics(os.path.join(data_directory, asr_model.replace(os.path.sep, "-") + ".ods"), dataset, file_name)["Lyrics"]

            # Then group by lyrics version
            for lyrics_version in actual_lyrics:
                if lyrics_version not in all_metrics[dataset][file_name][asr_model]:
                    all_metrics[dataset][file_name][asr_model][lyrics_version] = {}
                    
                # Then group by metric
                for metric_name in script_args().metrics:
                    if metric_name not in all_metrics[dataset][file_name][asr_model][lyrics_version]:

                        # Compute metric
                        print(f"Computing metric \"{metric_name}\" for \"{file_name}\" with model \"{asr_model}\" and lyrics version \"{lyrics_version}\"", file=sys.stderr, flush=True)
                        metric = lib.metrics.get_metric(metric_name)
                        all_metrics[dataset][file_name][asr_model][lyrics_version][metric_name] = metric.compute(actual_lyrics[lyrics_version], found_lyrics)

            # Save results to file
            with open(metrics_file_path, "wb") as file:
                pickle.dump(all_metrics, file)
            os.chmod(metrics_file_path, 0o777)

# Print results
for dataset in sorted(all_files):
    print(f"[DATASET] {dataset}", file=sys.stdout, flush=True)
    for file_name in sorted(all_files[dataset]):
        print(f"|__ [FILE] {file_name}", file=sys.stdout, flush=True)
        for asr_model in script_args().asr_models:
            print(f"|   |__ [MODEL] {asr_model}", file=sys.stdout, flush=True)
            for metric_name in script_args().metrics:
                metric = lib.metrics.get_metric(metric_name)
                all_values = [all_metrics[dataset][file_name][asr_model][lyrics_version][metric_name] for lyrics_version in all_metrics[dataset][file_name][asr_model]]
                print(f"|   |   |__ [METRIC] {metric_name} -- {metric.best.__name__}({all_values}) = {metric.best(all_values)}", file=sys.stdout, flush=True)

#####################################################################################################################################################
#####################################################################################################################################################