#####################################################################################################################################################
################################################################### PREPARE STUFF ###################################################################
#####################################################################################################################################################

# External imports
import os
import sys
import pickle

# Project imports
from arguments import args
import lib_audio
import lib_metrics

#####################################################################################################################################################
######################################################################### GO ########################################################################
#####################################################################################################################################################

# Get the list of all file names to work on
all_file_names = lib_audio.list_from_dataset()

# Load metrics from file
metrics_file_name = os.path.join(args().output_directory, "data", "metrics.pt")
all_metrics = {}
if not args().force_compute_metrics and os.path.exists(metrics_file_name):
    with open(metrics_file_name, "rb") as file:
        all_metrics = pickle.load(file)

# First group by dataset
for dataset in sorted(all_file_names):
    if dataset not in all_metrics:
        all_metrics[dataset] = {}

    # Then group by file
    for file_name in sorted(all_file_names[dataset]):
        if file_name not in all_metrics[dataset]:
            all_metrics[dataset][file_name] = {}

        # Then group by ASR model
        for asr_model in args().asr_models:
            if asr_model not in all_metrics[dataset][file_name]:
                all_metrics[dataset][file_name][asr_model] = {}

            # Get lyrics
            actual_lyrics = lib_audio.get_lyrics(os.path.join(args().datasets_path, "lyrics.ods"), dataset.split(os.path.sep)[-1], file_name)
            found_lyrics = lib_audio.get_lyrics(os.path.join(args().output_directory, "data", asr_model.replace(os.path.sep, "-") + ".ods"), dataset, file_name)["Lyrics"]

            # Then group by lyrics version
            for lyrics_version in actual_lyrics:
                if lyrics_version not in all_metrics[dataset][file_name][asr_model]:
                    all_metrics[dataset][file_name][asr_model][lyrics_version] = {}
                    
                # Then group by metric
                for metric_name in args().metrics:
                    if metric_name not in all_metrics[dataset][file_name][asr_model][lyrics_version]:

                        # Compute metric
                        print(f"Computing metric \"{metric_name}\" for \"{file_name}\" with model \"{asr_model}\" and lyrics version \"{lyrics_version}\"", file=sys.stderr, flush=True)
                        metric = lib_metrics.get_metric(metric_name)
                        all_metrics[dataset][file_name][asr_model][lyrics_version][metric_name] = metric.compute(actual_lyrics[lyrics_version], found_lyrics)

# Save results to file
with open(metrics_file_name, "wb") as file:
    pickle.dump(all_metrics, file)
os.chmod(metrics_file_name, 0o777)

# Print results
for dataset in sorted(all_file_names):
    print(f"[DATASET] {dataset}", file=sys.stdout, flush=True)
    for file_name in sorted(all_file_names[dataset]):
        print(f"|__ [FILE] {file_name}", file=sys.stdout, flush=True)
        for asr_model in args().asr_models:
            print(f"|   |__ [MODEL] {asr_model}", file=sys.stdout, flush=True)
            for metric_name in args().metrics:
                metric = lib_metrics.get_metric(metric_name)
                all_values = [all_metrics[dataset][file_name][asr_model][lyrics_version][metric_name] for lyrics_version in all_metrics[dataset][file_name][asr_model]]
                print(f"|   |   |__ [METRIC] {metric_name} -- {metric.best.__name__}({all_values}) = {metric.best(all_values)}", file=sys.stdout, flush=True)

#####################################################################################################################################################
#####################################################################################################################################################