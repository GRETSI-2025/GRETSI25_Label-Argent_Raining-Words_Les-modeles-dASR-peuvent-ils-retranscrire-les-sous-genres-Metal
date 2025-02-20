#####################################################################################################################################################
################################################################### PREPARE STUFF ###################################################################
#####################################################################################################################################################

# External imports
import os
import pickle

# Project imports
from arguments import args
from lib_metrics import get_metric
from lib_audio import get_lyrics, list_from_source

#####################################################################################################################################################
######################################################################### GO ########################################################################
#####################################################################################################################################################

# Get the list of all file names to work on
all_file_names = list_from_source(args().source)

# Load metrics from file
metrics_file_name = os.path.join(args().output_directory, "metrics.pt")
all_metrics = {}
if not args().force_compute_metrics and os.path.exists(metrics_file_name):
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
            actual_lyrics = get_lyrics(os.path.join(args().dataset, "lyrics.ods"), source.split(os.path.sep)[-1], file_name)
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

# Set permissions for shared use
os.chmod(metrics_file_name, 0o777)

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

#####################################################################################################################################################
#####################################################################################################################################################