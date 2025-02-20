#####################################################################################################################################################
################################################################### PREPARE STUFF ###################################################################
#####################################################################################################################################################

# External imports
import argparse
import os

#####################################################################################################################################################
##################################################################### FUNCTIONS #####################################################################
#####################################################################################################################################################

def args ():

    # Prepare parser
    parser = argparse.ArgumentParser()

    # Paths
    parser.add_argument("--dataset", type=str, help="Path to the dataset", default="/Brain/public/datasets/metal/data")
    parser.add_argument("--output_directory", type=str, help="Path to the output directory", default="/Brain/public/datasets/metal/output")
    parser.add_argument("--models_directory", type=str, help="Path to where models are downloaded", default="/Brain/public/models")

    # Credentials
    parser.add_argument("--hf_key", type=str, help="Path to the Hugging Face token file", default=f"/Brain/private/{os.environ["LOGNAME"]}/misc/hugging_face.key")

    # Elements to use
    parser.add_argument("--asr_models", type=list, help="List of models to evaluate", default=[#"Canary_1B",
                                                                                               #"Wav2vec2_Large_960h_Lv60_Self",
                                                                                               "Whisper_Large_V2",
                                                                                               "Whisper_Large_V3"])
    parser.add_argument("--metrics", type=list, help="Metrics or models used for computing similarity/error", default=["WER",
                                                                                                                       ("EmbeddingSimilarity", "Gte_Qwen2_1d5B_Instruct"),
                                                                                                                       ("EmbeddingSimilarity", "All_MiniLM_L6_V2"),
                                                                                                                       ("EmbeddingSimilarity", "All_MPNet_Base_V2")])
    parser.add_argument("--source", type=str, help="Source to use in the scripts, as a subdirectory of 'audio' (None for all)", default="songs")

    # Control parts of the scripts to run or not
    parser.add_argument("--force_download_audio", type=bool, help="Force re-downloading of audio files", default=False)
    parser.add_argument("--force_lyrics_extraction", type=bool, help="Force re-extraction of lyrics from audio files", default=False)
    parser.add_argument("--force_compute_metrics", type=bool, help="Force re-computation of metrics", default=False)

    # Go
    return parser.parse_args()

#####################################################################################################################################################
#####################################################################################################################################################