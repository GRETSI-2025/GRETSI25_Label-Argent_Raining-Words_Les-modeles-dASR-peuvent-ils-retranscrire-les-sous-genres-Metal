#####################################################################################################################################################
################################################################### PREPARE STUFF ###################################################################
#####################################################################################################################################################

# External imports
import argparse
import os

#####################################################################################################################################################
##################################################################### FUNCTIONS #####################################################################
#####################################################################################################################################################

@property
def args ():

    """
        Parse arguments for the script.
        :return: Parsed arguments.
    """

    # Prepare parser
    parser = argparse.ArgumentParser()

    # Paths
    parser.add_argument("--dataset", type=str, help="Path to the dataset", default="/Brain/public/datasets/metal/data")
    parser.add_argument("--output_directory", type=str, help="Path to the output directory", default="/Brain/public/datasets/metal/output")
    parser.add_argument("--models_directory", type=str, help="Path to where models are downloaded", default="/Brain/public/models")

    # Credentials
    parser.add_argument("--hf_key", type=str, help="Path to the Hugging Face token file", default=f"/Brain/private/{os.environ["LOGNAME"]}/misc/hugging_face.key")

    # Models to use
    parser.add_argument("--asr_models", type=list, help="List of models to evaluate", default=["WhisperLargeV2",
                                                                                               "WhisperLargeV3"])
    parser.add_argument("--metrics", type=list, help="Metrics or models used for computing similarity/error", default=["WER"])
                                                                                                                    #"Alibaba-NLP/gte-Qwen2-1.5B-instruct",
                                                                                                                    #"sentence-transformers/all-MiniLM-L6-v2",
                                                                                                                    #"sentence-transformers/all-mpnet-base-v2"])

    # Control parts of the script to run or not
    parser.add_argument("--extract_lyrics", type=bool, help="Force re-extraction of lyrics from audio files", default=False)
    parser.add_argument("--compute_metrics", type=bool, help="Force re-computation of metrics", default=False)

    # Go
    return parser.parse_args()

#####################################################################################################################################################
#####################################################################################################################################################