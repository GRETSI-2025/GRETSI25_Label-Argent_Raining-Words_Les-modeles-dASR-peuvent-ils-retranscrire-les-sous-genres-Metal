# extreme_vocals_asr

This script is used to evaluate the performance of ASR models on a dataset of metal songs.

### Contacts

- Bastien Pasdeloup (bastien.pasdeloup@imt-atlantique.fr)
- Axel Marmoret (axel.marmoret@imt-atlantique.fr)

### Dataset

Dataset should be formatted as follows:
```
- data
    - audio
        - source_1
            - file_name_1.{wav, flac...}
            - file_name_2.{wav, flac...}
        - source_2
            - file_name_3.{wav, flac...}
            - file_name_4.{wav, flac...}
        - ...
    - lyrics.odt
```

Audio files can be in any format, as long as the ASR model can handle it.
They should be grouped by source to keep datasets together (*e.g.*, EMVD, songs).

File `lyrics.ods` should have one tab per source and one line per song.
You can provide multiple transcripts of the same lyrics (*e.g.*, various listener perceptions, declared lyrics).

The first script (see below) will auomatically download a curated list of songs, but you can also use your own dataset.

### Installation

Installation is mostly straightforward, but requires some dependencies to be installed using the following command:
```bash
pip install -r requirements.txt
```

Also, please make sure that you have a HuggingFace key for downloading models. If in a file, you can pass it to the script using the `--hf_key_path` parameter below.

### Usage

Each script file is a standalone script that can be run independently as long as results from the previous scripts are available.

- `script_1_get_dataset.py` downloads the dataset, and extracts vocals from the songs.
- `script_2_extract_lyrics.py` extracts the lyrics from all files in the dataset using ASR models.
- `script_3_compute_metrics` computes all similarity/distance metrics between the lyrics and the ASR outputs. 
- `script_4_analyze_emvd.py` reproduces all figures from the paper on the EMVD dataset.
- `script_5_analyze_songs.py` reproduces all figures from the paper on the songs & source-separated datasets.

Each script can be run as follows:
```bash
python3 <script.py> [<parameters>]
```

In addition, all scripts can be run in a single command using the `script_run_all.py` script.

### Parameters

- `--dataset_path`: Path to the dataset directory.
- `--output directory`: Path to the output directory.
- `--models_directory`: Path to where models are downloaded.
- `--hf_key_path`: Path to a file containing your Hugging Face token.
- `--source_separation_models`: List of models to use for source separation (default: `[("Demucs", "mdx_extra")]`).
- `--asr_models_emvd`: List of models to evaluate on the EMVD dataset (default: `["Whisper_Large_V3", "Whisper_Large_V2", "Phi_4_Multimodal_Instruct", "Canary_1B", "Wav2vec2_Large_960h_Lv60_Self"]`).
- `--asr_models_songs`: List of models to evaluate on the songs and source-separated datasets (default: `["Whisper_Large_V3", "Whisper_Large_V2", "Phi_4_Multimodal_Instruct"]`). 
- `--metrics`: List of metrics to compute (default: `["WER", "BLEU", "ROUGE", ("EmbeddingSimilarity", "Gte_Qwen2_1d5B_Instruct"), ("EmbeddingSimilarity", "All_MiniLM_L6_V2"), ("EmbeddingSimilarity", "All_MPNet_Base_V2")]`).