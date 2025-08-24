# extreme_vocals_asr

<hr>

**_Dépôt labelisé dans le cadre du [Label Reproductible du GRESTI'25](https://gretsi.fr/colloque2025/recherche-reproductible/)_**

| Label décerné | Auteur | Rapporteur | Éléments reproduits | Liens |
|:-------------:|:------:|:----------:|:-------------------:|:------|
| ![](label_argent.png) | Bastien PASDELOUP<br>[@BastienPasdeloup](https://github.com/BastienPasdeloup) | Félix RIEDEL<br>[@felix-riedel-UJM](https://github.com/felix-riedel-UJM) |  Tous les résultats | 📌&nbsp;[Dépôt&nbsp;original](https://github.com/BastienPasdeloup/extreme_vocals_asr)<br>⚙️&nbsp;[Issue](https://github.com/GRETSI-2025/Label-Reproductible/issues/15)<br>📝&nbsp;[Rapport](https://github.com/akrah/test/tree/main/rapports/Rapport_issue_15) |

<hr>

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

Scripts have been tested with `Python 3.12.3`.

To install the required packages, you can use `pip` to create a virtual environment and install the dependencies as follows:
```bash
pip install -r requirements.txt
```

**Note:** Among the dependencies, `yt-dlp` is used to download the songs from YouTube. To work properly, it requires `ffmpeg` to be installed on your system. You can install it using `apt-get install ffmpeg` on Ubuntu or `brew install ffmpeg` on MacOS.

**Note:** One of the models used, `Phi_4_Multimodal_Instruct`, supports flash attention. If your GPU supports it, you can install it following instructions [https://huggingface.co/docs/transformers/perf_infer_gpu_one#flashattention](here). If not installed, the model will use the standard attention implementation.

**Note**:** The scripts will take care of downloading the models and datasets. Please make sure that you have a HuggingFace key for downloading models. If in a file, you can pass it to the script using the `--hf_key_path` parameter below.

<details>
<summary>**Note:** In case of version conflicts, click here for the output of `pip freeze` in a working fresh `venv`.</summary>

```bash
absl-py==2.2.2
accelerate==1.6.0
aiohappyeyeballs==2.6.1
aiohttp==3.11.18
aiosignal==1.3.2
alembic==1.15.2
annotated-types==0.7.0
antlr4-python3-runtime==4.9.3
asttokens==3.0.0
attrs==25.3.0
audioread==3.0.1
backoff==2.2.1
braceexpand==0.1.7
certifi==2025.1.31
cffi==1.17.1
charset-normalizer==3.4.1
click==8.1.8
cloudpickle==3.1.1
colorama==0.4.6
colorlog==6.9.0
contourpy==1.3.2
cycler==0.12.1
cytoolz==1.0.1
datasets==3.5.0
decorator==5.2.1
defusedxml==0.7.1
demucs==4.0.1
dill==0.3.8
Distance==0.1.3
docker-pycreds==0.4.0
docopt==0.6.2
dora_search==0.1.12
editdistance==0.8.1
einops==0.8.1
evaluate==0.4.3
executing==2.2.0
fiddle==0.3.0
filelock==3.18.0
fonttools==4.57.0
frozenlist==1.6.0
fsspec==2024.12.0
future==1.0.0
g2p-en==2.1.0
gitdb==4.0.12
GitPython==3.1.44
graphviz==0.20.3
greenlet==3.2.1
grpcio==1.71.0
huggingface-hub==0.30.2
hydra-core==1.3.2
idna==3.10
inflect==7.5.0
intervaltree==3.1.0
ipython==9.1.0
ipython_pygments_lexers==1.1.1
jedi==0.19.2
Jinja2==3.1.6
jiwer==3.1.0
joblib==1.4.2
julius==0.2.7
kaldi-python-io==1.2.2
kaldiio==2.18.1
kaleido==0.2.1
kiwisolver==1.4.8
lameenc==1.8.1
lazy_loader==0.4
Levenshtein==0.27.1
lhotse==1.31.0
libcst==1.7.0
librosa==0.11.0
lightning==2.4.0
lightning-utilities==0.14.3
lilcom==1.8.1
llvmlite==0.44.0
loguru==0.7.3
lxml==5.4.0
Mako==1.3.10
Markdown==3.8
markdown-it-py==3.0.0
MarkupSafe==3.0.2
marshmallow==4.0.0
matplotlib==3.10.1
matplotlib-inline==0.1.7
mdurl==0.1.2
mediapy==1.1.6
more-itertools==10.7.0
mpmath==1.3.0
msgpack==1.1.0
multidict==6.4.3
multiprocess==0.70.16
narwhals==1.36.0
nemo-toolkit==2.2.1
networkx==3.4.2
nltk==3.9.1
numba==0.61.0
numpy==2.1.3
nvidia-cublas-cu12==12.6.4.1
nvidia-cuda-cupti-cu12==12.6.80
nvidia-cuda-nvrtc-cu12==12.6.77
nvidia-cuda-runtime-cu12==12.6.77
nvidia-cudnn-cu12==9.5.1.17
nvidia-cufft-cu12==11.3.0.4
nvidia-cufile-cu12==1.11.1.6
nvidia-curand-cu12==10.3.7.77
nvidia-cusolver-cu12==11.7.1.2
nvidia-cusparse-cu12==12.5.4.2
nvidia-cusparselt-cu12==0.6.3
nvidia-nccl-cu12==2.26.2
nvidia-nvjitlink-cu12==12.6.85
nvidia-nvtx-cu12==12.6.77
odfpy==1.4.1
omegaconf==2.3.0
onnx==1.17.0
openunmix==1.3.0
optuna==4.3.0
packaging==24.2
pandas==2.2.3
parso==0.8.4
peft==0.15.2
pexpect==4.9.0
pillow==11.2.1
plac==1.4.5
platformdirs==4.3.7
plotly==6.0.1
pooch==1.8.2
portalocker==3.1.1
prompt_toolkit==3.0.51
propcache==0.3.1
protobuf==3.20.3
psutil==7.0.0
ptyprocess==0.7.0
pure_eval==0.2.3
pyannote.core==5.0.0
pyannote.database==5.1.3
pyannote.metrics==3.2.1
pyarrow==19.0.1
pybind11==2.13.6
pycparser==2.22
pydantic==2.11.3
pydantic_core==2.33.1
pydub==0.25.1
Pygments==2.19.1
pyloudnorm==0.1.1
pyparsing==3.2.3
python-dateutil==2.9.0.post0
pytorch-lightning==2.5.1
pytz==2025.2
PyYAML==6.0.2
RapidFuzz==3.13.0
regex==2024.11.6
requests==2.32.3
resampy==0.4.3
retrying==1.3.4
rich==14.0.0
rouge==1.0.1
ruamel.yaml==0.18.10
ruamel.yaml.clib==0.2.12
sacrebleu==2.5.1
sacremoses==0.1.1
safetensors==0.5.3
scikit-learn==1.6.1
scipy==1.15.2
sentencepiece==0.2.0
sentry-sdk==2.26.1
setproctitle==1.3.5
setuptools==79.0.0
shellingham==1.5.4
six==1.17.0
smmap==5.0.2
sortedcontainers==2.4.0
soundfile==0.13.1
sox==1.5.0
soxr==0.5.0.post1
SQLAlchemy==2.0.40
stack-data==0.6.3
submitit==1.5.2
sympy==1.13.3
tabulate==0.9.0
tensorboard==2.19.0
tensorboard-data-server==0.7.2
termcolor==3.0.1
text-unidecode==1.3
texterrors==0.5.1
threadpoolctl==3.6.0
tokenizers==0.21.1
toolz==1.0.0
torch==2.7.0
torchaudio==2.7.0
torchmetrics==1.7.1
torchvision==0.22.0
tqdm==4.67.1
traitlets==5.14.3
transformers==4.48.3
treetable==0.2.5
triton==3.3.0
typeguard==4.4.2
typer==0.15.2
typing-inspection==0.4.0
typing_extensions==4.13.2
tzdata==2025.2
urllib3==2.4.0
wandb==0.19.10
wcwidth==0.2.13
webdataset==0.2.111
Werkzeug==3.1.3
wget==3.2
wrapt==1.17.2
xxhash==3.5.0
yarl==1.20.0
yt-dlp==2025.3.31
```

</details>

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

Results will be saved in the `output_directory` specified in the command line.
The present repository contains an `output` folder with the results of the paper, which can be used to compare with your own results.

### Parameters

- `--dataset_path`: Path to the dataset directory.
- `--output directory`: Path to the output directory.
- `--models_directory`: Path to where models are downloaded.
- `--hf_key_path`: Path to a file containing your Hugging Face token.
- `--source_separation_models`: List of models to use for source separation (default: `[("Demucs", "mdx_extra")]`)*.
- `--asr_models_emvd`: List of models to evaluate on the EMVD dataset (default: `["Whisper_Large_V3", "Whisper_Large_V2", "Phi_4_Multimodal_Instruct", "Canary_1B", "Wav2vec2_Large_960h_Lv60_Self"]`)*.
- `--asr_models_songs`: List of models to evaluate on the songs and source-separated datasets (default: `["Whisper_Large_V3", "Whisper_Large_V2", "Phi_4_Multimodal_Instruct"]`)*. 
- `--metrics`: List of metrics to compute (default: `["WER", "BLEU", "ROUGE", ("EmbeddingSimilarity", "Gte_Qwen2_1d5B_Instruct"), ("EmbeddingSimilarity", "All_MiniLM_L6_V2"), ("EmbeddingSimilarity", "All_MPNet_Base_V2")]`)*.

**Note: (*)** Values in those lists should be the name of the class in `lib/models/xxx.py` or a tuple `(model name, model arguments)` if needed.
