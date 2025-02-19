#####################################################################################################################################################
################################################################### PREPARE STUFF ###################################################################
#####################################################################################################################################################

# External imports
import abc
import os
import sys
import huggingface_hub
import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline

# Project imports
from arguments import args

#####################################################################################################################################################
##################################################################### FUNCTIONS #####################################################################
#####################################################################################################################################################

def get_model (model_class_name, memoize=True, *args, **kwargs):

    """
        Wrapper to load models from this file.
        This function allows to memoize the models to avoid reloading them multiple times.
        :param model_class_name: The name of the class of the model to load.
        :param memoize: Whether to store the model in global memory to avoid reloading if calling the function multiple times.
        :param args: Arguments to pass to the model class.
        :param kwargs: Keyword arguments to pass to the model class.
    """

    # Check if the model is already in global memory to avoid reloading
    if memoize:
        if "loaded_models" not in globals():
            globals()["loaded_models"] = {}
        if model_class_name in globals()["loaded_models"]:
            return globals()["loaded_models"][model_class_name]

    # Load the model
    model_class = getattr(sys.modules[__name__], model_class_name)
    model = model_class(*args, **kwargs)

    # Memoize if needed
    if memoize:
        globals()["loaded_models"][model_class_name] = model
    return model

#####################################################################################################################################################
###################################################################### CLASSES ######################################################################
#####################################################################################################################################################

class BaseModel (abc.ABC):

    ###################################
    ########### CONSTRUCTOR ###########
    ###################################

    def __init__ (self, model_id, *args, **kwargs):
        
        """
            Constructor for the BaseModel class.
            :param model_id: Identifier for the model to load.
            :param args: Arguments to pass to the parent class.
            :param kwargs: Keyword arguments to pass to the parent class.
        """

        # Inherit from parent class
        super().__init__(*args, **kwargs)

        # Attributes
        self.model_id = model_id

        # Download model if needed
        self.__download_model()
    
    ###################################
    ######### PRIVATE METHODS #########
    ###################################

    def __download_model (self):

        """
            Download the model from HuggingFace if not already downloaded.
            Model is stored in the models directory provided in the arguments.
        """

        # Download the model if not already downloaded
        model_path = os.path.join(args.models_directory, self.model_id)
        if not os.path.exists(model_path):

            # Get model from HuggingFace
            print(f"Downloading model {self.model_id} to {model_path}", file=sys.stderr, flush=True)
            huggingface_hub.login(token=open(args.hf_key, "r").read().strip())
            huggingface_hub.snapshot_download(repo_id=self.model_id, local_dir=model_path)

            # Set permissions for shared use
            os.chmod(f"/Brain/public/models/{model_path}", 0o777)

    ###################################
    ######### ABSTRACT METHODS ########
    ###################################

    @abc.abstractmethod
    def __setup (self):

        """
            Models should be settup before being used.
            Children classes should implement this method and call it in their constructors.
        """

        # Abstract method
        raise NotImplementedError("Should be defined in children classes.")

#####################################################################################################################################################

class ASRModel (BaseModel, abc.ABC):

    ###################################
    ########### CONSTRUCTOR ###########
    ###################################

    def __init__ (self, *args, **kwargs):
        
        """
            Constructor for the ASRModel class.
            :param args: Arguments to pass to the parent class.
            :param kwargs: Keyword arguments to pass to the parent class.
        """

        # Inherit from parent class
        super().__init__(*args, **kwargs)

    ###################################
    ######### ABSTRACT METHODS ########
    ###################################

    @abc.abstractmethod
    def transcribe (self, audio_path):

        """
            Transcribe the audio signal x into text.
            :param audio_path: Path to the audio file to transcribe.
            :return: Transcribed text.
        """

        # Abstract method
        raise NotImplementedError("Should be defined in children classes.")

#####################################################################################################################################################

def WhisperLargeV2 (ASRModel):

    ###################################
    ########### CONSTRUCTOR ###########
    ###################################

    def __init__ (self, *args, **kwargs):

        # Inherit from parent class
        super().__init__(model_id="openai/whisper-large-v2", *args, **kwargs)

        # Attributes
        self.pipe = None

        # Setup
        self.__setup()

    ###################################
    ######### PRIVATE METHODS #########
    ###################################

    def __setup (self):
        
        """
            Setup the model for use.
        """

        # Instructions from https://huggingface.co/openai/whisper-large-v2
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
        model = AutoModelForSpeechSeq2Seq.from_pretrained(self.model_id,
                                                          torch_dtype=torch_dtype,
                                                          low_cpu_mem_usage=True,
                                                          use_safetensors=True)
        model.to(device)
        processor = AutoProcessor.from_pretrained(self.model_id)
        self.pipe = pipeline("automatic-speech-recognition",
                             model=model,
                             tokenizer=processor.tokenizer,
                             feature_extractor=processor.feature_extractor,
                             torch_dtype=torch_dtype,
                             device=device)

    ###################################
    ########## PUBLIC METHODS #########
    ###################################

    def transcribe (self, audio_path):
        
        """
            Transcribe the audio signal x into text.
            :param audio_path: Path to the audio file to transcribe.
            :return: Transcribed text.
        """

        # Go through the pipe
        return self.pipe(audio_path)

#####################################################################################################################################################

def WhisperLargeV3 (ASRModel):

    ###################################
    ########### CONSTRUCTOR ###########
    ###################################

    def __init__ (self, *args, **kwargs):

        # Inherit from parent class
        super().__init__(model_id="openai/whisper-large-v3", *args, **kwargs)

        # Attributes
        self.pipe = None

        # Setup
        self.__setup()

    ###################################
    ######### PRIVATE METHODS #########
    ###################################

    def __setup (self):
        
        """
            Setup the model for use.
        """

        # Instructions from https://huggingface.co/openai/whisper-large-v3
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
        model = AutoModelForSpeechSeq2Seq.from_pretrained(self.model_id,
                                                          torch_dtype=torch_dtype,
                                                          low_cpu_mem_usage=True,
                                                          use_safetensors=True)
        model.to(device)
        processor = AutoProcessor.from_pretrained(self.model_id)
        self.pipe = pipeline("automatic-speech-recognition",
                             model=model,
                             tokenizer=processor.tokenizer,
                             feature_extractor=processor.feature_extractor,
                             torch_dtype=torch_dtype,
                             device=device)

    ###################################
    ########## PUBLIC METHODS #########
    ###################################

    def transcribe (self, audio_path):
        
        """
            Transcribe the audio signal x into text.
            :param audio_path: Path to the audio file to transcribe.
            :return: Transcribed text.
        """

        # Go through the pipe
        return self.pipe(audio_path)

#####################################################################################################################################################
#####################################################################################################################################################



"""








def load_model(model_name):
    if model_name == "Whisper":
        return Whisper()
    elif model_name == "Canary":
        return Canary()
    elif model_name == "Wav2vec":
        return Wav2vec()
    else:
        raise ValueError("Model not found")
    

## Whisper
import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
# from datasets import load_dataset

def Whisper(BaseModel):
    def __init__(self):
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

        model_id = "openai/whisper-large-v3"

        model = AutoModelForSpeechSeq2Seq.from_pretrained(
            model_id, torch_dtype=torch_dtype, low_cpu_mem_usage=True, use_safetensors=True
        )
        model.to(device)

        processor = AutoProcessor.from_pretrained(model_id)

        pipe = pipeline(
            "automatic-speech-recognition",
            model=model,
            tokenizer=processor.tokenizer,
            feature_extractor=processor.feature_extractor,
            torch_dtype=torch_dtype,
            device=device,
        )

        self.pipe = pipe

    def transcribe(self, audio_path):
        # We can provide the entire dataset to the model if properly formatted.
        # (doc: dataset = load_dataset("distil-whisper/librispeech_long", "clean", split="validation"))
        return self.pipe(audio_path) # Expects the path to an audio file here

## Canary
# pip install git+https://github.com/NVIDIA/NeMo.git@r1.23.0#egg=nemo_toolkit[asr]
from nemo.collections.asr.models import EncDecMultiTaskModel

def Canary(BaseModel):
    def __init__(self):
        # load model
        canary_model = EncDecMultiTaskModel.from_pretrained('nvidia/canary-1b')

        # update dcode params
        decode_cfg = canary_model.cfg.decoding
        decode_cfg.beam.beam_size = 1
        canary_model.change_decoding_strategy(decode_cfg)

        self.model = canary_model

    def transcribe(self, audio_path):
        # might need to resample audio: 
        # "This model accepts single channel (mono) audio sampled at 16000 Hz, along with the task/languages/PnC tags as input."
        return self.model.transcribe(paths2audio_files=[audio_path]) #['path1.wav', 'path2.wav'])#,batch_size=16)

## Wav2vec
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC
# from datasets import load_dataset
# import torch

def Wav2vec(BaseModel):
    def __init__(self):
        # load model and processor
        processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-large-960h-lv60-self")
        model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-large-960h-lv60-self")

        self.processor = processor
        self.model = model

    def transcribe(self, audio_path):
        # Can work with the etire dataset if properly formatted
        # (doc: ds = load_dataset("patrickvonplaten/librispeech_asr_dummy", "clean", split="validation"))
 
        # tokenize
        # Not clear if audio_path or audio_signal is expected here
        input_values = self.processor(audio_path, return_tensors="pt", padding="longest").input_values

        # retrieve logits
        logits = self.model(input_values).logits

        # take argmax and decode
        predicted_ids = torch.argmax(logits, dim=-1)
        transcription = self.processor.batch_decode(predicted_ids)
        return transcription
"""