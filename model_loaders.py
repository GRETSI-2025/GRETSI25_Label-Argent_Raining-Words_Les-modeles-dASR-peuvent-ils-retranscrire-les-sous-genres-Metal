
def load_model(model_name):
    if model_name == "Whisper":
        return Whisper()
    elif model_name == "Canary":
        return Canary()
    elif model_name == "Wav2vec":
        return Wav2vec()
    else:
        raise ValueError("Model not found")
    
def BaseModel():
    def __init__(self):
        pass

    def transcribe(self, x):
        raise NotImplementedError("Should be defined in children classes.")


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
