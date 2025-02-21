#####################################################################################################################################################
################################################################### PREPARE STUFF ###################################################################
#####################################################################################################################################################

# External imports
import abc
import evaluate
import sys
from typing import override

# Project imports
import lib_models

#####################################################################################################################################################
##################################################################### FUNCTIONS #####################################################################
#####################################################################################################################################################

def get_metric (metric_class_name, memoize=True):

    # Check if the metric is already in global memory to avoid reloading
    if memoize:
        memoization_key = str(metric_class_name)
        if "loaded_metrics" not in globals():
            globals()["loaded_metrics"] = {}
        if memoization_key in globals()["loaded_metrics"]:
            return globals()["loaded_metrics"][memoization_key]

    # Metric can be passed as a tuple with arguments
    extra_args = []
    if type(metric_class_name) in [list, tuple]:
        metric_class_name, *extra_args = metric_class_name
    
    # Load the metric
    metric_class = getattr(sys.modules[__name__], metric_class_name)
    metric = metric_class(*extra_args)

    # Memoize if needed
    if memoize:
        globals()["loaded_metrics"][memoization_key] = metric
    return metric

#####################################################################################################################################################
################################################################## ABSTRACT CLASSES #################################################################
#####################################################################################################################################################

class TextMetrics (abc.ABC):



    def __init__ (self, best, *args, **kwargs):
        
        # Inherit from parent class
        super().__init__(*args, **kwargs)

        # Attributes
        self.best = best



    @abc.abstractmethod
    def compute (self, text_1, text_2):

        # Abstract method
        raise NotImplementedError("Should be defined in children classes.")



#####################################################################################################################################################
################################################################### METRIC CLASSES ##################################################################
#####################################################################################################################################################

class WER (TextMetrics):



    def __init__ (self, *args, **kwargs):
        
        # Inherit from parent class
        super().__init__(best=min, *args, **kwargs)

        # Attributes
        self.wer = evaluate.load("wer")



    @override
    def compute (self, text_1, text_2):

        # Compute the error
        error = self.wer.compute(predictions=[text_2], references=[text_1])
        return error



#####################################################################################################################################################

class EmbeddingSimilarity (TextMetrics):



    def __init__ (self, model_name, *args, **kwargs):
        
        # Inherit from parent class
        super().__init__(best=max, *args, **kwargs)

        # Attributes
        self.model = lib_models.get_model(model_name)



    @override
    def compute (self, text_1, text_2):

        # In case of empty texts, return 0
        if text_1 == "" or text_2 == "" or "<|nospeech|>" in text_1 or "<|nospeech|>" in text_2:
            return 0.0

        # Compute the embeddings
        embedding_1 = self.model.embed(text_1)
        embedding_2 = self.model.embed(text_2)

        # Compute the similarity
        similarity = float(embedding_1 @ embedding_2 / (embedding_1.norm() * embedding_2.norm()))
        return similarity



#####################################################################################################################################################
#####################################################################################################################################################