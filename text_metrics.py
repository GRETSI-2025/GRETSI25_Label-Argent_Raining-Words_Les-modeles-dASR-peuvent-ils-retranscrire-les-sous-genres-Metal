#####################################################################################################################################################
################################################################### PREPARE STUFF ###################################################################
#####################################################################################################################################################

# External imports
import abc
import evaluate

# Project imports
from arguments import args
from model_loaders import *

#####################################################################################################################################################
##################################################################### FUNCTIONS #####################################################################
#####################################################################################################################################################

def get_metric (metric_class_name, memoize=True, *args, **kwargs):

    """
        Wrapper to load metrics from this file.
        :param metric_class_name: Name of the metric class to load.
        :param args: Arguments to pass to the metric class.
        :param kwargs: Keyword arguments to pass to the metric class.
    """

    # Load the metric
    metric_class = getattr(sys.modules[__name__], metric_class_name)
    metric = metric_class(*args, **kwargs)
    return metric

#####################################################################################################################################################
###################################################################### CLASSES ######################################################################
#####################################################################################################################################################

class TextMetrics (abc.ABC):

    ###################################
    ########### CONSTRUCTOR ###########
    ###################################

    def __init__ (self, best, *args, **kwargs):
        
        """
            Constructor for the TextMetrics class.
            :param best: Function to use to determine which is the best.
            :param args: Arguments to pass to the parent class.
            :param kwargs: Keyword arguments to pass to the parent class.
        """

        # Inherit from parent class
        super().__init__(*args, **kwargs)

        # Attributes
        self.best = best

    ###################################
    ######### ABSTRACT METHODS ########
    ###################################

    @abc.abstractmethod
    def compute (self, text_1, text_2):

        """
            Compute the similarity/difference between two texts.
            :param text_1: First text to compare.
            :param text_2: Second text to compare.
            :return: Similarity/difference between the two texts.
        """

        # Abstract method
        raise NotImplementedError("Should be defined in children classes.")

#####################################################################################################################################################

class WER (TextMetrics):

    ###################################
    ########### CONSTRUCTOR ###########
    ###################################

    def __init__ (self, *args, **kwargs):
        
        """
            Constructor for the WER class.
            :param args: Arguments to pass to the parent class.
            :param kwargs: Keyword arguments to pass to the parent class.
        """

        # Inherit from parent class
        super().__init__(best=min, *args, **kwargs)

    ###################################
    ########## PUBLIC METHODS #########
    ###################################

    def compute (self, text_1, text_2):

        """
            Compute the similarity/difference between two texts.
            :param text_1: First text to compare.
            :param text_2: Second text to compare.
            :return: Similarity/difference between the two texts.
        """

        # Get the metric
        wer = evaluate.load("wer")

        # Compute the error
        error = wer.compute(predictions=[text_2], references=[text_1])
        return error

#####################################################################################################################################################
#####################################################################################################################################################


"""


#####################################################################################################################################################

def compute_metrics (predicted_lyrics, actual_lyrics):

    ""
        Compute various metrics between two sets of lyrics.
        :param predicted_lyrics: The predicted lyrics.
        :param actual_lyrics: The reference lyrics.
        :return: The similarities/errors between the two sets of lyrics.
    ""

    # Function to compute similarity in an embedding
    def _embedding_similarity (st_model):
        pipe = get_pipeline("feature-extraction", st_model)
        embedding_actual = pipe(actual_lyrics, return_tensors=True)[0].mean(dim=0)
        embedding_predicted = pipe(predicted_lyrics, return_tensors=True)[0].mean(dim=0)
        return float(embedding_actual @ embedding_predicted / (embedding_actual.norm() * embedding_predicted.norm()))
    
    # Function to compute error with Word Error Rate
    def _word_error_rate ():
        wer = evaluate.load("wer")
        error = wer.compute(predictions=[actual_lyrics], references=[predicted_lyrics])
        return error

    # Return a dictionary of metrics
    metrics = {}
    for metric in args.metrics:
        if "/" in metric:
            metrics[metric] = _embedding_similarity(metric)
        elif metric == "WER":
            metrics["WER"] = _word_error_rate()
    return metrics


"""