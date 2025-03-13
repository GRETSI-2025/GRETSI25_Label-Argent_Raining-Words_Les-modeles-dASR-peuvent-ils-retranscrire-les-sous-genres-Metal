#####################################################################################################################################################
###################################################################### LICENSE ######################################################################
#####################################################################################################################################################
#
#    Copyright (C) 2025  Bastien Pasdeloup & Axel Marmoret
#
#    This program is free software: you can redistribute it and/or modify
#    it under the terms of the GNU General Public License as published by
#    the Free Software Foundation, either version 3 of the License, or
#    (at your option) any later version.
#
#    This program is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#    GNU General Public License for more details.
#
#    You should have received a copy of the GNU General Public License
#    along with this program.  If not, see <https://www.gnu.org/licenses/>.
#
#####################################################################################################################################################
################################################################### DOCUMENTATION ###################################################################
#####################################################################################################################################################

"""
    This module contains the classes to evaluate metrics.
    Metrics should be loaded using the "get_metric" function.
"""

#####################################################################################################################################################
################################################################### PREPARE STUFF ###################################################################
#####################################################################################################################################################

# External imports
import abc
import evaluate
import sys
from typing import *
import rouge
import sacrebleu.metrics

# Project imports
import lib.models.loader

#####################################################################################################################################################
###################################################################### CLASSES ######################################################################
#####################################################################################################################################################

class TextMetrics (abc.ABC):

    """
        Abstract class to evaluate text metrics.
    """

    #############################################################################################################################################

    def __init__ ( self,
                   best:     Callable[[float, float], float],
                   *args:    Optional[list[any]],
                   **kwargs: Optional[dict[any, any]]
                 ) ->        None:

        """
            Constructor for the class.
            In:
                * best:   The best value for the metric (max or min).
                * args:   Extra arguments.
                * kwargs: Extra keyword arguments.
            Out:
                * A new instance of the class.
        """

        # Inherit from parent class
        super().__init__(*args, **kwargs)

        # Attributes
        self.best = best

    #############################################################################################################################################

    def compute ( self,
                  text_ref:  str,
                  text_pred: str
                ) ->         float:

        """
            Function to compute the metric.
            Takes care of edge cases.
            In:
                * text_ref:  The reference text.
                * text_pred: The predicted text.
            Out:
                * The computed metric.
        """

        # In case of empty texts, return the worst score
        if text_ref == "" or text_pred == "" \
        or "<|nospeech|>" in text_ref or "<|nospeech|>" in text_pred \
        or "lyrics are not provided" in text_ref or "lyrics are not provided" in text_pred:
            return 0.0 if self.best == max else 1.0
        
        # Compute the metric
        return self._compute(text_ref, text_pred)
    
    #############################################################################################################################################

    @abc.abstractmethod
    def _compute ( self,
                   text_ref:  str,
                   text_pred: str
                 ) ->         float:

        """
            Function to compute the metric, to be defined in children classes.
            In:
                * text_ref:  The reference text.
                * text_pred: The predicted text.
            Out:
                * The computed metric.
        """

        # Abstract method
        raise NotImplementedError("Should be defined in children classes.")

    #############################################################################################################################################
    #############################################################################################################################################

#####################################################################################################################################################

class WER (TextMetrics):

    """
        Class to compute the Word Error Rate.
    """

    #############################################################################################################################################

    def __init__ ( self,
                   *args:    Optional[list[any]],
                   **kwargs: Optional[dict[any, any]]
                 ) ->        None:
        
        """
            Constructor for the class.
            In:
                * args:   Extra arguments.
                * kwargs: Extra keyword arguments.
            Out:
                * A new instance of the class.
        """

        # Inherit from parent class
        super().__init__(best=min, *args, **kwargs)

        # Attributes
        self.model = None

    #############################################################################################################################################

    @override
    def _compute ( self,
                  text_ref:  str,
                  text_pred: str
                ) ->         float:

        """
            Function to compute the metric.
            In:
                * text_ref:  The reference text.
                * text_pred: The predicted text.
            Out:
                * The computed metric.
        """

        # Load the model as late as possible
        if self.model is None:
            self.model = evaluate.load("wer")

        # Compute the error
        error = min(1.0, self.model.compute(predictions=[text_pred], references=[text_ref]))
        return error

    #############################################################################################################################################
    #############################################################################################################################################

#####################################################################################################################################################

class EmbeddingSimilarity (TextMetrics):

    """
        Class to compute the similarity between two texts using embeddings.
    """

    #############################################################################################################################################

    def __init__ ( self,
                   model_name: str,
                   *args:      Optional[list[any]],
                   **kwargs:   Optional[dict[any, any]]
                 ) ->          None:
        
        # Inherit from parent class
        super().__init__(best=max, *args, **kwargs)

        # Attributes
        self.model_name = model_name
        self.model = None

    #############################################################################################################################################

    @override
    def _compute ( self,
                   text_ref:  str,
                   text_pred: str
                 ) ->         float:
        
        """
            Function to compute the metric.
            In:
                * text_ref:  The reference text.
                * text_pred: The predicted text.
            Out:
                * The computed metric.
        """

        # Load the model as late as possible
        if self.model is None:
            self.model = lib.models.loader.get_model(self.model_name)

        # Compute the embeddings
        embedding_ref = self.model.run(text_ref[:self.model.max_seq_length])
        embedding_pred = self.model.run(text_pred[:self.model.max_seq_length])

        # Compute the similarity
        similarity = float(embedding_ref @ embedding_pred / (embedding_ref.norm() * embedding_pred.norm()))
        return similarity

    #############################################################################################################################################
    #############################################################################################################################################

#####################################################################################################################################################

class BLEU (TextMetrics):

    """
        Class to compute the BLEU score.
    """

    #############################################################################################################################################

    def __init__ ( self,
                   *args:    Optional[list[any]],
                   **kwargs: Optional[dict[any, any]]
                 ) ->        None:
        
        """
            Constructor for the class.
            In:
                * args:   Extra arguments.
                * kwargs: Extra keyword arguments.
            Out:
                * A new instance of the class.
        """

        # Inherit from parent class
        super().__init__(best=max, *args, **kwargs)

        # Attributes
        self.scorer = None

    #############################################################################################################################################

    @override
    def _compute ( self,
                   text_ref:  str,
                   text_pred: str
                 ) ->         float:

        """
            Function to compute the metric.
            In:
                * text_ref:  The reference text.
                * text_pred: The predicted text.
            Out:
                * The computed metric.
        """

        # Load the scorer as late as possible
        if self.scorer is None:
            self.scorer = sacrebleu.metrics.BLEU(effective_order=True)

        # Compute the score
        score = self.scorer.sentence_score(hypothesis=text_pred, references=[text_ref])
        return score.score / 100

    #############################################################################################################################################
    #############################################################################################################################################

#####################################################################################################################################################

class ROUGE (TextMetrics):

    """
        Class to compute the ROUGE score.
    """

    #############################################################################################################################################

    def __init__ ( self,
                   *args:    Optional[list[any]],
                   **kwargs: Optional[dict[any, any]]
                 ) ->        None:
        
        """
            Constructor for the class.
            In:
                * args:   Extra arguments.
                * kwargs: Extra keyword arguments.
            Out:
                * A new instance of the class.
        """
        
        # Inherit from parent class
        super().__init__(best=max, *args, **kwargs)

        # Attributes
        self.scorer = None

    #############################################################################################################################################

    @override
    def _compute ( self,
                   text_ref:  str,
                   text_pred: str
                 ) ->         float:

        """
            Function to compute the metric.
            In:
                * text_ref:  The reference text.
                * text_pred: The predicted text.
            Out:
                * The computed metric.
        """

        # Load the scorer as late as possible
        if self.scorer is None:
            self.scorer = rouge.Rouge()

        # Compute the score
        score = self.scorer.get_scores(hyps=text_pred, refs=text_ref)
        return score[0]["rouge-l"]["f"]

    #############################################################################################################################################
    #############################################################################################################################################

#####################################################################################################################################################
##################################################################### FUNCTIONS #####################################################################
#####################################################################################################################################################

def get_metric ( metric_class_name: Union[str, list[str], tuple[str]]
               ) ->                 TextMetrics:

    """
        Function to get a metric from its name.
        In:
            * metric_class_name: The name of the metric class.
        Out:
            * The metric.
    """

    # Metric can be passed as a tuple with arguments
    extra_args = []
    if type(metric_class_name) in [list, tuple]:
        metric_class_name, *extra_args = metric_class_name
    
    # Load the metric
    metric_class = getattr(sys.modules[__name__], metric_class_name)
    return metric_class(*extra_args)
    
#####################################################################################################################################################
#####################################################################################################################################################