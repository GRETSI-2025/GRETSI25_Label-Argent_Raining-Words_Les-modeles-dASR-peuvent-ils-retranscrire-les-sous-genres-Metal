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

# Project imports
import lib.models.loader

#####################################################################################################################################################
###################################################################### CLASSES ######################################################################
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
        self.model = None



    @override
    def compute (self, text_1, text_2):

        # Load the model as late as possible
        if self.model is None:
            self.model = evaluate.load("wer")

        # Compute the error
        error = self.model.compute(predictions=[text_2], references=[text_1])
        return error



#####################################################################################################################################################

class EmbeddingSimilarity (TextMetrics):



    def __init__ (self, model_name, *args, **kwargs):
        
        # Inherit from parent class
        super().__init__(best=max, *args, **kwargs)

        # Attributes
        self.model_name = model_name
        self.model = None



    @override
    def compute (self, text_1, text_2):

        # In case of empty texts, return 0
        if text_1 == "" or text_2 == "" or "<|nospeech|>" in text_1 or "<|nospeech|>" in text_2:
            return 0.0

        # Load the model as late as possible
        if self.model is None:
            self.model = lib.models.loader.get_model(self.model_name)

        # Compute the embeddings
        embedding_1 = self.model.run(text_1)
        embedding_2 = self.model.run(text_2)

        # Compute the similarity
        similarity = float(embedding_1 @ embedding_2 / (embedding_1.norm() * embedding_2.norm()))
        return similarity



#####################################################################################################################################################
##################################################################### FUNCTIONS #####################################################################
#####################################################################################################################################################

def get_metric ( metric_class_name: Union[str, list[str], tuple[str]]
               ) ->                 TextMetrics:

    # Metric can be passed as a tuple with arguments
    extra_args = []
    if type(metric_class_name) in [list, tuple]:
        metric_class_name, *extra_args = metric_class_name
    
    # Load the metric
    metric_class = getattr(sys.modules[__name__], metric_class_name)
    return metric_class(*extra_args)
    
#####################################################################################################################################################
#####################################################################################################################################################