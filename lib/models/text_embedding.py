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
    This module contains the classes to perform text embedding.
    Models should be loaded using the "get_model" function from "lib.models.loader".
"""

#####################################################################################################################################################
################################################################### PREPARE STUFF ###################################################################
#####################################################################################################################################################

# External imports
from typing import *
from transformers import pipeline

# Project imports
import lib.models.base

#####################################################################################################################################################
###################################################################### CLASSES ######################################################################
#####################################################################################################################################################

class Gte_Qwen2_1d5B_Instruct (lib.models.base.HuggingFaceModel):
    
    """
        Class to perform text embedding using the GTE Qwen2 1.5B Instruct model.
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
        super().__init__(model_id="Alibaba-NLP/gte-Qwen2-1.5B-instruct", *args, **kwargs)

        # Attributes
        self.pipe = None
        self.max_seq_length = 32000

    #############################################################################################################################################

    @override
    def _setup ( self,
               ) -> None:
        
        """
            Method to setup the model.
            In:
                * None.
            Out:
                * None.
        """

        # Use a pipeline
        self.pipe = pipeline("feature-extraction", self.model_id)

    #############################################################################################################################################

    @override
    def _apply ( self,
                 text: str,
               ) ->    str:

        """
            Method to apply the model to some data.
            In:
                * text: The text to embed.
            Out:
                * The embedding of the text.
        """
        
        # Go through the pipe
        return self.pipe(text, return_tensors=True)[0].mean(dim=0)
    
    #############################################################################################################################################
    #############################################################################################################################################

#####################################################################################################################################################

class All_MiniLM_L6_V2 (lib.models.base.HuggingFaceModel):
    
    """
        Class to perform text embedding using the All MiniLM L6 V2 model.
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
        super().__init__(model_id="sentence-transformers/all-MiniLM-L6-v2", *args, **kwargs)

        # Attributes
        self.pipe = None
        self.max_seq_length = 256

    #############################################################################################################################################

    @override
    def _setup ( self,
               ) -> None:
        
        """
            Method to setup the model.
            In:
                * None.
            Out:
                * None.
        """

        # Use a pipeline
        self.pipe = pipeline("feature-extraction", self.model_id)

    #############################################################################################################################################

    @override
    def _apply ( self,
                 text: str,
               ) ->    str:

        """
            Method to apply the model to some data.
            In:
                * text: The text to embed.
            Out:
                * The embedding of the text.
        """
        
        # Go through the pipe
        return self.pipe(text, return_tensors=True)[0].mean(dim=0)

    #############################################################################################################################################
    #############################################################################################################################################

#####################################################################################################################################################

class All_MPNet_Base_V2 (lib.models.base.HuggingFaceModel):
    
    """
        Class to perform text embedding using the All MPNet Base V2 model.
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
        super().__init__(model_id="sentence-transformers/all-mpnet-base-v2", *args, **kwargs)

        # Attributes
        self.pipe = None
        self.max_seq_length = 128

    #############################################################################################################################################

    @override
    def _setup ( self,
               ) -> None:
        
        """
            Method to setup the model.
            In:
                * None.
            Out:
                * None.
        """

        # Use a pipeline
        self.pipe = pipeline("feature-extraction", self.model_id)

    #############################################################################################################################################

    @override
    def _apply ( self,
                 text: str,
               ) ->    str:

        """
            Method to apply the model to some data.
            In:
                * text: The text to embed.
            Out:
                * The embedding of the text.
        """
        
        # Go through the pipe
        return self.pipe(text, return_tensors=True)[0].mean(dim=0)

    #############################################################################################################################################
    #############################################################################################################################################

#####################################################################################################################################################
#####################################################################################################################################################