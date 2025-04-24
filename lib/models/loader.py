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
    This module contains the function to load models.
"""

#####################################################################################################################################################
################################################################### PREPARE STUFF ###################################################################
#####################################################################################################################################################

# External imports
import sys
from typing import *

# Project imports
import lib.models.base
import lib.models.automatic_speech_recognition
import lib.models.text_embedding
import lib.models.source_separation

#####################################################################################################################################################
##################################################################### FUNCTIONS #####################################################################
#####################################################################################################################################################

def get_model ( model_class_name: Union[str, list[str], tuple[str]],
                memoize:          bool = True
              ) ->                lib.models.base.BaseModel:

    """
        Function to get a model instance.
        The model will be memoized in global memory to avoid reloading.
        In:
            * model_class_name: The name of the model class, possibly with extra arguments.
            * The model instance.
    """

    # Check if the model is already in global memory to avoid reloading
    if memoize:
        memoization_key = str(model_class_name)
        if "loaded_models" not in globals():
            globals()["loaded_models"] = {}
        if memoization_key in globals()["loaded_models"]:
            return globals()["loaded_models"][memoization_key]

    # Model can be passed as a tuple with arguments
    extra_args = []
    if type(model_class_name) in [list, tuple]:
        model_class_name, *extra_args = model_class_name
    
    # Load the model
    print(f"Loading model \"{model_class_name}\"", file=sys.stderr, flush=True)
    for module in sys.modules:
        if module.startswith("lib.models.") and hasattr(sys.modules[module], model_class_name):
            model_class = getattr(sys.modules[module], model_class_name)
            model = model_class(*extra_args)
            break

    # Memoize if needed
    if memoize:
        globals()["loaded_models"][memoization_key] = model
    return model

#####################################################################################################################################################

def free_model ( model_class_name: Union[str, list[str], tuple[str]]
               ) ->                None:

    """
        Function to free a model from global memory.
        In:
            * model_class_name: The name of the model class, possibly with extra arguments.
        Out:
            * None.
    """

    # Model can be passed as a tuple with arguments
    if type(model_class_name) in [list, tuple]:
        model_class_name = model_class_name[0]
    
    # Free the model
    memoization_key = str(model_class_name)
    if "loaded_models" in globals() and memoization_key in globals()["loaded_models"]:
        del globals()["loaded_models"][memoization_key]
        print(f"Unloaded model \"{model_class_name}\"", file=sys.stderr, flush=True)

#####################################################################################################################################################
#####################################################################################################################################################