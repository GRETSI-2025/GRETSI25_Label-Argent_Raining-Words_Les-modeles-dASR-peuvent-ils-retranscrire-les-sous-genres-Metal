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
    This module contains abstract classes to handle models.
    These classes are meant to be inherited by other classes.
"""

#####################################################################################################################################################
################################################################### PREPARE STUFF ###################################################################
#####################################################################################################################################################

# External imports
import abc
import os
import sys
from typing import *
import huggingface_hub

# Project imports
from lib.arguments import script_args

#####################################################################################################################################################
###################################################################### CLASSES ######################################################################
#####################################################################################################################################################

class BaseModel (abc.ABC):

    """
        This class describes a base model.
        It is meant to be inherited by other classes.
        A model should be able to be downloaded, setup and applied.
        This class also describes a run method that makes sure everything is ready before applying the model.
    """

    #############################################################################################################################################

    def __init__ ( self,
                   model_id:         str,
                   models_directory: Optional[str] = None,
                   *args:            Optional[list[any]],
                   **kwargs:         Optional[dict[any, any]]
                 ) ->                None:

        """
            Constructor for the class.
            In:
                * model_id:         The model identifier.
                * models_directory: The local directory where models are stored.
                * args:             Extra arguments.
                * kwargs:           Extra keyword arguments.
            Out:
                * A new instance of the class.
        """

        # Inherit from parent class
        super().__init__(*args, **kwargs)

        # Attributes
        self.model_id = model_id
        self.models_directory = models_directory if models_directory is not None else script_args().models_directory
        self.model_path = os.path.join(self.models_directory, self.model_id)
        self._setup_done = False

    #############################################################################################################################################

    @abc.abstractmethod
    def _download ( self
                  ) -> None:

        """
            Method to download the model.
            In:
                * None.
            Out:
                * None.
        """

        # Crash if not overriden
        raise NotImplementedError("Should be defined in children classes.")

    #############################################################################################################################################

    @abc.abstractmethod
    def _setup ( self
               ) -> None:

        """
            Method to setup the model.
            In:
                * None.
            Out:
                * None.
        """

        # Crash if not overriden
        raise NotImplementedError("Should be defined in children classes.")

    #############################################################################################################################################

    @abc.abstractmethod
    def _apply ( self,
                 *args:    Optional[list[any]],
                 **kwargs: Optional[dict[any, any]]
               ) ->        any:

        """
            Method to apply the model to some data.
            In:
                * args:   Extra arguments.
                * kwargs: Extra keyword arguments.
            Out:
                * The result of the model application.
        """

        # Crash if not overriden
        raise NotImplementedError("Should be defined in children classes.")

    #############################################################################################################################################

    def run ( self,
              *args:    Optional[list[any]],
              **kwargs: Optional[dict[any, any]]
            ) ->        any:
        

        """
            Make sure everything is ready, and use the model.
            In:
                * args:   Extra arguments.
                * kwargs: Extra keyword arguments.
            Out:
                * The result of the model application.
        """

        # Make sure the model is downloaded
        if not os.path.exists(self.model_path):
            self._download()

        # Make sure it is setup
        if not self._setup_done:
            self._setup()
            self._setup_done = True

        # Correct access rights for shared usage
        for root, dirs, files in os.walk(self.model_path):
            for directory in dirs:
                os.chmod(os.path.join(root, directory), 0o777)
            for file in files:
                os.chmod(os.path.join(root, file), 0o777)
                
        # Apply the model
        return self._apply(*args, **kwargs)

    #############################################################################################################################################
    #############################################################################################################################################

#####################################################################################################################################################

class HuggingFaceModel (BaseModel, abc.ABC):

    """
        This class describes a model that can be downloaded from HuggingFace.
        It is meant to be inherited by other classes.
    """

    #############################################################################################################################################

    def __init__ ( self,
                   hf_key_path: Optional[str] = None,
                   *args:       Optional[list[any]],
                   **kwargs:    Optional[dict[any, any]]
                 ) ->           None:

        """
            Constructor for the class.
            In:
                * hf_key_path: The path to the HuggingFace API key.
                * args:        Extra arguments.
                * kwargs:      Extra keyword arguments.
            Out:
                * A new instance of the class.
        """

        # Inherit from parent class
        super().__init__(*args, **kwargs)

        # Attributes
        self.hf_key_path = hf_key_path if hf_key_path is not None else script_args().hf_key_path
    
    #############################################################################################################################################

    @override
    def _download ( self,
                  ) -> None:

        """
            Method to download the model.
            In:
                * None.
            Out:
                * None.
        """

        # Get model from HuggingFace
        print(f"Downloading model {self.model_id} to {self.model_path}", file=sys.stderr, flush=True)
        huggingface_hub.login(token=open(self.hf_key_path, "r").read().strip())
        huggingface_hub.snapshot_download(repo_id=self.model_id, local_dir=self.model_path)

    #############################################################################################################################################
    #############################################################################################################################################

#####################################################################################################################################################

class NvidiaNemoModel (BaseModel, abc.ABC):

    """
        This class describes a model that can be downloaded from Nvidia NeMo.
        It is meant to be inherited by other classes.
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
        super().__init__(*args, **kwargs)
    
    #############################################################################################################################################

    @override
    def _download ( self,
                  ) -> None:

        """
            Method to download the model.
            In:
                * None.
            Out:
                * None.
        """

        # Requires to set NEMO_CACHE_DIR to local directory
        os.environ["NEMO_CACHE_DIR"] = self.models_directory

        # Model is downloaded when setting up
        pass

    #############################################################################################################################################
    #############################################################################################################################################

#####################################################################################################################################################

class ManualDlModel (BaseModel, abc.ABC):

    """
        This class describes a model that needs to be downloaded manually.
        It is meant to be inherited by other classes.
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
        super().__init__(*args, **kwargs)
    
    #############################################################################################################################################

    @override
    def _download ( self,
                  ) -> None:

        """
            Method to download the model.
            In:
                * None.
            Out:
                * None.
        """

        # Model needs to be downloaded manually and put to the right place
        raise Exception(f"Cannnot download model {self.model_id} automatically. Please download it manually and put it in directory {self.model_path}.")

    #############################################################################################################################################
    #############################################################################################################################################

#####################################################################################################################################################
#####################################################################################################################################################