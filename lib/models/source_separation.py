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
    This module contains the class to perform source separation.
    Models should be loaded using the "get_model" function from "lib.models.loader".
"""

#####################################################################################################################################################
################################################################### PREPARE STUFF ###################################################################
#####################################################################################################################################################

# External imports
import os
from typing import *
import torch
import shutil
import demucs.separate
from demucs.hdemucs import HDemucs

# Project imports
import lib.models.base
import lib.audio

#####################################################################################################################################################
###################################################################### CLASSES ######################################################################
#####################################################################################################################################################

class Demucs (lib.models.base.ManualDlModel):

    """
        Class to perform source separation using the Demucs model.
    """

    #############################################################################################################################################

    def __init__ ( self,
                   version:  str = "mdx_extra",
                   *args:    Optional[list[any]],
                   **kwargs: Optional[dict[any, any]]
                 ) ->        None:

        """
            Constructor for the class.
            In:
                * version: The version of the model to use (see https://github.com/adefossez/demucs for a list).
                * args:    Extra arguments.
                * kwargs:  Extra keyword arguments.
            Out:
                * A new instance of the class.
        """

        # Inherit from parent class
        super().__init__(model_id=f"demucs/{version}", *args, **kwargs)

        # Attributes
        self.version = version
        self.model = None

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

        # Needed fix
        torch.serialization.add_safe_globals([HDemucs])

        # Find the .th file in the directory
        self.model = [f[:-3] for f in os.listdir(self.model_path) if f.endswith(".th")][0]
        if "-" in self.model:
            self.model = self.model.split("-")[0]

    #############################################################################################################################################

    @override
    def _apply ( self,
                 source_file_path: str,
                 target_file_path: str,
                 instrument:       str = "vocals",
               ) ->                None:

        """
            Method to apply the model to some data.
            In:
                * source_file_path: The path to the source file.
                * target_file_path: The path to the target file.
                * instrument:       The instrument to extract.
            Out:
                * None.
        """

        # Use the model to separate the audio
        demucs.separate.main(["--two-stems", instrument, "--name", self.model, "--repo", self.model_path, "-o", "separated", source_file_path])

        # Reorganize the output
        source_file_name = source_file_path.split(os.path.sep)[-1]
        source_file_name = source_file_name[:source_file_name.rfind(".")]
        os.rename(os.path.join("separated", self.model, source_file_name, f"{instrument}.wav"), target_file_path)
        shutil.rmtree("separated", ignore_errors=True)

    #############################################################################################################################################
    #############################################################################################################################################

#####################################################################################################################################################
#####################################################################################################################################################