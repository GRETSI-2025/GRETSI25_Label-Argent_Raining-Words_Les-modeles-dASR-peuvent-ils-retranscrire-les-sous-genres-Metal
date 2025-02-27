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
    This script runs all the other scripts.
    It is the main script of the project.
"""

#####################################################################################################################################################
################################################################### PREPARE STUFF ###################################################################
#####################################################################################################################################################

# External imports
import sys

#####################################################################################################################################################
##################################################################### FUNCTIONS #####################################################################
#####################################################################################################################################################

def print_title ( title: str,
                  size: str = 100,
                  character: str = "#",
                  targets: list[any] = [sys.stdout, sys.stderr]
                ) -> None:

    """
        Function to print a centered title.
        In:
            * title:      The title to print.
            * size:       The size of the title.
            * character:  The character to use to print the title.
            * targets:    The targets to print the title to.
        Out:
            * None.
    """

    # Print centered title on all targets
    for target in targets:
        print("", file=target, flush=True)
        print(character * size, file=target, flush=True)
        print(character + " " * ((size - len(title) - 1) // 2) + title + " " * ((size - len(title) - 2) // 2) + character, file=target, flush=True)
        print(character * size, file=target, flush=True)
        print("", file=target, flush=True)

#####################################################################################################################################################
######################################################################### GO ########################################################################
#####################################################################################################################################################

# Get dataset
print_title("Get dataset")
import script_1_get_dataset
print("Done")

# Extract lyrics
print_title("Extract lyrics")
import script_2_extract_lyrics
print("Done")

# Compute metrics
print_title("Compute metrics")
import script_3_compute_metrics
print("Done")

# Analyze EMVD
print_title("Analyze EMVD")
import script_4_analyze_emvd
print("Done")

# Analyze songs
print_title("Analyze songs")
import script_5_analyze_songs
print("Done")

#####################################################################################################################################################
#####################################################################################################################################################