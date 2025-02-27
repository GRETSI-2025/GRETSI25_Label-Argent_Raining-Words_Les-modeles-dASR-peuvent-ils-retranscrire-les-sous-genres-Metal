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
    This script downloads the dataset.
    It downloads the songs and separates the vocals from the songs.
    The songs are stored in the "songs" directory.
    The vocals are stored in the "demucs_xxx" directory, where "xxx" is the name of the source separation model.
    The dataset is composed of songs from different styles of metal music.
    Songs are used here under the fair use policy for research purposes.
    Rights are reserved to the respective owners.
    The list of songs was curated by the authors.
"""

#####################################################################################################################################################
################################################################### PREPARE STUFF ###################################################################
#####################################################################################################################################################

# External imports
import os

# Project imports
from lib.arguments import script_args
import lib.audio

# List of files to get
dataset_urls = {
    "death": [
        ("https://www.youtube.com/watch?v=_SE2ayZIM50", "Bloodbath", "Like Fire"),
        ("https://www.youtube.com/watch?v=DIDjSfdVe9w", "Demilich", "Emptiness of Vanishing"),
        ("https://www.youtube.com/watch?v=j6ljyqDB15o", "Portal", "Abysmill")
    ],
    "black": [
        ("https://www.youtube.com/watch?v=KphlVeJX6fE", "Darkthrone", "Transilvanian Hunger"),
        ("https://www.youtube.com/watch?v=z8VIhIIq-kk", "Mayhem", "Freezing Moon")
    ],
    "powerviolence": [
        ("https://www.youtube.com/watch?v=sRGmazKRbB4", "Death Toll 80k", "No Escape"),
        ("https://www.youtube.com/watch?v=Z1A1PSre14M", "Insect Warfare", "Human Trafficking")
    ],
    "thrash": [
        ("https://www.youtube.com/watch?v=-BzWmtY7tVg", "Hexecutor", "Visitations of a Lascivious Entity"),
        ("https://www.youtube.com/watch?v=kzYMnGn5NSI", "Deathhammer", "Lead Us Into Hell")
    ]
}

#####################################################################################################################################################
######################################################################### GO ########################################################################
#####################################################################################################################################################

# Get all songs
for style in dataset_urls:
    for url, artist, title in dataset_urls[style]:

        # Download audio to "songs" directory
        song_file_path = os.path.join(script_args().datasets_path, "audio", "songs", style, f"{artist} - {title}.wav")
        lib.audio.download_audio(url, song_file_path)

        # Extract audio to "demucs" directory
        for model_name in script_args().source_separation_models:
            model_subdir = "_".join(model_name if type(model_name) in [list, tuple] else [model_name]).lower()
            vocals_file_path = os.path.join(script_args().datasets_path, "audio", model_subdir, style, f"{artist} - {title}.wav")
            lib.audio.extract_vocals(song_file_path, vocals_file_path)

#####################################################################################################################################################
#####################################################################################################################################################