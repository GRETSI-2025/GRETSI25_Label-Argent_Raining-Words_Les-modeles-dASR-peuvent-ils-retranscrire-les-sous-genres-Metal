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
    "brutal_death_slam": [
        ("https://www.youtube.com/watch?v=uWKLN6boesU", "Abominable Putridity", "Remnants of the Tortured"),
        ("https://www.youtube.com/watch?v=TNrsvLbYU3U", "Aborted", "The Saw and the Carnage Done"),
        ("https://www.youtube.com/watch?v=8fnjlDEHc6w", "Benighted", "Slut"),
        ("https://www.youtube.com/watch?v=xXxsudXDfxI", "Cerebral Bore", "The Bald Cadaver"),
        ("https://www.youtube.com/watch?v=wQx-yQJT1h8", "Cryptopsy", "Slit Your Guts"),
        ("https://www.youtube.com/watch?v=OYFf73g7Iy0", "Cytotoxin", "Radiophobia"),
        ("https://www.youtube.com/watch?v=7Oz5jFULJK4", "Defeated Sanity", "Into the Soil"),
        ("https://www.youtube.com/watch?v=2RqYS65yD2U", "Devourment", "Postmortal Coprophagia"),
        ("https://www.youtube.com/watch?v=bQd8CisPRU8", "Disfiguring the Goddess", "Admiration of Anger"),
        ("https://www.youtube.com/watch?v=Bn8rLOFpDxE", "Disgorge", "Atonement"),
        ("https://www.youtube.com/watch?v=K3rDRsEMay0", "Dying Fetus", "Grotesque Impalement"),
        ("https://www.youtube.com/watch?v=AvSTYZ0P-rY", "Extermination Dismemberment", "Survival"),
        ("https://www.youtube.com/watch?v=Ix3XzVDgVAM", "Guttural Secrete", "Clotting the Vacant Stare"),
        ("https://www.youtube.com/watch?v=m0_XX8my2pg", "Infecting the Swarm", "Aberrated Antibiosis"),
        ("https://www.youtube.com/watch?v=DeKlh2WYIAU", "Internal Suffering", "Haunters of the Dark"),
        ("https://www.youtube.com/watch?v=xNQR8GZUPRM", "Malignancy", "Biological Absurdity"),
        ("https://www.youtube.com/watch?v=b9RJXWxth5g", "Necrophagist", "Stabwound"),
        ("https://www.youtube.com/watch?v=RVji2A3wW9w", "Nile", "Cast Down the Heretic"),
        ("https://www.youtube.com/watch?v=P2_ZDRx62BI", "Pathology", "Earth's Downfall"),
        ("https://www.youtube.com/watch?v=AKRf2dbyJko", "Prostitute Disfigurement", "Freaking on the Mutilated"),
        ("https://www.youtube.com/watch?v=Jz6CzAQM8uk", "Putrid Pile", "Blood Runs Red"),
        ("https://www.youtube.com/watch?v=964aYTPlXTY", "Skinless", "Affirmation of Hatred"),
        ("https://www.youtube.com/watch?v=laYMT1wD18o", "Sublime Cadaveric Decomposition", "Hole of Oblivion"),
        ("https://www.youtube.com/watch?v=JHNhxGlrC1g", "Suffocation", "Surgery of Impalement"),
        ("https://www.youtube.com/watch?v=piygoU5KSak", "Visceral Disgorge", "Sedated and Amputated")
    ],
    "black": [
        ("https://www.youtube.com/watch?v=pMByAIXmFCI", "Abigor", "I Face the Eternal Winter"),
        ("https://www.youtube.com/watch?v=EcWEd_OONk0", "Arcturus", "To Thou Who Dwellest in the Night"),
        ("https://www.youtube.com/watch?v=Ad1vITzPLJc", "Bathory", "Blood Fire Death"),
        ("https://www.youtube.com/watch?v=yPMAVfkSatY", "Blut Aus Nord", "Metaphor of the Moon"),
        ("https://www.youtube.com/watch?v=nBrYE93ib3c", "Darkened Nocturn Slaughtercult", "The Descent to the Last Circle"),
        ("https://www.youtube.com/watch?v=KphlVeJX6fE", "Darkthrone", "Transilvanian Hunger"),
        ("https://www.youtube.com/watch?v=1zG8DL4ic0A", "Dissection", "Where Dead Angels Lie"),
        ("https://www.youtube.com/watch?v=YgQRRI9goFg", "Emperor", "I Am the Black Wizards"),
        ("https://www.youtube.com/watch?v=GBaZz66ajb8", "Enslaved", "The Watcher"),
        ("https://www.youtube.com/watch?v=3cVWr2ABV_4", "Immortal", "Solarfall"),
        ("https://www.youtube.com/watch?v=sDNdtU_wc74", "Marduk", "Panzer Division Marduk"),
        ("https://www.youtube.com/watch?v=z8VIhIIq-kk", "Mayhem", "Freezing Moon"),
        ("https://www.youtube.com/watch?v=6jIpZhLYyVE", "Mgła", "With Hearts Toward None III"),
        ("https://www.youtube.com/watch?v=l87EztQdPb4", "Obsequiae", "The Palms of Sorrowed Kings"),
        ("https://www.youtube.com/watch?v=wicsUbVugog", "Regarde Les Hommes Tomber", "A New Order"),
        ("https://www.youtube.com/watch?v=aI1ItXX3QV0", "Sarcófago", "I.N.R.I."),
        ("https://www.youtube.com/watch?v=otKmqy89zZM", "Satyricon", "K.I.N.G."),
        ("https://www.youtube.com/watch?v=Ujj6jsBUjt0", "Shining", "Fields of Faceless"),
        ("https://www.youtube.com/watch?v=cnkqBwdf8xs", "Sigh", "At My Funeral"),
        ("https://www.youtube.com/watch?v=4x1_rxzMwxo", "Spectral Wound", "Frigid and Spellbound"),
        ("https://www.youtube.com/watch?v=HRtYjH7EiAc", "Stormkeep", "A Journey Through Storms"),
        ("https://www.youtube.com/watch?v=kk1PkZ3PTRc", "Summoning", "Flammifer"),
        ("https://www.youtube.com/watch?v=TSYh1mFVyUs", "Watain", "Malfeitor"),
        ("https://www.youtube.com/watch?v=gqubRzRq3xM", "Windir", "Journey to the End"),
        ("https://www.youtube.com/watch?v=iet1wf7tM90", "Wolves in the Throne Room", "Queen of the Borrowed Light")
    ],
    "death": [
        ("https://www.youtube.com/watch?v=oF5YDb6rQrc", "Asphyx", "Deathhammer"),
        ("https://www.youtube.com/watch?v=24_BYwfqV4A", "Autopsy", "Charred Remains"),
        ("https://www.youtube.com/watch?v=_znQvXC3uw4", "Behemoth", "Demigod"),
        ("https://www.youtube.com/watch?v=qAdvSli74nQ", "Blood Incantation", "Vitrification of Blood (Part 1)"),
        ("https://www.youtube.com/watch?v=_SE2ayZIM50", "Bloodbath", "Like Fire"),
        ("https://www.youtube.com/watch?v=jZmL94xsMLM", "Bolt Thrower", "When Cannons Fade"),
        ("https://www.youtube.com/watch?v=fA837n8HcTA", "Cannibal Corpse", "Evirescation Plague"),
        ("https://www.youtube.com/watch?v=8y11Bs8hm_Q", "Dead Congregation", "Teeth Into Red"),
        ("https://www.youtube.com/watch?v=wK4VPM-0ssU", "Death", "Flesh and the Power It Holds"),
        ("https://www.youtube.com/watch?v=qAn3PpolPj0", "Deathspell Omega", "Abscission"),
        ("https://www.youtube.com/watch?v=9i1s8DH4uEg", "Deicide", "Sacrificial Suicide"),
        ("https://www.youtube.com/watch?v=wQXJaE5icVk", "Demigod", "Slumber of Sullen Eyes"),
        ("https://www.youtube.com/watch?v=DIDjSfdVe9w", "Demilich", "Emptiness of Vanishing"),
        ("https://www.youtube.com/watch?v=U2efccXjnZs", "Disembowelment", "The Tree of Life and Death"),
        ("https://www.youtube.com/watch?v=s8qm5lFAnrg", "Disma", "Chaos Apparition"),
        ("https://www.youtube.com/watch?v=_7G1nqw8RcE", "Dismember", "And So Is Life"),
        ("https://www.youtube.com/watch?v=dHcaWwaNkLM", "Deam Unending", "In Cipher I Weep"),
        ("https://www.youtube.com/watch?v=FAVAu_s9ADw", "Épitaphe", "Melancholia"),
        ("https://www.youtube.com/watch?v=JwdrlRN_z6I", "Krypts", "Open the Crypt"),
        ("https://www.youtube.com/watch?v=D-yKIZyixIU", "Morbid Angel", "Immortal Rites"),
        ("https://www.youtube.com/watch?v=aF9M0vetBB0", "Mythic", "Winter Solstice"),
        ("https://www.youtube.com/watch?v=wzTC3J0xRUs", "Obituary", "Infected"),
        ("https://www.youtube.com/watch?v=j6ljyqDB15o", "Portal", "Abysmill"),
        ("https://www.youtube.com/watch?v=j1cTZIYQrs0", "Possessed", "Fallen Angel"),
        ("https://www.youtube.com/watch?v=ev4PS7Fk6j8", "Ulcerate", "Extinguished Light")
    ],
    "goregrind": [
        ("https://www.youtube.com/watch?v=l4-Ya_Xa3f8", "Carcass", "Reek of Putrefaction"),
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

# Create directories if they do not exist
for style in dataset_urls:
    directories = [os.path.join(script_args().datasets_path, "audio", "songs", style)] + [os.path.join(script_args().datasets_path, "audio", "_".join(model_name if type(model_name) in [list, tuple] else [model_name]).lower(), style) for model_name in script_args().source_separation_models]
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        os.chmod(directory, 0o777)

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