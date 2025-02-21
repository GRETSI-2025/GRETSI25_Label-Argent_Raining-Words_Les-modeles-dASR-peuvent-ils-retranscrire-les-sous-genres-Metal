#####################################################################################################################################################
################################################################### PREPARE STUFF ###################################################################
#####################################################################################################################################################

# Project imports
from arguments import args
import lib_audio

# List of files to get
dataset_urls = {
    "death": [
        ("https://www.youtube.com/watch?v=_SE2ayZIM50", "Bloodbath", "Like Fire"),
        ("https://www.youtube.com/watch?v=DIDjSfdVe9w", "Demilich", "Emptiness of Vanishing")
    ],
    "black": [
        ("https://www.youtube.com/watch?v=KphlVeJX6fE", "Darkthrone", "Transilvanian Hunger"),
        ("https://www.youtube.com/watch?v=z8VIhIIq-kk", "Mayhem", "Freezing Moon")
    ],
    "powerviolence": [
        ("https://www.youtube.com/watch?v=sRGmazKRbB4", "Death Toll 80k", "No Escape"),
        ("https://www.youtube.com/watch?v=Z1A1PSre14M", "Insect Warfare", "Human Trafficking")
    ]
}

#####################################################################################################################################################
######################################################################### GO ########################################################################
#####################################################################################################################################################

# Get all songs
for style in dataset_urls:
    for url, artist, title in dataset_urls[style]:

        # Download audio to "songs" directory
        file_name = f"{artist} - {title}"
        lib_audio.download_audio(url, style, file_name, force_dl=args().force_download_audio)

        # Extract audio to "demucs" directory
        lib_audio.extract_vocals(style, file_name, force_extract=args().force_download_audio)

#####################################################################################################################################################
#####################################################################################################################################################