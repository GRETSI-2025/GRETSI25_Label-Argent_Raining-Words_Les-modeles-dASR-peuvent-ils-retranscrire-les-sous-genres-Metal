#####################################################################################################################################################
################################################################### PREPARE STUFF ###################################################################
#####################################################################################################################################################

# Project imports
from arguments import args
from lib_audio import download_audio, extract_singer

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
for source in dataset_urls:
    for url, artist, title in dataset_urls[source]:

        # Download audio to "songs" directory
        file_name = f"{artist} - {title}"
        download_audio(url, source, file_name, force_dl=args().force_download_audio)

        # Extract audio to "demucs" directory
        #extract_singer(source, file_name, force_extract=args().force_download_audio)

#####################################################################################################################################################
#####################################################################################################################################################