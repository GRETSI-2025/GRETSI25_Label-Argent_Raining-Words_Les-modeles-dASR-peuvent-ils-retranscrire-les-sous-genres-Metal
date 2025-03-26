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
        ("https://www.youtube.com/watch?v=WFCtZkTQINw", "Obtained Enslavement", "Veils of Wintersorrow"),
        ("https://www.youtube.com/watch?v=wicsUbVugog", "Regarde Les Hommes Tomber", "A New Order"),
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
        ("https://www.youtube.com/watch?v=fA837n8HcTA", "Cannibal Corpse", "Evisceration Plague"),
        ("https://www.youtube.com/watch?v=8y11Bs8hm_Q", "Dead Congregation", "Teeth Into Red"),
        ("https://www.youtube.com/watch?v=wK4VPM-0ssU", "Death", "Flesh and the Power It Holds"),
        ("https://www.youtube.com/watch?v=qAn3PpolPj0", "Deathspell Omega", "Abscission"),
        ("https://www.youtube.com/watch?v=9i1s8DH4uEg", "Deicide", "Sacrificial Suicide"),
        ("https://www.youtube.com/watch?v=wQXJaE5icVk", "Demigod", "Slumber of Sullen Eyes"),
        ("https://www.youtube.com/watch?v=DIDjSfdVe9w", "Demilich", "Emptiness of Vanishing"),
        ("https://www.youtube.com/watch?v=U2efccXjnZs", "Disembowelment", "The Tree of Life and Death"),
        ("https://www.youtube.com/watch?v=s8qm5lFAnrg", "Disma", "Chaos Apparition"),
        ("https://www.youtube.com/watch?v=_7G1nqw8RcE", "Dismember", "And So Is Life"),
        ("https://www.youtube.com/watch?v=dHcaWwaNkLM", "Dream Unending", "In Cipher I Weep"),
        ("https://www.youtube.com/watch?v=FAVAu_s9ADw", "Épitaphe", "Melancholia"),
        ("https://www.youtube.com/watch?v=JwdrlRN_z6I", "Krypts", "Open the Crypt"),
        ("https://www.youtube.com/watch?v=D-yKIZyixIU", "Morbid Angel", "Immortal Rites"),
        ("https://www.youtube.com/watch?v=aF9M0vetBB0", "Mythic", "Winter Solstice"),
        ("https://www.youtube.com/watch?v=wzTC3J0xRUs", "Obituary", "Infected"),
        ("https://www.youtube.com/watch?v=j6ljyqDB15o", "Portal", "Abysmill"),
        ("https://www.youtube.com/watch?v=j1cTZIYQrs0", "Possessed", "Fallen Angel"),
        ("https://www.youtube.com/watch?v=ev4PS7Fk6j8", "Ulcerate", "Extinguished Light")
    ],
    "deathcore_metalcore": [
        ("https://www.youtube.com/watch?v=S1jpivGF2xk", "Acrania", "Disillusion in a Discordant System"),
        ("https://www.youtube.com/watch?v=RIiaP4zsH6c", "AngelMaker", "What I Would Give"),
        ("https://www.youtube.com/watch?v=-nl_ttX031Y", "August Burns Red", "Composure"),
        ("https://www.youtube.com/watch?v=DWGIgDwbdzo", "All Shall Perish", "Eradication"),
        ("https://www.youtube.com/watch?v=V5zkEz31_Jc", "Born of Osiris", "Abstract Art"),
        ("https://www.youtube.com/watch?v=AWggPLXeOkU", "Bring Me the Horizon", "Pray for Plagues"),
        ("https://www.youtube.com/watch?v=7ca6B2uMCoc", "Chelsea Grin", "Recreant"),
        ("https://www.youtube.com/watch?v=kOtgze9J708", "Carnifex", "Die Without Hope"),
        ("https://www.youtube.com/watch?v=mUaAjEB3acg", "Despised Icon", "Furtive Monologue"),
        ("https://www.youtube.com/watch?v=TTkZt8r2lko", "Heaven Shall Burn", "Endzeit"),
        ("https://www.youtube.com/watch?v=f9nMq5cYSxY", "I Declare War", "Now You're Going to Be Famous"),
        ("https://www.youtube.com/watch?v=PdeB3Q8K9qk", "Infant Annihilator", "Decapitation Fornication"),
        ("https://www.youtube.com/watch?v=JdONCIPaipE", "Ingested", "Altar of Flesh"),
        ("https://www.youtube.com/watch?v=fWbV-4LEHBU", "Job for a Cowboy", "Entombment of a Machine"),
        ("https://www.youtube.com/watch?v=_QcTWxZu2f8", "Lorna Shore", "To the Hellfire"),
        ("https://www.youtube.com/watch?v=BKe4Z9mmyVk", "Make Them Suffer", "Elegies"),
        ("https://www.youtube.com/watch?v=DCcYj_8oI0E", "Oceano", "District of Misery"),
        ("https://www.youtube.com/watch?v=dNvOVey0MSs", "Parkway Drive", "Carrion"),
        ("https://www.youtube.com/watch?v=UyI6S9uYL3A", "Shadow of Intent", "Farewell"),
        ("https://www.youtube.com/watch?v=Z-t-ug0cg-Q", "Slaughter To Prevail", "Cultural Ills"),
        ("https://www.youtube.com/watch?v=Zw7KfVMp6H0", "Suffokate", "Not the Fallen"),
        ("https://www.youtube.com/watch?v=VT13aP_GnuM", "Suicide Silence", "Unanswered"),
        ("https://www.youtube.com/watch?v=47Plg93oJ1M", "Thy Art Is Murder", "Reign of Darkness"),
        ("https://www.youtube.com/watch?v=eVI6c0TlM2g", "Whitechapel", "The Saw Is the Law"),
        ("https://www.youtube.com/watch?v=n2JX9VeL7n4", "Winds of Plague", "The Impaler")
    ],
    "doom": [
        ("https://www.youtube.com/watch?v=fC0OFbhLeqk", "Agalloch", "Limbs"),
        ("https://www.youtube.com/watch?v=Ea16YmkcnO0", "Antimatter", "Fold"),
        ("https://www.youtube.com/watch?v=D5Du4BhqE14", "Black Sabbath", "Behind the Wall of Sleep"),
        ("https://www.youtube.com/watch?v=5IDNY2mvnDc", "Candlemass", "At the Gallows End"),
        ("https://www.youtube.com/watch?v=_ju5k57YuEc", "Cathedral", "Ride"),
        ("https://www.youtube.com/watch?v=AvIZN71kt7Y", "Dark Quarterer", "Dark Quarterer"),
        ("https://www.youtube.com/watch?v=tcO1u6f3Ops", "Death SS", "Black And Violet"),
        ("https://www.youtube.com/watch?v=eaqVQ2gvxYE", "Draconian", "Sorrow of Sophia"),
        ("https://www.youtube.com/watch?v=ZdlEIlq9nZg", "Electric Wizard", "Funeralopolis"),
        ("https://www.youtube.com/watch?v=S1jpzNoko78", "My Dying Bride", "Your Broken Shore"),
        ("https://www.youtube.com/watch?v=0WNreJ7yQCU", "Pagan Altar", "The Sorcerer"),
        ("https://www.youtube.com/watch?v=LTTt-ikVJSk", "Pallbearer", "The Ghost I Used To Be"),
        ("https://www.youtube.com/watch?v=OfJJWia2B4o", "Paradise Lost", "The Longest Winter"),
        ("https://www.youtube.com/watch?v=wRIkfMSnED4", "Pentagram", "The Ghoul"),
        ("https://www.youtube.com/watch?v=prCu08DySn8", "Quicksand Dream", "A Child Was Born"),
        ("https://www.youtube.com/watch?v=PbTrz56nows", "Saint Vitus", "The Walking Dead"),
        ("https://www.youtube.com/watch?v=Wc8sOqIWwvk", "Scald", "Night Sky"),
        ("https://www.youtube.com/watch?v=_OWwgGYhsBs", "Sleep", "Dragonaut"),
        ("https://www.youtube.com/watch?v=in1TBIekq00", "Solitude Aeternus", "Mirror of Sorrow"),
        ("https://www.youtube.com/watch?v=nbbZrBJceqw", "Sorcerer", "Lamenting of the Innocent"),
        ("https://www.youtube.com/watch?v=0l3-OrBJwrI", "Triptykon", "Aurorae"),
        ("https://www.youtube.com/watch?v=A9SzZlElSoE", "Warning", "Watching from a Distance"),
        ("https://www.youtube.com/watch?v=nxCeC3Mxqxw", "While Heaven Wept", "Icarus And I"),
        ("https://www.youtube.com/watch?v=qWcjYA17FkQ", "Windhand", "Woodbine"),
        ("https://www.youtube.com/watch?v=oxliU58k87Q", "Witchfinder General", "Burning a Sinner")
    ],
    "dsbm": [
        ("https://www.youtube.com/watch?v=gAzvUZrrsJc", "Alene Misantropi", "Golden Blood Sea"),
        ("https://www.youtube.com/watch?v=mPjbH5kRc8Q", "Anti", "Death Into Life"),
        ("https://www.youtube.com/watch?v=k016kdJT8ew", "Austere", "Sullen"),
        ("https://www.youtube.com/watch?v=RBm1LsOMBxU", "Be Persecuted", "Be Resented for Livelihood"),
        ("https://www.youtube.com/watch?v=1rAFun2LQ5g", "ColdWorld", "Suicide"),
        ("https://www.youtube.com/watch?v=4bSMZdaDXrE", "Forgive Me", "Funerals Of Birth"),
        ("https://www.youtube.com/watch?v=SIxt7FGbF0Q", "Forgotten Thought", "Black Ink Soaked Page"),
        ("https://www.youtube.com/watch?v=FN5aQsVbAhk", "Forgotten Tomb", "Disheartenment"),
        ("https://www.youtube.com/watch?v=02d5lTEU1_c", "Happy Days", "Don't Go"),
        ("https://www.youtube.com/watch?v=LQiMT23QqXM", "Leviathan", "The Idiot Sun"),
        ("https://www.youtube.com/watch?v=HosL3hnADfs", "Lifelover", "Lethargy"),
        ("https://www.youtube.com/watch?v=r0etjUb-VpY", "Lost Inside", "I Hate Myself"),
        ("https://www.youtube.com/watch?v=m-IAORB6jQM", "Lyrinx", "Another Life Ready to End"),
        ("https://www.youtube.com/watch?v=Nqfp9rUUPkM", "Make a Change... Kill Yourself", "VII"),
        ("https://www.youtube.com/watch?v=lrlQi3Qqm3Q", "Misere Nobis", "Sopor Aeternus"),
        ("https://www.youtube.com/watch?v=KGsboZbYxZ8", "Mortualia", "Emptiness of All"),
        ("https://www.youtube.com/watch?v=Ro8-wEhEqDQ", "Nocturnal Depression", "Dead Children"),
        ("https://www.youtube.com/watch?v=pCI4-CXEXzk", "Nyktalgia", "Cold Void"),
        ("https://www.youtube.com/watch?v=H6E678QXSeE", "Psychonaut 4", "Overdose Was the Best Way to Die"),
        ("https://www.youtube.com/watch?v=if2EN0Zyu84", "Silencer", "Sterile Nails and Thunderbowels"),
        ("https://www.youtube.com/watch?v=oVzx0Tvd2ZM", "Sorry...", "Everything Is Falling Apart"),
        ("https://www.youtube.com/watch?v=RjU6qSfehK4", "Suicide Emotions", "A New Dawn"),
        ("https://www.youtube.com/watch?v=YS1RuumZVk4", "Thy Light", "The Bridge"),
        ("https://www.youtube.com/watch?v=wtT-1M4WfeU", "Veil", "Mater Maternis"),
        ("https://www.youtube.com/watch?v=BIkNLwmpMV0", "Xasthur", "Screaming at Forgotten Fears")
    ],
    "goregrind_porngrind": [
        ("https://www.youtube.com/watch?v=zxgL4ZPPoPY", "Agathocles", "Splattered Brains"),
        ("https://www.youtube.com/watch?v=bw8xvJiqux4", "Blasted Pancreas", "Lymphangiosarcoma"),
        ("https://www.youtube.com/watch?v=WP2TWpv5NxY", "Butcher ABC", "Maximum Rotting Corpse"),
        ("https://www.youtube.com/watch?v=l4-Ya_Xa3f8", "Carcass", "Reek of Putrefaction"),
        ("https://www.youtube.com/watch?v=rnSHLCnQsdI", "Cliteater", "Slimming Party at Kelly's"),
        ("https://www.youtube.com/watch?v=8yG1pXg5Ac4", "Cock and Ball Torture", "Anal Sex Terror"),
        ("https://www.youtube.com/watch?v=spzaVx33Xvs", "Crash Syndrom", "Paragraph of Expertise Conundrum"),
        ("https://www.youtube.com/watch?v=eVHVuz3bCTY", "Dead", "Delicious Taste Of Vaginal Excrements"),
        ("https://www.youtube.com/watch?v=QOV0khHFz-c", "Dead Infection", "Maggots In Your Flesh"),
        ("https://www.youtube.com/watch?v=kMCb-nbfam8", "Disgorge", "Faecalized"),
        ("https://www.youtube.com/watch?v=LzJcZY5p_Qg", "Exhumed", "Open the Abscess"),
        ("https://www.youtube.com/watch?v=cs9fVfptIVU", "General Surgery", "Arterial Spray Obsession"),
        ("https://www.youtube.com/watch?v=mkJMjAhltUE", "Gore Beyond Necropsy", "Reek of Putridfashionpig"),
        ("https://www.youtube.com/watch?v=q7RnRVOZ9u0", "Haemmorhage", "Mortuary Riot"),
        ("https://www.youtube.com/watch?v=0bCXy2zH-hk", "Impetigo", "Cannibale Ballet"),
        ("https://www.youtube.com/watch?v=-HDq-1khxaM", "Intense Hammer Rage", "Eeeuw What’s That Goo"),
        ("https://www.youtube.com/watch?v=HCg6MHBnQkg", "Lord Gore", "Resickened"),
        ("https://www.youtube.com/watch?v=JOW2YggMLG0", "Lymphatic Phlegm", "Chronic Pachymeningitis with Diffuse Dura Mater Inspissation"),
        ("https://www.youtube.com/watch?v=hkhvX6AxfF4", "Offal", "Putr-Essence"),
        ("https://www.youtube.com/watch?v=GrKQiBpUwG8", "Pathologist", "Infectious Agonizing Parasitism"),
        ("https://www.youtube.com/watch?v=dtzO25vgraU", "Pharmacist", "Nursery Aesthetics"),
        ("https://www.youtube.com/watch?v=JQZtcCj8LvI", "Regurgitate", "Carnivorous Erection"),
        ("https://www.youtube.com/watch?v=TtMrNod44_8", "Septage", "Intolerant Spree of Infesting Forms (Septic Worship)"),
        ("https://www.youtube.com/watch?v=uLLQyQym8m0", "The County Medical Examiners", "Casper's Dictum"),
        ("https://www.youtube.com/watch?v=Xy39_E68EmY", "Torsofuck", "Retarded Anal Whore")
    ],
    "symphonic_gothic": [
        ("https://www.youtube.com/watch?v=EpNOjlPraSs", "After Forever", "Leaden Legacy (The Embrace That Smothers - Part I)"),
        ("https://www.youtube.com/watch?v=jbigMRL9VMI", "Amberian Dawn", "Valkyries"),
        ("https://www.youtube.com/watch?v=TEYCWvuYq6Y", "Angellore", "Blood for Lavinia"),
        ("https://www.youtube.com/watch?v=jlZSgH2583Q", "Angtoria", "Do You See Me Now?"),
        ("https://www.youtube.com/watch?v=VUaUMUT1h4o", "Beyond the Black", "Hallelujah"),
        ("https://www.youtube.com/watch?v=bYznBldN7OY", "Delain", "We Are the Others"),
        ("https://www.youtube.com/watch?v=rtjGjIgnltY", "Elis", "Perfect Love"),
        ("https://www.youtube.com/watch?v=3q0w5dJt9Hc", "Epica", "Sensorium"),
        ("https://www.youtube.com/watch?v=BA9qVlrkQKk", "Edenbridge", "Cheyenne Spirit"),
        ("https://www.youtube.com/watch?v=3gliWhtxbyE", "Ex Libris", "Love Is Thy Sin"),
        ("https://www.youtube.com/watch?v=tjwfI0fZHJA", "Exit Eden", "Hold Back Your Fear"),
        ("https://www.youtube.com/watch?v=mfNXdBT0HW0", "Imperia", "Fata Morgana"),
        ("https://www.youtube.com/watch?v=35eio8imriw", "Kamelot", "Forever"),
        ("https://www.youtube.com/watch?v=pzK44v5PWMg", "Lacuna Coil", "Falling Again"),
        ("https://www.youtube.com/watch?v=E84gPHP8g5Q", "Leaves' Eyes", "Through our Veins"),
        ("https://www.youtube.com/watch?v=Nj66U769jHY", "Nemesea", "Lucifer"),
        ("https://www.youtube.com/watch?v=n1G5WiMoRjw", "Nightwish", "The Phantom of the Opera"),
        ("https://www.youtube.com/watch?v=nDbw6MEOpmI", "Pythia", "Ancient Soul"),
        ("https://www.youtube.com/watch?v=mP0IWOW3kO4", "ReVamp", "Sweet Curse"),
        ("https://www.youtube.com/watch?v=Otk_X_LckJw", "Sirenia", "My Mind's Eye"),
        ("https://www.youtube.com/watch?v=CwUxkzk0Slg", "Theatre of Tragedy", "Bacchante"),
        ("https://www.youtube.com/watch?v=jTdotKrKSV8", "Tristania", "My Lost Lenore"),
        ("https://www.youtube.com/watch?v=ulQIXCSCJeo", "UnSun", "A Single Touch"),
        ("https://www.youtube.com/watch?v=_yKiqBO9ZaM", "Within Temptation", "Ice Queen"),
        ("https://www.youtube.com/watch?v=kkjunJnObho", "Xandria", "Ravenheart")
    ],
    "punk_hardcore": [
        ("https://www.youtube.com/watch?v=_zvJ39bhI0Y", "Adolescents", "No Way"),
        ("https://www.youtube.com/watch?v=eMZD8g_3Bkk", "Agnostic Front", "Time Will Come"),
        ("https://www.youtube.com/watch?v=grixlGSNS6U", "Bad Brains", "Banned in D.C."),
        ("https://www.youtube.com/watch?v=WFMIxyb-8i4", "Anti Cimex", "Victims Of A Bombraid"),
        ("https://www.youtube.com/watch?v=ga6v91ZGL1U", "Black Flag", "My War"),
        ("https://www.youtube.com/watch?v=GPF_SlIQJos", "Chaos UK", "Selfish Few"),
        ("https://www.youtube.com/watch?v=ACZGgdNCFyU", "Comeback Kid", "Wake the Dead"),
        ("https://www.youtube.com/watch?v=649E9hYnynU", "Cro-Mags", "Street Justice"),
        ("https://www.youtube.com/watch?v=h49yLu6kkVs", "Discharge", "Hear Nothing See Nothing Say Nothing"),
        ("https://www.youtube.com/watch?v=farFbNMNJm8", "D.R.I.", "I Don't Need Society"),
        ("https://www.youtube.com/watch?v=cJHymIHSaeU", "G.B.H", "Give Me Fire"),
        ("https://www.youtube.com/watch?v=X_FiozAcDvU", "GG Allin", "Die When You Die"),
        ("https://www.youtube.com/watch?v=RBRnL9qQyXk", "Guilt Trip", "Broken Wings"),
        ("https://www.youtube.com/watch?v=WpNJ1GcDYD8", "Hatebreed", "I Will Be Heard"),
        ("https://www.youtube.com/watch?v=Na8t7flTOPI", "Incendiary", "Still Burning"),
        ("https://www.youtube.com/watch?v=3pr9TOf-RpU", "Inside Out", "Burning Fight"),
        ("https://www.youtube.com/watch?v=X3tg5PwzPok", "Madball", "Lockdown"),
        ("https://www.youtube.com/watch?v=smM_bphb6pU", "Minor Threat", "Minor Threat"),
        ("https://www.youtube.com/watch?v=yxZZ4A1Fshk", "Misfits", "She"),
        ("https://www.youtube.com/watch?v=Qdcfxk-e_iw", "Poison Idea", "Just to Get Away"),
        ("https://www.youtube.com/watch?v=oTclTVh2lxE", "Sick Of It All", "Just Lies"),
        ("https://www.youtube.com/watch?v=OCRT3CRNyNY", "Sorcerer", "Badlands"),
        ("https://www.youtube.com/watch?v=cVlddw6e8vI", "The Exploited", "Beat the Bastards"),
        ("https://www.youtube.com/watch?v=uwYg1hOGRQQ", "Turnstile", "Mystery"),
        ("https://www.youtube.com/watch?v=la_kXIiM8lQ", "Youth of Today", "Break Down the Walls")
    ],
    "heavy_power": [
        ("https://www.youtube.com/watch?v=Y1e1JOFcrNQ", "Atlantean Kodex", "Heresiarch (Thousand-Faced Moon)"),
        ("https://www.youtube.com/watch?v=edPckSU41g4", "Cirith Ungol", "King of the Dead"),
        ("https://www.youtube.com/watch?v=EaBcKfc3cH8", "Crimson Glory", "Queen of the Masquerade"),
        ("https://www.youtube.com/watch?v=gilVBNSRt6o", "Dokken", "In My Dreams"),
        ("https://www.youtube.com/watch?v=VmDgdDXH944", "Fates Warning", "Guardian"),
        ("https://www.youtube.com/watch?v=bGZ86q9Ox6k", "Fifth Angel", "In the Fallout"),
        ("https://www.youtube.com/watch?v=uEB48yMiQho", "Grim Reaper", "See You In Hell"),
        ("https://www.youtube.com/watch?v=0xmRgpXRC5A", "Heathen's Rage", "Knights of Steel"),
        ("https://www.youtube.com/watch?v=IB8ApQZxELM", "Heir Apparent", "Just Imagine"),
        ("https://www.youtube.com/watch?v=u7ozEjfhcN0", "Helloween", "Future World"),
        ("https://www.youtube.com/watch?v=Q3sJlr0kw8I", "Iron Maiden", "Run to the Hills"),
        ("https://www.youtube.com/watch?v=8Ksksu9JDkg", "Judas Priest", "The Sentinel"),
        ("https://www.youtube.com/watch?v=HrImHhQ02zA", "Manilla Road", "Flaming Metal Systems"),
        ("https://www.youtube.com/watch?v=T6zOT3IZ90U", "Manowar", "Hail and Kill"),
        ("https://www.youtube.com/watch?v=7vIEAnd7B8Q", "Medieval Steel", "Echoes"),
        ("https://www.youtube.com/watch?v=nItYGeORvm0", "Omen", "The Axeman"),
        ("https://www.youtube.com/watch?v=P5YxNUtvosc", "Praying Mantis", "The Messiah"),
        ("https://www.youtube.com/watch?v=vvgLj8pawGI", "Queensrÿche", "Eyes of a Stranger"),
        ("https://www.youtube.com/watch?v=nZUysqemMmg", "Riot", "Thundersteel"),
        ("https://www.youtube.com/watch?v=hF0WOfoJtH8", "Riot City", "In the Dark"),
        ("https://www.youtube.com/watch?v=LAN8v-sw0Tc", "Satan", "Break Free"),
        ("https://www.youtube.com/watch?v=wLXsfXXEEgY", "Seven Sisters", "Shadow of a Fallen Star"),
        ("https://www.youtube.com/watch?v=uBaIPGnHxEk", "Skid Row", "Youth Gone Wild"),
        ("https://www.youtube.com/watch?v=C4R9bdDmjHY", "Warlord", "Glory"),
        ("https://www.youtube.com/watch?v=xjZk58SNo20", "Winterhawk", "Period of Change")
    ],
    "melodic_death": [
        ("https://www.youtube.com/watch?v=weJBR8bEafA", "A Canorous Quintet", "Silence of the World Beyond"),
        ("https://www.youtube.com/watch?v=3Or87hx0R7w", "Amon Amarth", "Valhall Awaits Me"),
        ("https://www.youtube.com/watch?v=09Bbzq8CwRw", "Arch Enemy", "We Will Rise"),
        ("https://www.youtube.com/watch?v=nbz_1R02vPc", "Archons", "Plague of Corruption"),
        ("https://www.youtube.com/watch?v=pnHoNRADFOo", "At the Gates", "Slaughter of the Soul"),
        ("https://www.youtube.com/watch?v=zCkH5_7OBoU", "Be'lakor", "Countless Skies"),
        ("https://www.youtube.com/watch?v=8eeZu1gfAtI", "Ceremonial Oath", "Dreamsong"),
        ("https://www.youtube.com/watch?v=u4lApDmOGRg", "Children of Bodom", "Everytime I Die"),
        ("https://www.youtube.com/watch?v=ZRGQ5izfH4s", "Dark Tranquillity", "Shivers and Voids"),
        ("https://www.youtube.com/watch?v=pwnb1fqW9-o", "Eucharist", "Greeting Immortality"),
        ("https://www.youtube.com/watch?v=DZxpk8pAjtw", "Gates of Ishtar", "Perpetual Dawn (The Arrival of Eternity - End My Pain)"),
        ("https://www.youtube.com/watch?v=gz7kjQ9sfSk", "Hypocrisy", "Fractured Millennium"),
        ("https://www.youtube.com/watch?v=OQ98nYu4zJQ", "In Flames", "Insipid 2000"),
        ("https://www.youtube.com/watch?v=zGx1e4jGVco", "Inferi", "The Promethean Kings"),
        ("https://www.youtube.com/watch?v=EdLuXj25GF8", "Insomnium", "Mortal Share"),
        ("https://www.youtube.com/watch?v=xSYAhLYDa2c", "Kalmah", "They Will Return"),
        ("https://www.youtube.com/watch?v=lDcbCIzwq_I", "Kataklysm", "As I Slither"),
        ("https://www.youtube.com/watch?v=ncRZAKHDrqY", "Norther", "Betrayed"),
        ("https://www.youtube.com/watch?v=ep_9S9WH8TI", "Quo Vadis", "Silence Calls the Storm"),
        ("https://www.youtube.com/watch?v=KaCLjxss2_8", "Sacrilege", "Feed the Cold"),
        ("https://www.youtube.com/watch?v=u7bn3Dc4TEs", "The Agonist", "...And Their Eulogies Sang Me To Sleep"),
        ("https://www.youtube.com/watch?v=mruugwHX1YQ", "The Haunted", "Hollow Ground"),
        ("https://www.youtube.com/watch?v=jwbj2gzzTpk", "The Black Dahlia Murder", "What a Horrible Night to Have a Curse"),
        ("https://www.youtube.com/watch?v=1eJZGTedU6o", "Wintersun", "Beyond the Dark Sun"),
        ("https://www.youtube.com/watch?v=gdqMlknKYV8", "Wolfheart", "Ghost of Karelia")
    ],
    "grindcore_deathgrind_powerviolence": [
        ("https://www.youtube.com/watch?v=yyy65SEdj_0", "Agoraphobic Nosebleed", "Repercussions in the Life of an Opportunistic, Pseudo Intellectual Jackass"),
        ("https://www.youtube.com/watch?v=OJqAOV2Y198", "Anal Cunt", "Crankin’ My Bands Demo on a Box at the Beach"),
        ("https://www.youtube.com/watch?v=3s1n7eYQMr8", "Antigama", "Underterminate"),
        ("https://www.youtube.com/watch?v=v6T3iu5R-5A", "Assück", "Civilization Comes, Civilization Goes"),
        ("https://www.youtube.com/watch?v=BdIBNIfbzbY", "Brutal Truth", "Regression-Progression"),
        ("https://www.youtube.com/watch?v=aaRymbiyL4Q", "Contrastic", "Sex with Four Walls"),
        ("https://www.youtube.com/watch?v=sRGmazKRbB4", "Death Toll 80k", "No Escape"),
        ("https://www.youtube.com/watch?v=e16M1-4383M", "Discordance Axis", "The End of Rebirth"),
        ("https://www.youtube.com/watch?v=cQ1VAZ1ZBMk", "Extreme Noise Terror", "Being and Nothing"),
        ("https://www.youtube.com/watch?v=I-idpoyOOsk", "Feastem", "Sick"),
        ("https://www.youtube.com/watch?v=2RV2JlOah5Q", "Gravesend", "End of the Line"),
        ("https://www.youtube.com/watch?v=Z1A1PSre14M", "Insect Warfare", "Human Trafficking"),
        ("https://www.youtube.com/watch?v=SXUBY64Kj1g", "Leng Tch'e", "The Fist of the Leng Tch'e"),
        ("https://www.youtube.com/watch?v=C9aYcSElnuw", "Magrudergrind", "Siphon Then Slit"),
        ("https://www.youtube.com/watch?v=TlhwhCp8dx4", "Nails", "You Will Never Be One of Us"),
        ("https://www.youtube.com/watch?v=_-ywSPWu3K8", "Napalm Death", "You Suffer"),
        ("https://www.youtube.com/watch?v=PSTbpcJO5hI", "Nasum", "This is..."),
        ("https://www.youtube.com/watch?v=zmo_GBNv8SA", "No One Knows What The Dead Think", "Autumn Flower"),
        ("https://www.youtube.com/watch?v=NuVgG5RbXkc", "Pig Destroyer", "Evacuating Heaven"),
        ("https://www.youtube.com/watch?v=ESlipWyUU8M", "P.L.F.", "Ultimate Whirlwind of Incineration"),
        ("https://www.youtube.com/watch?v=gWzsFtiD-ow", "Repulsion", "Horrified"),
        ("https://www.youtube.com/watch?v=fJDRtQxOzAU", "Rotten Sound", "Sharing"),
        ("https://www.youtube.com/watch?v=AG-menlRlAg", "Terrorizer", "Fear of Napalm"),
        ("https://www.youtube.com/watch?v=PLxXC-V8wLs", "Trap Them", "Hellionaires"),
        ("https://www.youtube.com/watch?v=6qoKvbDcO3U", "Wormrot", "Deceased Occupation")
    ],
    "thrash_crossover": [
        ("https://www.youtube.com/watch?v=XphUURIAx5g", "Anthrax", "Caught in a Mosh"),
        ("https://www.youtube.com/watch?v=dDjVT0xL9nY", "Celtic Frost", "Circle of the Tyrants"),
        ("https://www.youtube.com/watch?v=hDDTmpw3fYg", "Dark Angel", "Darkness Descends"),
        ("https://www.youtube.com/watch?v=kzYMnGn5NSI", "Deathhammer", "Lead Us Into Hell"),
        ("https://www.youtube.com/watch?v=9BVF1nP96H0", "Demolition Hammer", "Skull Fracturing Nightmare"),
        ("https://www.youtube.com/watch?v=hf-2HbZi3gU", "Destruction", "Curse the Gods"),
        ("https://www.youtube.com/watch?v=8kA562gTff8", "Devastation", "Deliver the Suffering"),
        ("https://www.youtube.com/watch?v=OuFba4XhCNg", "Exodus", "Blacklist"),
        ("https://www.youtube.com/watch?v=-BzWmtY7tVg", "Hexecutor", "Visitations of a Lascivious Entity"),
        ("https://www.youtube.com/watch?v=7PyvU9iSq50", "Kreator", "Pleasure to Kill"),
        ("https://www.youtube.com/watch?v=L8HhOMNrulE", "Megadeth", "Tornado of Souls"),
        ("https://www.youtube.com/watch?v=beWvPkYHaic", "Metallica", "Whiplash"),
        ("https://www.youtube.com/watch?v=ByuQfqS8G00", "Morbid Saint", "Lock Up Your Children"),
        ("https://www.youtube.com/watch?v=c6ljmB8pHBA", "Mortuary Drape", "Primordial"),
        ("https://www.youtube.com/watch?v=9lZ2mNJrnDQ", "Municipal Waste", "Sadistic Magician"),
        ("https://www.youtube.com/watch?v=e1sDFdsDZY0", "Overkill", "Elimination"),
        ("https://www.youtube.com/watch?v=xdE4RO_P-qE", "Power Trip", "Executioner’s Tax (Swing of the Axe)"),
        ("https://www.youtube.com/watch?v=PDWwr7X0Iy4", "Protector", "Mortuary Nightmare"),
        ("https://www.youtube.com/watch?v=dAMPgKb1G6E", "Razor", "Take This Torch"),
        ("https://www.youtube.com/watch?v=Cuz3t3eUqVs", "S.O.D.", "Speak English or Die"),
        ("https://www.youtube.com/watch?v=TnRZhLRv6eM", "Slayer", "Angel of Death"),
        ("https://www.youtube.com/watch?v=CjP2HZ4-5ps", "Sodom", "Agent Orange"),
        ("https://www.youtube.com/watch?v=BPfkK7bcyfE", "Suicidal Tendencies", "You Can't Bring Me Down"),
        ("https://www.youtube.com/watch?v=p5tbtjYh7Bs", "Tankard", "Need Money For Beer"),
        ("https://www.youtube.com/watch?v=j8C3Tez3iAY", "Vio-lence", "Eternal Nightmare")
    ],
    "war_black_death": [
        ("https://www.youtube.com/watch?v=tdpkD80MW4g", "Abysmal Lord", "Exaltation of the Infernal Cabal"),
        ("https://www.youtube.com/watch?v=JJwx1RG7-zc", "Antediluvian", "Winged Ascent unto the Twelve Runed Solar Anus"),
        ("https://www.youtube.com/watch?v=PZxTf43C7xA", "Antichrist Siege Machine", "Chaos Insignia"),
        ("https://www.youtube.com/watch?v=D9suOBHYgS4", "Axis of Advance", "Annihilation"),
        ("https://www.youtube.com/watch?v=HHFlrSQYP7s", "Archgoat", "Nuns, Cunts & Darkness"),
        ("https://www.youtube.com/watch?v=UHSFojiJwbc", "Beherit", "Down There"),
        ("https://www.youtube.com/watch?v=UHSFojiJwbc", "Bestial Warlust", "Heathens"),
        ("https://www.youtube.com/watch?v=WAbJcDzjrLA", "Black Witchery", "Holocaustic Church Devastation"),
        ("https://www.youtube.com/watch?v=PX3f5VppN_g", "Blasphemophagher", "Ritual of Disintegration"),
        ("https://www.youtube.com/watch?v=OWD9Ht7HDaY", "Blasphemy", "Fallen Angel of Doom"),
        ("https://www.youtube.com/watch?v=ggNyyiF_IBM", "Conqueror", "Infinite Majesty"),
        ("https://www.youtube.com/watch?v=7lK84CyfEzk", "Death Worship", "Holocaust Altar"),
        ("https://www.youtube.com/watch?v=fxZeNpxpEF0", "Deiphago", "Into the Eye of Satan"),
        ("https://www.youtube.com/watch?v=8O4rBcIG2fk", "Diocletian", "Werewolf Directive"),
        ("https://www.youtube.com/watch?v=r7vJVxBcxXQ", "Goatpenis", "Glorious Immoral Carnage"),
        ("https://www.youtube.com/watch?v=Jw5oqm_x3cI", "Heresiarch", "Iron Harvest"),
        ("https://www.youtube.com/watch?v=3WPmVcpYcCs", "Impiety", "Christfuckingchrist"),
        ("https://www.youtube.com/watch?v=R1_UU2icSdY", "Kapala", "Martial Dominance"),
        ("https://www.youtube.com/watch?v=6ipQ_ly0IUA", "Nuclearhammer", "Multi-Dimensional Prism of Black Hatred"),
        ("https://www.youtube.com/watch?v=PVvxf-A1Ejo", "Proclamation", "Messiah of Darkness and Impurity"),
        ("https://www.youtube.com/watch?v=3Qq9NDGqbwY", "Revenge", "Flashpoint Heretic (Flame Thrown)"),
        ("https://www.youtube.com/watch?v=CnB3plYSFEk", "Sadistik Exekution", "Suspiral"),
        ("https://www.youtube.com/watch?v=aI1ItXX3QV0", "Sarcófago", "I.N.R.I."),
        ("https://www.youtube.com/watch?v=T5PFNbxpHzk", "Teitanblood", "Seven Chalices of Vomit and Blood"),
        ("https://www.youtube.com/watch?v=fMdt7fvhPh8", "Vassafor", "Archeonauts Return")
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