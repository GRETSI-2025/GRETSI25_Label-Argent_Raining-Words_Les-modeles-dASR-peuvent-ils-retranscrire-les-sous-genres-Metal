# extreme_vocals_asr

This script is used to evaluate the performance of ASR models on a dataset of metal songs.

**Contacts**

- Bastien Pasdeloup (bastien.pasdeloup@imt-atlantique.fr)
- Axel Marmoret (axel.marmoret@imt-atlantique.fr)

**Dataset**

Dataset should be formatted as follows:
```
- dataset
    - audio
        - source_1
            - file_name_1.{wav, flac...}
            - file_name_2.{wav, flac...}
        - source_2
            - file_name_3.{wav, flac...}
            - file_name_4.{wav, flac...}
        - ...
    - lyrics
        - source_1
            - file_name_1.txt
            - file_name_2.txt
        - source_2
            - file_name_3.txt
            - file_name_4.txt
        - ...
```

Make sure that source directories (*e.g.*, "emvd", "demucs", "songs) and file names (*e.g.*, `bloodbath___like_fire.{...}`) are consistent between audio and lyrics directories.
Files can be in any format, as long as the ASR model can handle it.