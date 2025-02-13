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
    - lyrics.odt
```

Audio files can be in any format, as long as the ASR model can handle it.
They should be grouped by source to keep datasets together (*e.g.*, EMVD, songs).

File `lyrics.ods` should have one tab per source and one line per song.
You can provide multiple transcripts of the same lyrics (*e.g.*, various listener perceptions, declared lyrics).