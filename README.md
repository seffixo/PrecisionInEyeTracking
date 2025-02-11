# Genauigkeit_Glasses



## Installation

- [ ] Lade dir WinPython herunter (Version >= 3.10.0) (https://winpython.github.io/)
- [ ] Installiere PIP (https://pip.pypa.io/en/stable/installation/)
- [ ] Installiere Gittyup (https://github.com/Murmele/Gittyup/releases/tag/gittyup_v1.4.0); kann sein, dass du PuTTY auch brauchst
- [ ] Hinterlege SSH Key im GitLab


# Implementation Workflow
- Implement your functionalities
- Check your script for PEP8 conformity
- Write or adapt the Unit Test for your script 
- Naming convention for Unit Test Scripts test_"name of the script to be tested".py


### Generate the requirements.txt
The pip command for installing: pip install pipreqs 

The pip command for executing: pipreqs "path of folder containing all python scrits" --force


### Run main.py
use this command to run the .jsonl-type with the corresponding .txt-file. 
the txt-file contains timestamp-ranges where the participant was looking at a certain point of interest. 

this main.py is still a WIP and is only preprocessing jsonl-files to filter unwanted information. 

python main.py --json_files InputFiles\gazedata_Simon1.jsonl --timestamps InputFiles\Simon1.txt