# Does the Higgs boson exist?
### Scientists were able to answer this question only in 2013. 
### And in this project we applied machine learning methods to CERN particle accelerator data to reproduce the process "discovery" of the Higgs particle.
#### Project structure:

- ML_EPFL_project1.pdf - our paper.

- hepers.py - file containing helper functions to load data

- implementations.py - file contatining ML methods used in this project

- final.ipynb - notebook with feature extraction and our experiments that are presented in our paper.

- run.py - script for reproducing our submission. Usage:

```shell
python3 run.py --data_path <path_to_data_folder>
```

the data folder should contain train.csv and test.csv files.
