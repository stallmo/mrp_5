# mrp_5: UM Master Research Project: Fusion of trajectory and binary information for human action recognition

|Programming language| Python 2.7
|---                 |---

This is the repository for all code that is related to the above Master's Research Project at Maastricht University.

The requirements.txt should contain all packages that are needed to run the code.


### Structure of the repository

* code
  * all python code goes here
* notebooks
  * all jupyter notebooks go here
  * note that some of the notebooks also produce the feature dataframes. If needed, they can be exported to .py files
* data
  * the data goes in here (has to be added manually)
  * there are two folders:
    1. behavior_AND_personality_dataset (initial dataset)
    2. data_recordings_master (dataset we recorded)
  * note: due to a duplicate subject20, files containing subject20 in their filename in data_recordings_master/ must be changed to subject50 
* pickle_data
  * all pickle files go here
    * pickled feature dataframes
