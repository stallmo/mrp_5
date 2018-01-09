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
* data
  * the data goes in here, but should not be committed! Every developper has the data locally stored.
  * there are two folders:
    1. behavior_AND_personality_dataset (initial dataset)
    2. data_recordings_master (dataset we recorded)
  * note: due to a duplicate subject20, files containing subject20 in their filename in data_recordings_master/ must be changed to subject50 
* pickle_data
  * all pickle files go here
    * especially pickled feature dataframes

### Basic proceeding for adding code
(eventually create issue)

1. create new branch
2. checkout branch
3. develop on new branch
4. commit to new branch
5. when the code is tested and ready, merge the branch into master

Getting familiar with git: http://rogerdudler.github.io/git-guide/

Useful for installing python and managing virtual environments: https://conda.io/docs/user-guide/install/index.html
