from math import sqrt

import pandas as pd

def calculate_joint_differences(frames_df, only_for_columns = None):
    """
    :type frames_df: pandas dataframe
    :param: dataframe that holds all joints for which differences should be calculated
    :type only_for_columns: (optional) list of strings
    :param: if given a list of columns for which the differences are calculated
    """
    #n_combinations = special.binom(n_joints_per_frame, 2) #chose two joints out of 21 without repition
    #n_frames = len(frames_df)
    if only_for_columns is None:
        joint_names = list(frames_df) # get column names
    else:
        #to do: check if given column names are are actually. the following is kinda hacky
        joint_names = set(only_for_columns).intersection(set(frames_df)) #only consider column names that are actually in given dataframe
    x_coord_names = [name for name in joint_names if name.endswith('_x')]
    y_coord_names = [name for name in joint_names if name.endswith('_y')]
    z_coord_names = [name for name in joint_names if name.endswith('_z')]
    columns_dict = {}
    #fcc = [[[0 for i in range(3)] for comb in range(int(n_combinations))] for frame in range(n_frames)] #initialize fcc array: differences in all coordinates, pairwise for all_joints for all frames
    for x_name_no, x_name in enumerate(x_coord_names):
        for x_name2_no in range(x_name_no+1, len(x_coord_names)):
            x_name2 = x_coord_names[x_name2_no]
            column_name='{0}-{1}'.format(x_name, x_name2)
            columns_dict[column_name] = frames_df[x_name]-frames_df[x_name2]
            
    for y_name_no, y_name in enumerate(y_coord_names):
        for y_name2_no in range(y_name_no+1, len(y_coord_names)):
            y_name2 = y_coord_names[y_name2_no]
            column_name='{0}-{1}'.format(y_name, y_name2)
            columns_dict[column_name] = frames_df[y_name]-frames_df[y_name2]
    
    for z_name_no, z_name in enumerate(z_coord_names):
        for z_name2_no in range(z_name_no+1, len(z_coord_names)):
            z_name2 = z_coord_names[z_name2_no]
            column_name='{0}-{1}'.format(z_name, z_name2)
            columns_dict[column_name] = frames_df[z_name]-frames_df[z_name2]
    
    differences_df = pd.DataFrame(columns_dict)
    return differences_df


def calculate_3Djoint_differences(frames_df, only_for_columns=None):
    """
    :type frames_df: pandas dataframe
    :param: dataframe that holds all joints for which differences should be calculated
    :type only_for_columns: (optional) list of strings
    :param: if given a list of columns for which the differences are calculated
    """
    # n_combinations = special.binom(n_joints_per_frame, 2) #chose two joints out of 21 without repition
    # n_frames = len(frames_df)
    if only_for_columns is None:
        joint_names = list(frames_df)  # get column names
    else:
        # to do: check if given column names are are actually. the following is kinda hacky
        joint_names = set(only_for_columns).intersection(
            set(frames_df))  # only consider column names that are actually in given dataframe
    x_coord_names = [name for name in joint_names if name.endswith('_x')]
    y_coord_names = [name for name in joint_names if name.endswith('_y')]
    z_coord_names = [name for name in joint_names if name.endswith('_z')]
    names = [name[:-2] for name in x_coord_names]
    columns_dict = {}
    # fcc = [[[0 for i in range(3)] for comb in range(int(n_combinations))] for frame in range(n_frames)] #initialize fcc array: differences in all coordinates, pairwise for all_joints for all frames
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            joint1_name = names[i]
            joint1_x_name = x_coord_names[i]
            joint1_y_name = y_coord_names[i]
            joint1_z_name = z_coord_names[i]

            joint2_name = names[j]
            joint2_x_name = x_coord_names[j]
            joint2_y_name = y_coord_names[j]
            joint2_z_name = z_coord_names[j]

            differences = frames_df.apply(lambda row: sqrt(
                pow(row[joint1_x_name] - row[joint2_x_name], 2) +
                pow(row[joint1_y_name] - row[joint2_y_name], 2) +
                pow(row[joint1_z_name] - row[joint2_z_name], 2)
            ), axis=1)
            columns_dict["{}-{}".format(joint1_name, joint2_name)] = differences

    differences_df = pd.DataFrame(columns_dict)
    return differences_df
