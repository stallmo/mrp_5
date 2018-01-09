from sklearn import preprocessing
import pandas as pd
import numpy as np
from math import tan
from pandas.stats.moments import ewma

def remove_outliers(df, low_percentil = 0.01, high_percentil=0.99):
    column_number_task = len(df.columns)-1
    filt_df = df.loc[:, df.columns[4:column_number_task]] #ignore subject, time, frameId, tracking, task
    quantiles = filt_df.quantile([low_percentil, high_percentil]) #dataframe holding quantiles for each column
    #print(quantiles.head())
    
    filt_df = filt_df.apply(lambda col: col[(col>quantiles.loc[low_percentil, col.name]) &
                                  (col<quantiles.loc[high_percentil, col.name])], axis=0) #subset based on quantiles (= returns nan for values that are not in given range)

    filt_df = pd.concat([df.loc[:,df.columns[:4]], filt_df, df.loc[:, df.columns[column_number_task]]], axis=1) #join subject, frameId, time, trackingId, task back to the df
    filt_df.dropna(inplace=True) #remove nan values
    filt_df.reset_index(inplace=True) #tidy messy index
    filt_df.drop('index', axis=1, inplace=True) #drop index row implicitly introduced in previous line
    #print(filt_df.head())
    return filt_df
    

def normalize_data(df, columns = None):
    """
    scales for given dataframe columns into range (0,1)
    if columns is specified (list of column names), only given columns are scaled
    """
    
    #columns not specified -> scale all columns
    scaler = preprocessing.MinMaxScaler() #performs scaling inplace when copy=false
    if columns is None:
        #df_values = df.loc[:,df.columns[4:]].values #np.array
        #last column is tasks: do not scale
        column_number_task = len(df.columns)-1
        scaled_values = scaler.fit_transform(df.loc[:,df.columns[4:column_number_task]])
        #print(type(scaled_values))
        df_scaled = pd.DataFrame(scaled_values, columns = df.columns[4:column_number_task])
        #print(df_scaled)
        df_scaled = pd.concat([df.loc[:, df.columns[:4]], df_scaled, df.loc[:, df.columns[column_number_task]]], axis=1)
    else:
        #to do: check if input columns are correct
        scaled_values = scaler.fit_transform(df.loc[:,columns])
        df_scaled = pd.DataFrame(scaled_values, columns = columns)
    
    return df_scaled

def get_sequences_with_little_movement(df, variables_to_check, max_mov = 0.05, min_frames_per_sequence = 60):
    """
    :type df: pandas dataframe
    :param df: dataframe (must be normalized) with all frames which should be checked
    :type variables_to_check: list of strings
    :param variables_to_check: column names of the variables that should be checked for movement, e.g. ['head_x', 'head_y', 'head_z']
    :type max_mov: double
    :param max_mov: defines "little movement", maximum difference between two joint positions
    :type min_frames_per_sequence: integer
    :param min_frames_per_sequence: minimum number of frames that are needed be added to the output
    :rtype: list of dataframes
    """
    little_movement_dfs = [] #list of dataframes
    not_moving = True
    starting_frame_no = 0
    out_of_bounds = False
    
    while not out_of_bounds:
        compare_frame_no = starting_frame_no +1
        not_moving = True
        base_frame = df.iloc[starting_frame_no][variables_to_check]
        current_seq_len = 0        
        
        while not_moving:
            if compare_frame_no>=len(df)-1:
                #print('Setting out of bounds')
                out_of_bounds = True
                break
                
            #print('Comparing frame {0} and frame {1}'.format(starting_frame_no, compare_frame_no))
            movements = base_frame.values - df.iloc[compare_frame_no][variables_to_check].values
            
            #check whether subeject moved
            if any([np.abs(m)>max_mov for m in movements]):                 
                #if sequence was long enough append to dataframes list
                if current_seq_len>=min_frames_per_sequence:
                    little_movement_dfs.append(df[starting_frame_no:starting_frame_no+current_seq_len])
                    
                starting_frame_no = compare_frame_no
                not_moving = False
            
            current_seq_len += 1
            compare_frame_no += 1
            
    return little_movement_dfs


def filter_noise(df, interval):
    # int interval over which applying the mean
    
    return df.groupby(np.arange(len(df))//interval).mean()

def ewma_noise_filter(df):
    return pd.ewma(df, span=30)


def recalculate_joint_positions(df, joint_name):
    # Parameter should be the joint names such that the x and y are JOINT_x and JOINT_y respectively.
    # The function adds a JOINT_real_x and JOINT_real_y to the Dataframe
    """
    Translates joint x and y coordinate axes to meters from pixels
    :param df: DataFrame containing the joint information
    :param joint_name: Name of the joint that has to be translated. The dataframeshould contain columns "JOINT_NAME_x",
    "JOINT_NAME_y" and "JOINT_NAME_z"
    :return:
    """
    # assumes 512 x 424 resolution for IR sensor
    width = 512
    height = 424
    middle_x = width/2
    middle_y = height/2

    # assumes FOV: 70 x 60 degrees
    alpha = 70
    beta = 60

    half_alpha = alpha/2
    half_beta = beta/2

    x_label = joint_name + "_x"
    y_label = joint_name + "_y"
    z_label = joint_name + "_z"
    # calculates the ratio of the of a coordinate axis from the middle
    get_x_ratio = lambda x: (x - middle_x) / middle_x
    get_y_ratio = lambda y: (y - middle_y) / middle_y
    # gets the dimensions of the view pane at some Z
    get_x = lambda z: tan(half_alpha) * z
    get_y = lambda z: tan(half_beta) * z

    def real_x(row):
        return get_x_ratio(row[x_label]) * get_x(row[z_label])

    # get ratio of x and y with regards to the middle of the screen
    df[joint_name + "_real_x"] = df.apply(real_x, axis=1)

    df[joint_name + "_real_y"] = df.apply(lambda row: get_y_ratio(row[y_label]) * get_y(row[z_label]), axis=1)
    df[joint_name + "_real_z"] = df[z_label]
    return df
