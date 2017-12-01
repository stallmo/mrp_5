from sklearn import preprocessing
import pandas as pd
import numpy as np
from math import tan

def remove_outliers(df, low_percentil = 0.01, high_percentil=0.99):
    filt_df = df.loc[:, df.columns[4:]] #ignore subject, time, frameId, tracking
    quantiles = filt_df.quantile([low_percentil, high_percentil]) #dataframe holding quantiles for each column
    #print(quantiles.head())
    
    filt_df = filt_df.apply(lambda col: col[(col>quantiles.loc[low_percentil, col.name]) &
                                  (col<quantiles.loc[high_percentil, col.name])], axis=0) #subset based on quantiles (=nan for values that are not in given range)

    filt_df = pd.concat([df.loc[:,df.columns[:4]], filt_df], axis=1) #join subject, frameId, time trackingId back to the df
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
        scaled_values = scaler.fit_transform(df.loc[:,df.columns[4:]])
        print(type(scaled_values))
        df_scaled = pd.DataFrame(scaled_values, columns = df.columns[4:])
        #print(df_scaled)
        df_scaled = pd.concat([df.loc[:, df.columns[:4]], df_scaled], axis=1)
    else:
        #to do: check if input columns are correct
        scaled_values = scaler.fit_transform(df.loc[:,columns])
        df_scaled = pd.DataFrame(scaled_values, columns = columns)
    
    return df_scaled

def recalculate_joint_positions(df, joint_name):
    # TODO: Add column parameter that defines which joints are to be translated.
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
    # get ratio of x and y with regards to the middle of the screen
    df[joint_name + "_real_x"] = df.apply(lambda row: get_x_ratio(row["head_x"]) * get_x(row['head_z']))
    df[joint_name + "_real_y"] = df.apply(lambda row: get_y_ratio(row["head_y"]) * get_y(row['head_z']))