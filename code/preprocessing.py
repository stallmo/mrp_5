from sklearn import preprocessing
import pandas as pd
import numpy as np

def remove_outliers(df, low_percentil = 0.01, high_percentil=0.99):
    filt_df = df.loc[:, df.columns[4:]] #ignore subject, time, frameId, tracking
    quantiles = filt_df.quantile([low_percentil, high_percentil]) #dataframe holding quantiles for each column
    #print(quantiles.head())
    
    filt_df = filt_df.apply(lambda col: col[(col>quantiles.loc[low_percentil, col.name]) &
                                  (col<quantiles.loc[high_percentil, col.name])], axis=0) #subset based on quantiles (= returns nan for values that are not in given range)

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
        #print(type(scaled_values))
        df_scaled = pd.DataFrame(scaled_values, columns = df.columns[4:])
        #print(df_scaled)
        df_scaled = pd.concat([df.loc[:, df.columns[:4]], df_scaled], axis=1)
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
                print('Setting out of bounds')
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
