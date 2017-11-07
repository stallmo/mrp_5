from sklearn import preprocessing
import pandas as pd
import numpy as np

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
