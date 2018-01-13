from sklearn.decomposition import PCA
from sklearn.feature_selection import VarianceThreshold
import pandas as pd
import numpy as np

def perform_pca_after_feature_selection(df, columns, correlate_to, n_var_features=10, n_cor_features=10):
    """
    performs pca on given dataframe and columns. bases pre feature selection on most variant and correlated features. returns dataframe with pca features and columns on which pca wasn't performed (if needed) and the fitted pca model
    
    :type df: pandas dataframe
    :param df: dataframe that holds columns to perform PCA on
    :type columns: list of strings
    :param columns: column names to perform PCA on
    :type correlate_to: string
    :param: variable name to use for correlation threshold
    :type n_var_features: integer
    :param: n_var_features: specifies how many top variance features should be used
    :type n_cor_features: int
    :param: specifies how many top correlated features should be used
    :rtype: (dataframe, list of features to use for pca, fitted PCA model)
    """
    #TODO: check for validity
    other_columns = [c for c in list(df.columns) if not c in columns+['index']]
    
    #TODO: check if n_var_features and n_cor_features are valid. the sum can not be greater than number of samples!
    
    #get features with highest variance 
    if correlate_to in columns:
        columns.remove(correlate_to)
    promising_features = list(pd.DataFrame(df[columns].var()).sort_values(by=0, ascending=False).index[:n_var_features])
    #promising_features = list(pd.DataFrame(df[columns].var()).sort_values(by=0, ascending=False).index[:n_var_features])
    #get features with highest correlation
    if not correlate_to in columns:
        columns.append(correlate_to)
    promising_features += list(pd.DataFrame(df[columns].corr()[correlate_to]).sort_values(by=correlate_to, ascending=False).index[1:n_cor_features+1])
    promising_features = list(set(promising_features)) #only unique
    #print 'number of features left: ', len(promising_features)
    
    pca = PCA(n_components='mle',
          copy=True,
          whiten=False,
          svd_solver='full',
          tol=0.0,
          iterated_power='auto',
          random_state=42)
    
    #print len(df)
    X_transformed = pca.fit_transform(df[promising_features])
    #print X_train_transformed
    #print(df[other_columns])
    new_column_names = ['pca_'+str(i) for i in range(X_transformed.shape[1])]
    
    df_transformed = pd.DataFrame(X_transformed, columns=new_column_names)
    #df_transformed = pd.concat([pd.DataFrame(X_train_transformed, columns=new_column_names), df[other_columns] ], axis=1)
    
    return pca, promising_features, pd.concat([df_transformed, df[other_columns].reset_index().drop('index', axis=1)], axis=1)

def top_variance_variables(df, feature_columns, threshold, remove_from_feature_columns = None):
    """
    :type df: pandas dataframe
    :param df: df to calculate variances from
    :type all_feature_columns: list of strings
    :param all_feature_columns: column names of features in df
    :type threshold: integer or float
    :param threshold: if integer: `threshold` variables with highest variance; if float: variables with variance higher than threshold
    :type remove_from_feature_columns: list of strings
    :param remove_from_feature_columns: (optional) column names to remove from feature columns (should not be considered when getting variance)
    :rtype: list with features
    """
    
    if not remove_from_feature_columns is None:
        feature_columns = [c for c in feature_columns if not c in remove_from_feature_columns]
    
    if threshold<1:
        return _get_variables_variance_threshold(df, feature_columns, threshold)
    else:
        return _get_top_n_variance_variables(df, feature_columns, threshold)
    
def _get_variables_variance_threshold(df, feature_columns, threshold):
    var_df = pd.DataFrame(df[feature_columns].var())
    return list(var_df[var_df[0]>threshold].index)

def _get_top_n_variance_variables(df, feature_columns, n):
    #print n
    var_features = list(pd.DataFrame(df[feature_columns].var()).sort_values(by=0, ascending=False).index)[:n]
    return var_features

def top_correlated_features(df, feature_columns, correlate_to, threshold, remove_from_feature_columns = None):
    """
    """
    
    if not remove_from_feature_columns is None:
        feature_columns = [c for c in feature_columns if not c in remove_from_feature_columns]
    
    if threshold<1:
        return
    
    else:
        #print 'Getting top n correlated'
        return _get_top_n_correlated_features(df, feature_columns, correlate_to, threshold)
    
def _get_top_n_correlated_features(df, feature_columns, correlate_to, n):
    
    if not correlate_to in feature_columns:
        feature_columns+=[correlate_to]
        
    return list(np.abs(df[feature_columns].corr()[correlate_to]).nlargest(n=n).index)[1:]
    