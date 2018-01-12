from sklearn.decomposition import PCA
import pandas as pd

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
    promising_features = list(pd.DataFrame(df[columns].var()).sort_values(by=0, ascending=False).index[:n_var_features])
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