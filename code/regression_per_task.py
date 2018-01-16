import pickle
import copy
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.model_selection import train_test_split
from sklearn.gaussian_process.kernels import RBF, RationalQuadratic,ConstantKernel
from sklearn.svm import SVR
from sklearn.linear_model import BayesianRidge
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from  sklearn.ensemble import AdaBoostRegressor

import featureSpaceProcessing

def main(path_to_pickle, test_size=0.2, print_extended=True, pca=False, n_pca_var=10, n_pca_cor = 10, variance_features = None, correlation_features = None, n_min_corr_vars=3, only_normal_distributed = False, random_seed=42):
    
    all_features_per_task = pickle.load(open(path_to_pickle, 'rb'))

    big5 = ['extraversion', 'agreeableness', 'conscientiousness', 'neuroticism', 'openness_to_experience']
    #potential_features = [col for col in list(all_features_per_task[0].columns) if
    #                      not col in big5 + ['subject', 'index', 'task']]
    
    # train models for extraversion per task

    models = [

        LinearRegression(fit_intercept=True,normalize=True,copy_X=True, n_jobs=1),
        # 0 good estim
        GaussianProcessRegressor (kernel=None, alpha=1e-10, optimizer='fmin_l_bfgs_b', n_restarts_optimizer=0, normalize_y=True, copy_X_train=True, random_state=None),
        # 6 good estim
        SVR(kernel='rbf', degree=3, gamma='auto', coef0=0.0, tol=0.001, C=10, epsilon=0.1, shrinking=True, cache_size=200, verbose=False, max_iter=-1),
        # 4 good estim
        BayesianRidge(n_iter=300, tol=0.001, alpha_1=1e-06, alpha_2=1e-06, lambda_1=1e-06, lambda_2=1e-06, compute_score=False, fit_intercept=True, normalize=False, copy_X=True, verbose=False),
        # 2 good estim
        #RandomForestRegressor(n_estimators=1000, criterion='mse', max_depth=None, min_samples_split=2, min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_features='auto', max_leaf_nodes=None, min_impurity_decrease=0.0, min_impurity_split=None, bootstrap=True, oob_score=False, n_jobs=1, random_state=None, verbose=0, warm_start=False),
        # 5 good estim
        KNeighborsRegressor(n_neighbors=3, weights='uniform', algorithm='auto', leaf_size=30, p=2, metric='euclidean', metric_params=None, n_jobs=1),
        # 7 good estim
        MLPRegressor(hidden_layer_sizes=(100, ), activation='tanh', solver='sgd', alpha=0.0001, batch_size='auto', learning_rate='adaptive', learning_rate_init=0.001, power_t=0.5, max_iter=200, shuffle=True, random_state=None, tol=0.0001, verbose=False, warm_start=False, momentum=0.9, nesterovs_momentum=True, early_stopping=False, validation_fraction=0.1, beta_1=0.9, beta_2=0.999, epsilon=1e-08),
        # 3 good estim
        AdaBoostRegressor(base_estimator=None, n_estimators=50, learning_rate=1.0, loss='linear', random_state=None),
    ]

    #mod_names = ['Lin_Reg', 'Gauss_P', 'SVM', 'Bayes_R', 'Rand_F', 'KNN', 'ANN', 'Ada_B']
    mod_names = ['Lin_Reg', 'Gauss_P', 'SVM', 'Bayes_R', 'KNN', 'ANN', 'Ada_B']

    predictions = np.zeros((6,5))-1
    best_models = np.chararray((6,5), itemsize = 10)
    best_models[:] = 'Nan'
    models_dict = _init_models_dict()
    #models_dict = {}


    for mod in range(len(models)):
        
        if print_extended:
            print models[mod]
            print '\n ---------------------- \n'

        mean_score = 0
        acc_no = 0
        task_no = 0
        big5_no = 0
        #test_sie = 0.2
        model = models[mod]

        for task_no in range(6):
            # var_features = list(all_features_per_task[task_no][potential_features].var().index[(all_features_per_task[task_no][potential_features].var()>0.001).values])
            potential_features = [col for col in list(all_features_per_task[task_no].columns) if
                         not col in big5 + ['subject', 'index', 'task']]
            
            if only_normal_distributed:
                potential_features = featureSpaceProcessing.select_normally_distributed(all_features_per_task[task_no], potential_features, p_threshold=0.05)[0]
                #print potential_features

            for big5_no in range(5):
                
                X_train, X_test, y_train, y_test = train_test_split(
                    all_features_per_task[task_no],
                    all_features_per_task[task_no][big5[big5_no]],
                    test_size=test_size, random_state=random_seed)
                
                pca = False #remove pca
                if pca:
                    
                    X_train, X_test, y_train, y_test = train_test_split(
                    all_features_per_task[task_no],
                    all_features_per_task[task_no][big5[big5_no]],
                    test_size=test_size, random_state=42)

                    pca, features_to_use, tmp = featureSpaceProcessing.perform_pca_after_feature_selection(X_train,
                                                                                                           potential_features,
                                                                                                           big5[big5_no],
                                                                                                           n_var_features=n_pca_var,
                                                                                                           n_cor_features=n_pca_cor)
                    X_train_transformed = pca.transform(X_train[features_to_use])
                    X_test_transformed = pca.transform(X_test[features_to_use])
                    
                    model.fit(X_train_transformed, y_train)
                    prediction = model.predict(X_test_transformed)
                    score = model.score(X_test_transformed, y_test)
                    
                else:
                    
                    """X_train, X_test, y_train, y_test = train_test_split(
                    all_features_per_task[task_no],
                    all_features_per_task[task_no][big5[big5_no]],
                    test_size=test_size, random_state=42)"""
                    
                    #var_features = []
                    cor_features = []                    
                    
                    if not variance_features is None:
                        """potential_features = featureSpaceProcessing.top_variance_variables(all_features_per_task,
                                                                                           potential_features,
                                                                                           variance_features,
                                                                                           big5)"""
                        var_features = featureSpaceProcessing.top_variance_variables(df=X_train,
                                                                      feature_columns=potential_features,
                                                                      threshold=variance_features,
                                                                      remove_from_feature_columns = big5)
                    
                    if not correlation_features is None:
                        #print 'Computing correlation features'
                        #print X_train
                        cor_features = featureSpaceProcessing.top_correlated_features(df=X_train,
                                                                                      feature_columns=potential_features,
                                                                                      correlate_to=big5[big5_no],
                                                                                      threshold = correlation_features,
                                                                                      remove_from_feature_columns = big5,
                                                                                      n_min_vars=n_min_corr_vars)

                    if any([not correlation_features is None, not variance_features is None]):    
                        #print len(cor_features)
                        #potential_features = var_features+cor_features #ignore var features
                        potential_features = cor_features
                        #print len(potential_features)
                        
                    #print(potential_features)
                    model.fit(X_train[potential_features], y_train)
                    prediction = model.predict(X_test[potential_features])
                    score = model.score(X_test[potential_features], y_test)
                
                #print score

                if score > predictions[task_no,big5_no]:
                    #print task_no, big5_no, model
                    predictions[task_no, big5_no] = score
                    best_models[task_no, big5_no] = mod_names[mod]                    
                    #save model in dictionary
                    models_dict['task'+str(task_no)]['trait'+str(big5_no)]['model'] = copy.deepcopy(model)
                    models_dict.get('task'+str(task_no)).get('trait'+str(big5_no))['feature_columns'] = copy.deepcopy(potential_features)
                    

                #print '***\nRegression for "{0}" from observing task {1}.\nScore: {2}'.format(big5[big5_no], task_no, score)

                if print_extended:
                #if score > 0.0:
                    mean_score += score
                    acc_no += 1
                    print '*' * 10
                    print 'Predictions for {0} from task {1}.'.format(big5[big5_no], task_no)
                    print '*' * 10
                    for pred_no, pred in enumerate(prediction):
                        print 'Prediction: {0}\nActual: {1}'.format(pred, y_test.values[pred_no])

                        print
        if acc_no > 0:
            mean_score /= acc_no

    #print '\n\n______ Total Accurate Predictions:  ' + str(acc_no)
    #print '______ Mean Score = '+ str(mean_score)

    print big5

    for i in predictions:
        print i
    print '\n\n'

    for i in best_models:
        print i

    sns.heatmap(predictions, annot=True)
    plt.show()
    
    return models_dict

def _init_models_dict():
    """trait_dict = {'model':None,
                  'feature_columns': None}
    task_dict = {'trait'+str(i):{'model':None,
                  'feature_columns': None} for i in range(5)}"""
    
    models_dict = {'task'+str(i):
                        {'trait'+str(i):
                            {'model':None,
                             'feature_columns': None}
                        for i in range(5)}
                    for i in range(6) }
    return models_dict

"""if __name__ == "__main__":
    main(path_to_pickle='../pickle_data/feature_dataframes/all_features_per_task.p',
         test_size=0.2,
         print_predictions=False,
         pca=False,
         variance_features = 10)
#%%
path_to_pickle = '../pickle_data/feature_dataframes/all_features_per_task_w_sensor_norm.p'
test_size = 0.2

models_dict = main(path_to_pickle=path_to_pickle,
                         test_size = test_size,
                         print_extended=False,
                         pca=False,
                         variance_features = 0,
                         correlation_features=8)
#%%
models_dict['task0']['trait0']['feature_columns']
#%%
feature_dfs = pickle.load(open(path_to_pickle, 'rb'))
feature_dfs[0].head()
#%%
model = models_dict['task0']['trait0']['model']
cols = models_dict['task0']['trait0']['feature_columns']
print cols
preds = model.predict(feature_dfs[0][cols])
actual = feature_dfs[0]['extraversion']
for pred, act in zip(preds, actual):
    print 'Predicted: {0}\nActual: {1}\n'.format(pred, act)
print 'Score: {0}'.format(model.score(feature_dfs[0][cols], actual))
#%%
pickle.dump(models_dict, open('../pickle_data/regression_top7_correlated.p', 'wb'))"""