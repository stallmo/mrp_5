from pandas import DataFrame
from cPickle import load

from sklearn.ensemble import AdaBoostRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.linear_model import LinearRegression, BayesianRidge
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.svm import SVR

regressors = [

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

def main(X_path, models_path):
    X = load(open(X_path, "r"))
    dfs = []
    for x in X:
        dfs.append(DataFrame(x))
    big5 = ['extraversion', 'agreeableness', 'conscientiousness', 'neuroticism', 'openness_to_experience']

    # LOAD MODELS
    models = load(open(models_path, "r"))

    # dict for mapping trait numbers to traits
    trait_map = {"trait{}".format(i): big5[i] for i in range(len(big5))}
    # BUILD PREDICTION DATAFRAMES

    # Dataframe per trait
    predictions_per_trait = {}
    # per trait
    for i in range(len(big5)):
        # get trait and create dataFrame
        trait = "trait" + str(i)
        # trait = big5[i]
        trait_df = DataFrame()
        # Add dataframe to dict under trait
        predictions_per_trait[trait] = trait_df
        trait_column = trait_map[trait]
        # per task
        for j in range(len(dfs)):
            task_name = "task" + str(j)
            # get classifier
            cls = models[task_name][trait]["model"]
            # get data columns
            cols = models[task_name][trait]["feature_columns"]
            # get data

            #print("{}, {}, columns: {}".format(trait, task_name, cols))
            if cls != None and cols != None:
                data = dfs[j][cols]
                # predict
                Y = cls.predict(data)
                # add predicted data to trait df column
                predictions_per_trait[trait][task_name] = Y
            else:
                print("#### Missing cls or features for {} and {} ####".format(trait, task_name))
        predictions_per_trait[trait] = predictions_per_trait[trait].join(dfs[0][trait_column])
        # add the trait data to the predictions per trait
    pass
    x = 1 + 1

    trait_models = {}
    for key in predictions_per_trait:
        # For trait train regressor and score
        data = predictions_per_trait[key]
        train, test = train_test_split(data)
        predictors = [x for x in data.columns if x not in big5]
        target = [x for x in data.columns if x in big5]

        max_score = -100
        best_regressor= None
        for regressor in regressors:
            regressor.fit(train[predictors], train[target])
            score = regressor.score(test[predictors], test[target])
            if score > max_score:
                best_regressor = regressor
                max_score = score
        trait_models[key] = {"regressor": best_regressor, "score": score}
    print(trait_models)
    pass



import os

if __name__ == '__main__':
    print "running from " + os.getcwd()
    print os.curdir
    main('pickle_data/feature_dataframes/all_features_per_task_w_sensor.p',
         'pickle_data/regression_top7_correlated.p')
