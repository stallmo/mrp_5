import pickle
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

def main(path_to_pickle, print_predictions=True):
    all_features_per_task = pickle.load(open(path_to_pickle, 'rb'))

    big5 = ['extraversion', 'agreeableness', 'conscientiousness', 'neuroticism', 'openness_to_experience']
    potential_features = [col for col in list(all_features_per_task[0].columns) if
                          not col in big5 + ['subject', 'index', 'task']]
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
        RandomForestRegressor(n_estimators=1000, criterion='mse', max_depth=None, min_samples_split=2, min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_features='auto', max_leaf_nodes=None, min_impurity_decrease=0.0, min_impurity_split=None, bootstrap=True, oob_score=False, n_jobs=1, random_state=None, verbose=0, warm_start=False),
        # 5 good estim
        KNeighborsRegressor(n_neighbors=3, weights='uniform', algorithm='auto', leaf_size=30, p=2, metric='euclidean', metric_params=None, n_jobs=1),
        # 7 good estim
        MLPRegressor(hidden_layer_sizes=(100, ), activation='tanh', solver='sgd', alpha=0.0001, batch_size='auto', learning_rate='adaptive', learning_rate_init=0.001, power_t=0.5, max_iter=200, shuffle=True, random_state=None, tol=0.0001, verbose=False, warm_start=False, momentum=0.9, nesterovs_momentum=True, early_stopping=False, validation_fraction=0.1, beta_1=0.9, beta_2=0.999, epsilon=1e-08),
        # 3 good estim
        AdaBoostRegressor(base_estimator=None, n_estimators=50, learning_rate=1.0, loss='linear', random_state=None),
    ]

    mod_names = ['Lin_Reg', 'Gauss_P', 'SVM', 'Bayes_R', 'Rand_F', 'KNN', 'ANN', 'Ada_B']

    predictions = np.zeros((6,5))-1
    best_models = np.chararray((6,5), itemsize = 10)
    best_models[:] = 'Nan'


    for mod in range(len(models)):

        print models[mod]
        print '\n ---------------------- \n'

        mean_score = 0
        acc_no = 0
        task_no = 0
        big5_no = 0

        for task_no in range(6):
            # var_features = list(all_features_per_task[task_no][potential_features].var().index[(all_features_per_task[task_no][potential_features].var()>0.001).values])

            for big5_no in range(5):
                X_train, X_test, y_train, y_test = train_test_split(
                    all_features_per_task[task_no][potential_features],
                    all_features_per_task[task_no][big5[big5_no]],
                    test_size=0.2, random_state=42)


                model = models[mod]

                model.fit(X_train, y_train)
                score = model.score(X_test, y_test)

                if score > predictions[task_no,big5_no]:
                    predictions[task_no, big5_no] = score
                    best_models[task_no, big5_no] = mod_names[mod]

                #print '***\nRegression for "{0}" from observing task {1}.\nScore: {2}'.format(big5[big5_no], task_no, score)
                prediction = model.predict(X_test)

                if False:
                #if score > 0.0:
                    mean_score += score
                    acc_no += 1
                    print '*' * 10
                    print 'Predictions'
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

if __name__ == "__main__":
    main(path_to_pickle='../pickle_data/feature_dataframes/all_features_per_task.p')
