import pickle
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.gaussian_process import GaussianProcessRegressor, GaussianProcessClassifier
from sklearn.model_selection import train_test_split
from sklearn.gaussian_process.kernels import RBF, RationalQuadratic,ConstantKernel
from sklearn.svm import SVR
from sklearn.svm import SVC
from sklearn.linear_model import BayesianRidge
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor, MLPClassifier
from  sklearn.ensemble import AdaBoostRegressor
from sklearn.naive_bayes import GaussianNB
import trait_labeling as tlab
from sklearn.model_selection import cross_val_score
import featureSpaceProcessing as fSP
import random
from sklearn.ensemble import VotingClassifier
from sklearn.model_selection import cross_val_predict

def main(path_to_pickle, print_predictions=True):

    feature_selection = True

    all_features_per_task = pickle.load(open(path_to_pickle, 'rb'))

    for tasks in all_features_per_task:
        tlab.label(tasks)


    big5 = ['extraversion', 'agreeableness', 'conscientiousness', 'neuroticism', 'openness_to_experience']
    big5_labels = ['extraversion_label', 'agreeableness_label', 'conscientiousness_label', 'neuroticism_label', 'openness_to_experience_label']
    potential_features = [col for col in list(all_features_per_task[0].columns) if
                          not col in ['subject', 'index', 'task']]

    scores = np.zeros((6,5))-1

    final_scores = []

    for big5_no in range(5):
        preds = []

        for task_no in range(6):

            if feature_selection:
                selected_features = fSP.top_correlated_features(df=all_features_per_task[task_no],
                                                                feature_columns=potential_features,
                                                                correlate_to=big5_labels[big5_no],
                                                                threshold=0.3,
                                                                remove_from_feature_columns=big5_labels + big5,
                                                                n_min_vars=6)
            else:
                selected_features = [col for col in list(all_features_per_task[0].columns) if
                          not col in big5+big5_labels+['subject', 'index', 'task']]
                # model =  RandomForestClassifier(n_estimators=500, max_depth=None, min_samples_split=2, min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_features='auto', max_leaf_nodes=None, bootstrap=True, oob_score=True, n_jobs=1, random_state=None, verbose=0, warm_start=False)
            model = GaussianProcessClassifier(kernel=RBF(), optimizer='fmin_l_bfgs_b', n_restarts_optimizer=5,
                                              max_iter_predict=500, warm_start=False, copy_X_train=True,
                                              random_state=None, multi_class='one_vs_rest', n_jobs=1)
            scores[task_no, big5_no] = np.array(
                cross_val_score(model, all_features_per_task[task_no][selected_features],
                                all_features_per_task[task_no][big5[big5_no] + "_label"])).mean()


            #model =  RandomForestClassifier(n_estimators=500, max_depth=None, min_samples_split=2, min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_features='auto', max_leaf_nodes=None, bootstrap=True, oob_score=True, n_jobs=1, random_state=None, verbose=0, warm_start=False)
            model = GaussianProcessClassifier(kernel=RBF(), optimizer='fmin_l_bfgs_b', n_restarts_optimizer=5, max_iter_predict=500, warm_start=False, copy_X_train=True, random_state=None, multi_class='one_vs_rest', n_jobs=1)

            scores[task_no, big5_no] = np.array(
                cross_val_score(model, all_features_per_task[task_no][selected_features],
                                all_features_per_task[task_no][big5[big5_no] + "_label"])).mean()

            preds.append(cross_val_predict(model, all_features_per_task[task_no][selected_features], all_features_per_task[task_no][big5_labels[big5_no]]))

        new_features = np.array(preds).T

        #model2 =  RandomForestClassifier(n_estimators=500, max_depth=None, min_samples_split=2, min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_features='auto', max_leaf_nodes=None, bootstrap=True, oob_score=True, n_jobs=1, random_state=None, verbose=0, warm_start=False)
        model2 = GaussianProcessClassifier(kernel=RBF(), optimizer='fmin_l_bfgs_b', n_restarts_optimizer=5, max_iter_predict=500, warm_start=False, copy_X_train=True, random_state=None, multi_class='one_vs_rest', n_jobs=1)

        X_train, X_test, y_train, y_test = train_test_split(
            new_features,
            all_features_per_task[task_no][big5_labels[big5_no]],
            test_size=0.15, random_state=43)


        model2.fit(X_train, y_train)
        score = model2.score(X_test, y_test)
        final_scores.append(score)



    print 'Initial scores:'
    print scores.mean(0)
    print 'Overall accuracy:  ' + str(scores.mean())
    print '\nFinal scores:'
    print final_scores
    print 'Overall accuracy:  ' + str(np.array(final_scores).mean())




if __name__ == "__main__":
    main(path_to_pickle='all_features_restructured.p')
