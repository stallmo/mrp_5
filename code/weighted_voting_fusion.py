import pickle
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
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
import featureSpaceProcessing
from sklearn.ensemble import VotingClassifier

def main(path_to_pickle, print_predictions=True):

    all_features_per_task = pickle.load(open(path_to_pickle, 'rb'))

    for tasks in all_features_per_task:
        tlab.label(tasks)


    big5 = ['extraversion', 'agreeableness', 'conscientiousness', 'neuroticism', 'openness_to_experience']
    big5_labels = ['extraversion_label', 'agreeableness_label', 'conscientiousness_label', 'neuroticism_label', 'openness_to_experience_label']
    potential_features = [col for col in list(all_features_per_task[0].columns) if
                          not col in  big5 + big5_labels+ ['subject', 'index', 'task']]

    scores = np.zeros((6,5))-1
    weights = np.zeros((6,5))

    mean_score = 0
    acc_no = 0
    task_no = 0
    big5_no = 0

    for task_no in range(6):

        for big5_no in range(5):

            #model =  RandomForestClassifier(n_estimators=500, max_depth=None, min_samples_split=2, min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_features='auto', max_leaf_nodes=None, bootstrap=True, oob_score=True, n_jobs=1, random_state=None, verbose=0, warm_start=False)
            model = GaussianProcessClassifier(kernel=RBF(), optimizer='fmin_l_bfgs_b', n_restarts_optimizer=5, max_iter_predict=500, warm_start=False, copy_X_train=True, random_state=None, multi_class='one_vs_rest', n_jobs=1)

            scores[task_no,big5_no] =  np.array(cross_val_score(model, all_features_per_task[task_no][potential_features],all_features_per_task[task_no][big5[big5_no]+"_label"])).mean()


    print big5

    for i in scores:
        print i
    print '\n'
    print 'Average Overall Score:   '+ str(scores.mean())

    print '\n\n'
    totals = []
    for i in range(5):
        totals.append(scores[:,i].sum())

    for i in range(5):
        for j in range(6):
            weights[j,i] = scores[j,i]/totals[i]

    print '\n\n'
    print weights

    accuracy = np.zeros(5)


    for big5_no in range(5):

        preds = []
        for task_no in range(6):


            X_train, X_test, y_train, y_test = train_test_split(
                all_features_per_task[task_no][potential_features],
                all_features_per_task[task_no][big5_labels[big5_no]],
                test_size=0.15, random_state=90)

            model.fit(X_train, y_train)
            preds.append(model.predict(X_test))



        final = []

        tmp = np.array((weights[:,big5_no].dot(np.array(preds))))


        print np.rint(tmp)
        print y_test.values

        acc = 0

        for i in range(len(y_test)):
            if np.rint(tmp)[i] == y_test.values[i]:
                acc +=1

        accuracy[big5_no] = float(acc)/len(y_test)

    print '\n\nBig5 personality    :  ' + str(big5)
    print 'Total final accuracy:  ' + str(accuracy)
    print 'Overall accuracy:  ' + str(accuracy.mean())

if __name__ == "__main__":
    main(path_to_pickle='all_features_restructured.p')
