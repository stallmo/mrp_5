import pickle
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
import trait_labeling as tlab
import featureSpaceProcessing as fSP

def main(path_to_pickle, random_seed, print_predictions=True):

    all_features_per_task = pickle.load(open(path_to_pickle, 'rb'))

    scores = np.zeros((6,5))-1

    for tasks in all_features_per_task:
        tlab.label(tasks)



    big5 = ['extraversion', 'agreeableness', 'conscientiousness', 'neuroticism', 'openness_to_experience']
    big5_labels = ['extraversion_label', 'agreeableness_label', 'conscientiousness_label', 'neuroticism_label', 'openness_to_experience_label']
    potential_features = [col for col in list(all_features_per_task[0].columns) if
                          not col in big5 + big5_labels+ ['subject', 'index', 'task']]


    predictions = np.zeros((6,5))-1
    best_models = np.chararray((6,5), itemsize = 10)
    best_models[:] = 'Nan'

    mean_score = 0
    acc_no = 0
    task_no = 0
    big5_no = 0
    for task_no in range(6):
        # var_features = list(all_features_per_task[task_no][potential_features].var().index[(all_features_per_task[task_no][potential_features].var()>0.001).values])

        for big5_no in range(5):

            task = all_features_per_task[task_no]

            correlated_features = fSP.top_correlated_features(df = task,
                                                              feature_columns = potential_features,
                                                              correlate_to = big5_labels[big5_no],
                                                              threshold = 0.3,
                                                              remove_from_feature_columns=big5_labels+big5,
                                                              n_min_vars=6)
            #print 'Features used for classifying {0} from task {1}: {2}'.format(big5_labels[big5_no], task_no, correlated_features)
            #print

            X_train, X_test, y_train, y_test = train_test_split(
                task[correlated_features],
                task[big5[big5_no] + "_label"],
                test_size=0.15, random_state=random_seed)


            model = GaussianProcessClassifier(kernel=None,
                                              optimizer='fmin_l_bfgs_b',
                                              n_restarts_optimizer=0,
                                              max_iter_predict=100,
                                              warm_start=False,
                                              copy_X_train=True,
                                              random_state=None,
                                              multi_class='one_vs_rest',
                                              n_jobs=1)
            scores[task_no, big5_no] = np.array(
                cross_val_score(model, all_features_per_task[task_no][correlated_features],
                                all_features_per_task[task_no][big5[big5_no] + "_label"])).mean()
            #model = MLPClassifier(hidden_layer_sizes=(100, ), activation='logistic', solver='sgd', alpha=0.0001, batch_size='auto', learning_rate='adaptive', learning_rate_init=0.001, power_t=0.5, max_iter=500, shuffle=True, random_state=None, tol=0.0001, verbose=False, warm_start=False, momentum=0.9, nesterovs_momentum=True, early_stopping=False, validation_fraction=0.1, beta_1=0.9, beta_2=0.999, epsilon=1e-8)


            model.fit(X_train, y_train)
            score = model.score(X_test, y_test)
            #for feature, importance in zip(correlated_features, model.feature_importances_):
                #print 'Importance of feature "{0}" for predicting "{1}" from task "{2}": {3}'.format(feature, big5_labels[big5_no], task_no, importance)
                

            if score > predictions[task_no,big5_no]:
                predictions[task_no, big5_no] = score


            #print '***\nRegression for "{0}" from observing task {1}.\nScore: {2}'.format(big5[big5_no], task_no, score)
            prediction = model.predict(X_test)
            if print_predictions:
                print 'Classes: {0}'.format(model.classes_)
                print 'Prediction for "{0}" from task {1}:\n{2}\nActual: {3}\nProbabilities: {4}\n'.format(big5_labels[big5_no], task_no, prediction, y_test.values, model.predict_proba(X_test))
                #print 'Prediction:', prediction
                #print 'Actual: ', y_test.values

    if acc_no > 0:
        mean_score /= acc_no

    #print '\n\n______ Total Accurate Predictions:  ' + str(acc_no)
    #print '______ Mean Score = '+ str(mean_score)

    if print_predictions:
        print str(model)+'\n'

        print big5

        for i in predictions:
            print i
        print '\n'
    print 'Average Overall Score:   '+ str(scores.mean())

    return scores

    sns.heatmap(predictions, annot=True)
    plt.title("Estimator accuracies per trait per task using random seed: {}".format(random_seed))
    plt.show()


if __name__ == "__main__":
    main("../pickle_data/feature_dataframes/all_features_restructured.p", 1)