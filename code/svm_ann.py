import pickle
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.gaussian_process.kernels import RBF, RationalQuadratic,ConstantKernel
from sklearn.svm import SVC
from sklearn.neural_network import MLPRegressor, MLPClassifier
import trait_labeling as tlab

def main(path_to_pickle, print_predictions=True):

    all_features_per_task = pickle.load(open(path_to_pickle, 'rb'))

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
            X_train, X_test, y_train, y_test = train_test_split(
                all_features_per_task[task_no][potential_features],
                all_features_per_task[task_no][big5[big5_no]+"_label"],
                test_size=0.15, random_state=42)


            model = SVC(C=1.0, kernel='rbf', degree=3, gamma='auto', coef0=0.0, shrinking=True, probability=False, tol=0.001, cache_size=500, class_weight=None, verbose=False, max_iter=-1, decision_function_shape='ovr', random_state=None)
            #model = MLPClassifier(hidden_layer_sizes=(100, ), activation='logistic', solver='sgd', alpha=0.0001, batch_size='auto', learning_rate='adaptive', learning_rate_init=0.001, power_t=0.5, max_iter=500, shuffle=True, random_state=None, tol=0.0001, verbose=False, warm_start=False, momentum=0.9, nesterovs_momentum=True, early_stopping=False, validation_fraction=0.1, beta_1=0.9, beta_2=0.999, epsilon=1e-8)


            model.fit(X_train, y_train)
            score = model.score(X_test, y_test)
            print 'score for "{0}" from task {1}: {2}'.format(big5[big5_no]+"_label", task_no, score)

            if score > predictions[task_no,big5_no]:
                predictions[task_no, big5_no] = score


            #print '***\nRegression for "{0}" from observing task {1}.\nScore: {2}'.format(big5[big5_no], task_no, score)
            prediction = model.predict(X_test)
            print prediction

    if acc_no > 0:
        mean_score /= acc_no

    #print '\n\n______ Total Accurate Predictions:  ' + str(acc_no)
    #print '______ Mean Score = '+ str(mean_score)

    print str(model)+'\n'

    print big5

    for i in predictions:
        print i
    print '\n'
    print 'Average Overall Score:   '+ str(predictions.mean())



    sns.heatmap(predictions, annot=True)
    plt.show()

if __name__ == "__main__":
    main(path_to_pickle='../pickle_data/feature_dataframes/all_features_per_task.p')
