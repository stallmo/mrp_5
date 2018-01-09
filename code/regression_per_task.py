import pickle
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

def main(path_to_pickle, print_predictions=True):
    all_features_per_task = pickle.load(open(path_to_pickle, 'rb'))
    
    big5 = ['extraversion', 'agreeableness', 'conscientiousness', 'neuroticism', 'openness_to_experience']
    potential_features = [col for col in list(all_features_per_task[0].columns) if not col in big5+['subject', 'index', 'task']]
    #train models for extraversion per task
    
    task_no = 0
    big5_no = 0
    for task_no in range(6):
        #var_features = list(all_features_per_task[task_no][potential_features].var().index[(all_features_per_task[task_no][potential_features].var()>0.001).values])
        for big5_no in range(5):
            X_train, X_test, y_train, y_test = train_test_split(
                    all_features_per_task[task_no][potential_features],
                    all_features_per_task[task_no][big5[big5_no]],
                    test_size=0.2, random_state=42)
            
            model = LinearRegression(fit_intercept=True,
                                     normalize=True,
                                     copy_X=True,
                                     n_jobs=1)
            model.fit(X_train, y_train)
            score = model.score(X_test, y_test)
            print '***\nRegression for "{0}" from observing task {1}.\nScore: {2}'.format(big5[big5_no], task_no, score)
            prediction = model.predict(X_test)
            if print_predictions:
                print '*'*10
                print 'Predictions'
                print'*'*10
                for pred_no, pred in enumerate(prediction):
                    print 'Prediction: {0}\nActual: {1}'.format(pred, y_test.values[pred_no])
                print
    
if __name__=="__main__":
    main(path_to_pickle='../pickle_data/all_features_per_task.p', print_predictions=False)