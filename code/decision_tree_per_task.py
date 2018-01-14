from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import DecisionTreeRegressor

from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import ExtraTreesRegressor

from sklearn.model_selection import train_test_split
from pickle import load
from pandas import DataFrame

from trait_labeling import label

def main(path_to_pickle, label_cutoff=None):
    data = load(open(path_to_pickle, "r"))
    dfs = []
    for x in data:
        dfs.append(DataFrame(x))
    big5 = ['extraversion', 'agreeableness', 'conscientiousness', 'neuroticism', 'openness_to_experience']
    potential_features = [col for col in list(data[0].columns) if
                          not col in big5 + ['subject', 'index', 'task']]

    unimportant_columns = big5 + ['subject', 'index', 'task']

    labels = [x + "_label" for x in big5]



    task_num = 0
    for task in dfs:
        #apply labels
        task = label(task)

        task_num += 1
        print "task " + str(task_num)
        feature_columns = [x for x in task.columns if x not in unimportant_columns]
        for trait in big5:
            training_set, validation_set = train_test_split(task)

            trait_label = trait + "_label"
            classifier = DecisionTreeClassifier()
            classifier.fit(training_set[feature_columns], training_set[trait_label])
            score = classifier.score(validation_set[feature_columns], validation_set[trait_label])
            print format("score of {} for task {} and trait {}", score, task_num, trait)






if __name__ == '__main__':
    main('../pickle_data/feature_dataframes/all_features_per_task.p')