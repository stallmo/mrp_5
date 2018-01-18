
from sklearn.ensemble import ExtraTreesClassifier

from sklearn.tree import DecisionTreeRegressor

from sklearn.ensemble import ExtraTreesRegressor
from numpy import average


from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis

from sklearn.model_selection import train_test_split
from pickle import load
from pandas import DataFrame

from trait_labeling import label

from seaborn import heatmap
import matplotlib.pyplot as plt

from numpy import array, zeros

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

    classifiers = [
        DecisionTreeClassifier(),
        ExtraTreesClassifier(),
        MLPClassifier(),
        KNeighborsClassifier(),
        SVC(),
        GaussianProcessClassifier(),
        RandomForestClassifier(),
        AdaBoostClassifier(),
        GaussianNB(),
        QuadraticDiscriminantAnalysis()
    ]

    results = array(zeros((len(dfs), len(big5))))
    task_num = 0
    for task_i in range(len(dfs)):
        task = dfs[task_i]
        feature_columns = [x for x in task.columns if x not in unimportant_columns]
        #apply labels
        label(task)

        task_num += 1

        #print "task " + str(task_num)
        print "###### Task " + str(task_num) + " #######"

        for trait_i in range(len(big5)):
            trait = big5[trait_i]
            training_set, validation_set = train_test_split(task)

            trait_label = trait + "_label"
            # label_average = task[trait_label].mean()
            # print str.format("TASK {}, TRAIT {}: average value {}", task_num, trait[:7], label_average)
            best_cls = None
            max_score = -1
            for cls in classifiers:
                cls.fit(training_set[feature_columns], training_set[trait_label])
                score = cls.score(validation_set[feature_columns], validation_set[trait_label])
                if score > max_score:
                    max_score = score
                    best_cls = cls
            print str.format("{}, score: {}, cls: {}", trait[:8], max_score, str(best_cls).replace("\n", ""))
            results[task_i, trait_i] = max_score
            #print str.format("score of {} for task {} and trait {}", score, task_num, trait)
    heatmap(results, annot=True, yticklabels=[str.format("task {}", x) for x in range(len(dfs))], xticklabels=big5)

    plt.show()
    pass


if __name__ == '__main__':
    main('../pickle_data/feature_dataframes/all_features_per_task.p')