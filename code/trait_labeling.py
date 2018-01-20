from enum import Enum

# Gives labels to based on the big 5 personality traits

traits = ['extraversion', 'agreeableness', 'conscientiousness', 'neuroticism', 'openness_to_experience']
def _give_labels(row, trait, cutoff=0.5):
    #print "using cutoff {} for trait {}".format(cutoff, trait)
    if row[trait] > cutoff:
        res = 1
    else:
        res = 0
    return res


def label(df):

    for trait in traits:
        df[trait + "_label"] = df.apply(_give_labels, axis=1, trait=trait, cutoff=df[trait].median())

#%%
"""
import pickle        
path_to_pickle='../data/all_features_per_task.p'
all_features_per_task = pickle.load(open(path_to_pickle, 'rb'))

label(all_features_per_task[0], cutoff=0.5)
#%%
label_cols = [c for c in list(all_features_per_task[0].columns)if c.endswith('_label')]
label_cols
#%%

for col in label_cols:
    print col
    print sum(all_features_per_task[0][col])"""