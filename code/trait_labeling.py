from enum import Enum

# Gives labels to based on the big 5 personality traits


_cutoff = 0.5
traits = ['extraversion', 'agreeableness', 'conscientiousness', 'neuroticism', 'openness_to_experience']
def _give_labels(row):
    for label in traits:
        if row[label] > _cutoff:
            res = 1
        else:
            res = 0
        row[label + "_label"] = res

def label(df, cutoff=0.5):

    _cutoff = cutoff
    return df.apply(_give_labels, axis=1)

