from enum import Enum

# Gives labels to based on the big 5 personality traits


_cutoff = 0.5
traits = ['extraversion', 'agreeableness', 'conscientiousness', 'neuroticism', 'openness_to_experience']
def _give_labels(row, trait):
    if row[trait] > _cutoff:
        res = 1
    else:
        res = 0
    return res

def label(df, cutoff=0.5):

    _cutoff = cutoff
    for trait in traits:
        df[trait + "_label"] = df.apply(_give_labels, axis=1, trait=trait)

