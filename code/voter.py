import pandas as pd
import numpy as np

class Voter:
    
    def __init__(self, X_full, models, feature_columns, weights=None):
        """
        X_full: full feature dataframe
        models: list of trained models (fixed trait, models for each task)
        feature_columns: list of feature columns to use, order corresponds to order of models
        """
        
        self.X_full = X_full
        if not weights is None:
            self.weights = [1]*len(models)
        else:
            self.weights = weights
        self.feature_columns = feature_columns
        self.models = models
        
    
    def predict(self, observation, method='majority_voting'):
        
        if method=='majority_voting':
            return self._simple_majority_voting(observation)
        
    def _simple_majority_voting(self, observation):
        
        #list of predictions per task
        predictions = [model.predict(observation[features]) for model, features in zip(self.models, self.feature_columns)]
        
        counts_for_1 = [] #[subject0_counts, subject1_counts, ...]
        for subject in range(len(observation)):            
            counts_for_1.append(sum([task[subject] for task in predictions]))
            
        voted_predictions = []
        for count in counts_for_1:
            if count>len(self.models):
                voted_predictions.append(1)
            else:
                voted_predictions.append(0)
        return voted_predictions
    
    def score(self, observation, y, method='majority_voting'):
        preds = self._simple_majority_voting(observation)
        
        acc = 0
        for pred, real in zip(preds, y):
            if pred==real:
               acc+=1
        
        return float(acc)/len(preds)

#%%
import pickle
path_to_pickle='../pickle_data/feature_dataframes/all_features_restructured.p'
X_all_tasks = pickle.load(open(path_to_pickle, 'rb'))
#voter trait 0
big5 = ['extraversion', 'agreeableness', 'conscientiousness', 'neuroticism', 'openness_to_experience']
big5_labels = ['extraversion_label', 'agreeableness_label', 'conscientiousness_label', 'neuroticism_label', 'openness_to_experience_label']

#%%
import classification_gp
fitted_models = classification_gp.main(path_to_pickle)
#fitted_models
#%%
#voter trait 0
trait_no = 0
trait = 'trait_'+str(0)
tasks = ['task_'+str(i) for i in range(6)]
models = []
feature_columns = []
unseen_data = []
for task in tasks:
    print task
    models.append(fitted_models.get(trait).get(task).get('model'))
    feature_columns.append(fitted_models.get(trait).get(task).get('feature_columns'))
    unseen_data.append(fitted_models.get(trait).get(task).get('data_unseen'))
models
#%%
feature_columns
#%%
voter = Voter(X_all_tasks, models, feature_columns)    
#%%
no=0
voter.score(unseen_data[no][feature_columns[no]], unseen_data[no][big5_labels[trait_no]])

#%%
list(unseen_data[no].columns)