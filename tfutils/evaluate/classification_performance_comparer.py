import tensorflow as tf
from sklearn import metrics as skmetrics
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

class ClassificationPerformanceComparer:
    
    MODEL_NON_UNIQUE_MSG = 'The model names are not unique! Please make model names unique or provide a dictionary of models'
    
    def __init__(self, models, data, class_names=None, model_names=None):


        if model_names is None:
            if isinstance(models, dict):
                self.model_names = list(models.keys())
            else:
                self.model_names = [f'model_{i+1}' for i in range(len(models))]

        else:
            self.model_names = model_names
        
        if isinstance(models, (tuple, list)):
            self.models = dict(zip(self.model_names, models))
            
            # TODO: which model name is duplicated? Show in message.
            assert len(self.models) == len(models), self.MODEL_NON_UNIQUE_MSG
            
        else:
            self.models = models
        
        if (class_names is None) and isinstance(data, tf.keras.preprocessing.image.DirectoryIterator):
            self.class_names = data.class_indices
        else:
            self.class_names = class_names
            
        self.data = data
        self.true_labels = self._get_true_labels_from_data(self.data)


    @classmethod
    def from_predictions(cls, predictions, data, model_names=None, class_names=None):

        n_models = len(predictions)
        models = [None for _ in range(n_models)] # Since we don't have models

        clf_comp = cls(models, data, class_names=class_names, model_names=model_names)
        predictions = [clf_comp._make_prediction_shape_consistent(pred) for pred in predictions]

        clf_comp.prediction_probs = dict(zip(clf_comp.model_names, 
                                            [clf_comp._get_pred_proba(pred) for pred in predictions]))
        clf_comp.predictions = dict(zip(clf_comp.model_names, 
                                        [clf_comp._get_pred_from_reshaped_pred_proba(pred_proba) for pred_proba in clf_comp.prediction_probs.values()]))

        return clf_comp

    @staticmethod
    def _get_pred_proba(pred):

        if len(pred.shape) == 1:
            return pred[:, np.newaxis]
        elif pred.shape[-1] == 1:
            return pred
        elif pred.shape[-1] == 2:
            return pred[:, 1, np.newaxis]
        else:
            return pred


    @staticmethod
    def _get_pred_from_reshaped_pred_proba(pred):
        if pred.shape[-1] == 1:
            return pred.round().astype(int)
        else:
            return pred.argmax(axis=1)

        
    def _get_prediction_from_data(self, model, data):    
        if isinstance(data, tf.keras.preprocessing.image.DirectoryIterator):
            y_pred_prob = model.predict(data)
        elif isinstance(data, (tuple, list)):
            y_pred_prob = model.predict(data[0])


        y_pred_prob = self._make_prediction_shape_consistent(y_pred_prob)

        return y_pred_prob.argmax(axis=1), y_pred_prob.max(axis=1)

    @staticmethod
    def _make_prediction_shape_consistent(pred):
        pred = np.squeeze(pred)
        if len(pred.shape) == 1:
            pred = pred.reshape(-1, 1)

        return pred

    @staticmethod
    def _get_true_labels_from_data(data):
        if isinstance(data, tf.keras.preprocessing.image.DirectoryIterator):
            return data.labels.reshape(-1, 1)
        elif isinstance(data, (tuple, list)):
            return data[1].reshape(-1, 1)


    def calculate_predictions(self):

        self.predictions = {}
        self.prediction_probs = {}

        for name, model in self.models.items():
            y_pred, y_pred_prob = self._get_prediction_from_data(model, self.data)
            self.predictions[name] = y_pred
            self.prediction_probs[name] = y_pred_prob

            
    def calculate_metric_comparison_df(self):
        
        compdf = []
        for name in self.model_names:
            crdf = pd.DataFrame(skmetrics.classification_report(
                self.true_labels, self.predictions[name], 
                target_names=self.class_names, output_dict=True))
                
            crdf['model'] = name
            compdf.append(crdf)
            
            
        compdf = pd.concat(compdf)
        compdf.index.name = 'metric'
        compdf.reset_index(inplace=True)
        
        compdf_small = compdf.loc[compdf['metric'] != 'support', ['metric', 'weighted avg', 'model']].rename(columns={'weighted avg': 'value'})
        acc_df = compdf.loc[compdf['metric'] != 'support', ['model', 'accuracy']].drop_duplicates().melt(id_vars='model', var_name='metric')
        
        compdf_small = pd.concat([compdf_small, acc_df]).sort_values('model')
        
        
        self.compdf = compdf
        self.compdf_small = compdf_small
        
    
    
    def plot_metric_comparison_df(self, ax=None):
        if ax is None:
            plt.figure(figsize=(8, 4))
        sns.barplot(x='metric', y='value', hue='model', data=self.compdf_small, ax=ax)
        plt.legend(bbox_to_anchor=[1.01, 0.6])
    
    
    
