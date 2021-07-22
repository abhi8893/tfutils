import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import pandas as pd
from ..utils import get_dataframe_cols


# TODO: Support for `extra_metrics` as a list
def plot_learning_curve(model_or_history, extra_metric=None, include_validation=True):
    '''Plots the loss and the extra metric curve for train and val set'''

    if isinstance(model_or_history, (tf.keras.models.Sequential, tf.keras.models.Model)):
        history_dict = model_or_history.history.history

    elif isinstance(model_or_history, tf.keras.callbacks.History):
        history_dict = model_or_history.history

    elif isinstance(model_or_history, dict):
        history_dict = model_or_history

    if extra_metric is None:
        fig, axn = plt.subplots(1, 1)
        axn = np.array([axn])
    else:
        fig, axn = plt.subplots(1, 2, figsize=(12, 3))

    history_df = pd.DataFrame(history_dict)

    cols = ['loss']
    if include_validation:
        cols += ['val_loss']

    # Plot loss curve
    get_dataframe_cols(history_df, cols).plot(ax=axn[0])
    axn[0].set_title('loss', fontdict=dict(weight='bold', size=20))

    # Plot extra metric curve
    if extra_metric:
        cols = [extra_metric]
        if include_validation:
            cols += [f'val_{extra_metric}']

        get_dataframe_cols(history_df, cols).plot(ax=axn[1])
        axn[1].set_title(extra_metric, fontdict=dict(weight='bold', size=20))

    return fig, axn