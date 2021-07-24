import tensorflow as tf
import os
import datetime
from pathlib import Path


def create_tensorboard_callback(experiment, task=None, parent_dir=None):

    if task is None:
        task = ''
    
    if parent_dir is None:
        parent_dir = ''
    
    log_dir = os.path.join(parent_dir, task, experiment, datetime.datetime.now().strftime('%Y%m%d-%H%M%S'))
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=os.path.abspath(log_dir), profile_batch=0)
    print(f'Saving TensorBoard log files to " {log_dir}"')
    return tensorboard_callback
class StopEarly(tf.keras.callbacks.Callback):

    def __init__(self, monitor='loss', min_change=0, patience=0):
        self.monitor = monitor
        self.patience = patience
        self.min_change = min_change
        self.last_val = None
        self.num_no_improv = 0
        

    def on_epoch_end(self, epoch, logs={}):

        cur_val = logs.get(self.monitor)

        if self.last_val is None:
            self.last_val = cur_val
            print('change:', 0, 'num_no_improv:', 0, '\n')
            return None

        change = self.last_val - cur_val
        print(f'change: {change:.4f} num_no_improv:{self.num_no_improv}', '\n')

        if change < self.min_change:
            if self.num_no_improv > self.patience:
                self.model.stop_training = True
            else:
                self.num_no_improv += 1
        else:
            self.num_no_improv = 0

        self.last_val = cur_val