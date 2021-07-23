import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.preprocessing.image import img_to_array, load_img



def visualize_convolutions(model, img_prep):

    successive_outputs = [layer.output for layer in model.layers[1:]]
    visualization_model = tf.keras.models.Model(inputs = model.input, outputs = successive_outputs)
    layer_names = [layer.name for layer in model.layers[1:]]

    successive_feature_maps = visualization_model.predict(img_prep)

    for layer_name, feature_map in zip(layer_names, successive_feature_maps):
        if len(feature_map.shape) == 4:
            
            n_features = feature_map.shape[-1]  
            
            size = feature_map.shape[1]
            
            display_grid = np.zeros((size, size * n_features))
            for i in range(n_features):
            
                x = feature_map[0, :, :, i]
                x -= x.mean()
                x /= x.std()
                x *= 64
                x += 128
                x = np.clip(x, 0, 255).astype('uint8')
                
                display_grid[:, i * size : (i + 1) * size] = x
            
            scale = 20. / n_features
            plt.figure(figsize=(scale * n_features, scale))
            plt.title(layer_name)
            plt.grid(False)
            plt.imshow(display_grid, aspect='auto', cmap='viridis')