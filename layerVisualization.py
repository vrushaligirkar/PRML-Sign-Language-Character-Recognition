# This file is for tnse visualization
# Reference: https://towardsdatascience.com/visualizing-intermediate-activation-in-convolutional-neural-networks-with-keras-260b36d60d0

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from keras import models
from keras.preprocessing import image


def layer_viz(model, x1):

    layer_outputs = [layer.output for layer in model.layers[:15]] 
    activation_model = models.Model(inputs=model.input, outputs=layer_outputs)
    
    pix = np.array(x1[0,:]).reshape((28,28)) 
    img_tensor = image.img_to_array(pix) 
    img_tensor = np.expand_dims(img_tensor, axis=0)
    img_tensor /= 255. 
    activations = activation_model.predict(img_tensor)
    
    layer_names = []
    for layer in model.layers[:10]:
        layer_names.append(layer.name) 
        
    images_per_row = 16
    
    for layer_name, layer_activation in zip(layer_names, activations):
        n_features = layer_activation.shape[-1] 
        size = layer_activation.shape[1] 
        n_cols = n_features // images_per_row 
        display_grid = np.zeros((size * n_cols, images_per_row * size))
        for col in range(n_cols): 
            for row in range(images_per_row):
                channel_image = layer_activation[0,:, :, col * images_per_row + row]
                channel_image -= channel_image.mean() 
                channel_image /= channel_image.std()
                channel_image *= 64
                channel_image += 128
                channel_image = np.clip(channel_image, 0, 255).astype('uint8')
                display_grid[col * size : (col + 1) * size, row * size : (row + 1) * size] = channel_image
        scale = 1. / size
        plt.figure(figsize=(scale * display_grid.shape[1], scale * display_grid.shape[0]))
        plt.title(layer_name)
        plt.grid(False)
        plt.imshow(display_grid, aspect='auto', cmap='gray')
    
