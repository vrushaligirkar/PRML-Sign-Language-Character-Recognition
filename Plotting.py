# This file includes the plotting functions for performance metrics of the models

import matplotlib.pyplot as plt
import seaborn as sns

def conf_matplot(confusion_mtx, data):
    fig = plt.figure(figsize=(10,8))
    ax1 = fig.add_subplot()
    sns.heatmap(confusion_mtx, annot=True, fmt="d", cmap="Blues");
    labels=['A','B','C','D','E','F','G','H','I','K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y']
    ax1.xaxis.set_ticklabels(labels); 
    ax1.yaxis.set_ticklabels(labels);
    plt.title('Confustion Matrix for '+data+' Data')
    plt.show()
    
def acc_loss_plot(model_desc):
    plt.subplot(211)
    plt.title('Loss')
    plt.plot(model_desc.history['loss'], label='train')
    plt.plot(model_desc.history['val_loss'], label='validation')
    plt.legend()
    # plot accuracy during training
    plt.subplot(212)
    plt.title('Accuracy')
    plt.plot(model_desc.history['accuracy'], label='train')
    plt.plot(model_desc.history['val_accuracy'], label='validation')
    plt.legend()
    plt.show()