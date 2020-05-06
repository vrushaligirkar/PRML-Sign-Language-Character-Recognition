# This file describes the overall nature of the dataset

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA


## Load the data
train = pd.read_csv("data/sign_mnist_train.csv")
test = pd.read_csv("data/sign_mnist_test.csv")

x_train = train.values[:, 1:]
y_train = train.values[:, 0]

x_test = test.values[:, 1:]
y_test = test.values[:, 0]


## Samples per class
full = train.append(test)
print("Data size (samples, features+label):", np.shape(full))
n_class = len(full.label.unique())
print("Number of classes:", n_class)
count = full.label.value_counts().sort_index()
print('\nSamples per class:\n', count)


## Subplot of one sample per class
dict = {0:'A',1:'B',2:'C',3:'D',4:'E',5:'F',6:'G',7:'H',8:'I',10:'K',11:'L',12:'M',13:'N',14:'O',15:'P',16:'Q',17:'R', 18:'S',19:'T',20:'U',21:'V',22:'W',23:'X',24:'Y'}
labels = list(count.index)
fig, axes = plt.subplots(nrows=4, ncols=6)
fig.subplots_adjust(hspace=0.5)
fig.suptitle('ASL example per class (Letter)')
for ax, i in zip(axes.flatten(), labels):
    for ind, row in full.iterrows():         
        if int(row['label']) == i:
            img = row[1:].values
            img = img.reshape((28,28))
            ax.imshow(img, cmap='gray')
            ax.set(title=dict[i])
            ax.set_axis_off()
            plt.imsave('op/Letter'+str(i)+'.png', img, cmap='gray')
            break
#fig.savefig('All.png',  bbox_inches='tight')


## Features and Standard deviation plot
traindf = pd.DataFrame(x_train)
testdf = pd.DataFrame(x_test)
fulldf = pd.concat([traindf, testdf])
dfvr = fulldf.var().tolist()
fig = plt.figure(2)
plt.plot(np.sqrt(dfvr), 'o') 
plt.xlabel('Features')
plt.ylabel('Standard deviation')
plt.show()


## PCA
features = list(full.columns)
x = full.loc[:, features[1:]].values
y = full.loc[:,['label']].values
xstd = StandardScaler().fit_transform(x)
pca = PCA(n_components=3)
pca_ext = pca.fit_transform(xstd)
pca_df = pd.DataFrame(data = pca_ext, columns = ['pc1', 'pc2', 'pc3'])
fulldf = pd.DataFrame(full[['label']])
pca_df = pca_df.reset_index(drop=True)
fulldf = fulldf.reset_index(drop=True)
fdf = pd.concat([pca_df, fulldf], axis = 1)

# 2D plot
l = ['A','B','C','D','E','F','G','H','I','K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y']
groups = fdf.groupby('label')
fig = plt.figure(3)
ax = fig.add_subplot(111)
for name, group in groups:
    plt.plot(group['pc1'], group['pc2'], 'o', label=name)
plt.legend(l, bbox_to_anchor=(0, -0.4), loc='lower left', ncol=8)
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.title('PCA 2D plot')
plt.show()

# 3D plot
fig = plt.figure(4)
ax = fig.add_subplot(111, projection='3d')
for name, group in groups:
    ax.scatter(group['pc1'], group['pc2'], group['pc3'], 'o', label=name)
ax.legend(l, bbox_to_anchor=(0, -0.4), loc='lower left', ncol=8)
ax.set_xlabel('Principal Component 1')
ax.set_ylabel('Principal Component 2')
ax.set_zlabel('Principal Component 3')
ax.set_title('PCA 3D plot')
plt.show()


## LDA
lda = LDA(n_components=3)
lda_ext = lda.fit_transform(xstd, y.ravel())
lda_df = pd.DataFrame(data = lda_ext, columns = ['c1', 'c2', 'c3'])
lda_df = lda_df.reset_index(drop=True)
lda_fdf = pd.concat([lda_df, fulldf], axis = 1)

# 2D plot
groups = lda_fdf.groupby('label')
fig = plt.figure(5)
ax = fig.add_subplot(111)
for name, group in groups:
    plt.plot(group['c1'], group['c2'], 'o', label=name)
plt.legend(l, bbox_to_anchor=(0, -0.4), loc='lower left', ncol=8)
plt.xlabel('Component 1')
plt.ylabel('Component 2')
plt.title('LDA 2D plot')
plt.show()

# 3D plot
fig = plt.figure(6)
ax = fig.add_subplot(111, projection='3d')
for name, group in groups:
    ax.scatter(group['c1'], group['c2'], group['c3'], 'o', label=name)
ax.legend(l, bbox_to_anchor=(0, -0.4), loc='lower left', ncol=8)
ax.set_xlabel('Component 1')
ax.set_ylabel('Component 2')
ax.set_zlabel('Component 3')
ax.set_title('LDA 3D plot')
plt.show()
