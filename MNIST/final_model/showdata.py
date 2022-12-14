
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from getdata import read_data


train_set, test_image = read_data()
train_images, val_images, train_labels, val_labels = train_test_split(train_set.iloc[:, 1:], train_set.iloc[:, 0], test_size=0.2)

train_images.reset_index(drop=True, inplace = True)
val_images.reset_index(drop=True, inplace = True)
train_labels.reset_index(drop=True, inplace = True)
val_labels.reset_index(drop=True, inplace = True)

fig, ax = plt.subplots(nrows = 5, ncols = 5, figsize = (8, 8))
fig.subplots_adjust(hspace = .3)

for i in range(5):
    for j in range(5):
        ax[i][j].axis('off')
        ax[i][j].imshow(train_images.iloc[i+(j*5), :].to_numpy().astype(np.uint8).reshape(28, 28), cmap = 'gray')
        ax[i][j].set_title(train_labels[i+(j*5)])
plt.show()

fig, ax = plt.subplots(nrows=10, ncols=10, figsize=(10, 10))
for i in range(10):
    num_i = train_images[train_labels == i]
    ax[0][i].set_title(i)
    for j in range(10):
        ax[j][i].axis('off')
        ax[j][i].imshow(num_i.iloc[j, :].to_numpy().astype(np.uint8).reshape(28,28), cmap='gray')
plt.show()

