
from sklearn.model_selection import train_test_split


def split_data(dataset):
    train_images, val_images, train_labels, val_labels = train_test_split(dataset.iloc[:, 1:], dataset.iloc[:, 0], test_size=0.2)

    train_images.reset_index(drop=True, inplace = True)
    val_images.reset_index(drop=True, inplace = True)
    train_labels.reset_index(drop=True, inplace = True)
    val_labels.reset_index(drop=True, inplace = True)
    return train_images, val_images, train_labels, val_labels