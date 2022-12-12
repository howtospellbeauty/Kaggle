import pandas as pd
import numpy as np

train_df = pd.read_csv('./input/train.csv')
test_df = pd.read_csv('./input/test.csv')
combine = [train_df, test_df]

guess_ages = np.zeros((2,3))

for dataset in combine:
    for i in range(0,2)