import pandas as pd
import numpy as np 
import os 


df = pd.read_csv("data/example.csv")

target = "GPA"

X = df.drop(columns=[target])
Y = df[target]

np.random.seed(42)

n = len(df)

indices = np.random.permutation(n)

train_end = int(0.7*n)

val_end = int(0.85*n)



train_idx = indices[:train_end]

val_idx = indices[train_end:val_end]

test_idx = indices[val_end:n]


X_train = X.iloc[train_idx]
Y_train = Y.iloc[train_idx]

X_val = X.iloc[val_idx]
Y_val = Y.iloc[val_idx]

X_test = X.iloc[test_idx]
Y_test = Y.iloc[test_idx]

train_df = X_train.copy()
train_df[target] = Y_train

val_df = X_val.copy()
val_df[target] = Y_val

test_df = X_test.copy()
test_df[target] = Y_test

#export data

data_path = "data/processed"

os.makedirs(data_path, exist_ok = True)

train_df.to_csv("data/processed/train_df.csv", index=False)
val_df.to_csv("data/processed/val_df.csv", index=False)
test_df.to_csv("data/processed/test_df.csv", index=False)







