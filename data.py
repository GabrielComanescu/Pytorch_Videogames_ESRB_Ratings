import pandas as pd
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
import torch
import numpy as np
import matplotlib.pyplot as plt


def DistributionPlotter(dataFrame, featureName, setName):
    uniqueVals = set(dataFrame[featureName])
    countArr =  []
    
    for mem in uniqueVals:
        countArr.append(len(dataFrame[dataFrame[featureName] == mem]))
    
    plt.figure(figsize=(17, 5))
    plt.subplot(1,2,1)
    plt.bar(list(uniqueVals), countArr, color = 'orange')
    plt.title('Bar graph: ' + str(setName))
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.pie(countArr, labels = list(uniqueVals),autopct='%1.2f%%')
    plt.title('Pie chart:  ' + str(setName))
    plt.show()


def prepare_data():
    df = pd.read_csv('Data/Video_games_esrb_rating.csv')
    df_test = pd.read_csv('Data/test_esrb.csv')
    df['target'] = df['esrb_rating'].astype('category').cat.codes
    df_test['target'] = df_test['esrb_rating'].astype('category').cat.codes
    
    # DistributionPlotter(df_test, 'esrb_rating', 'Rating distribution [Test set]')

    X_train = df.drop(['target', 'esrb_rating', 'title'], axis=1).values
    y_train = df['target'].values


    X_test = df_test.drop(['target', 'esrb_rating', 'title'], axis=1).values
    y_test = df_test['target'].values

    X_train = torch.FloatTensor(np.copy(X_train))
    X_test = torch.FloatTensor(np.copy(X_test))
    y_train = torch.LongTensor(np.copy(y_train))
    y_test = torch.LongTensor(np.copy(y_test))

    train_data = TensorDataset(X_train, y_train)
    test_data = TensorDataset(X_test, y_test)

    return train_data, test_data


def get_data():
  torch.manual_seed = 42

  train_data, test_data = prepare_data()

  train_loader = DataLoader(train_data, batch_size=200,shuffle=True)
  test_loader = DataLoader(test_data, batch_size=100,shuffle=False)

  return train_loader, test_loader