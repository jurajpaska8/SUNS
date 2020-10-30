# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

import pandas as pd
import plotly as plotly
import matplotlib.pyplot as plt


from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score


def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press Ctrl+F8 to toggle the breakpoint.


def load_data(filepath):
    return pd.read_csv(filepath, ',')


def drop_all_na(dataframe):
    return pd.DataFrame.dropna(dataframe)


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    # LOAD
    train_data = load_data('./data/train.csv')
    test_data = load_data('./data/test.csv')

    # CLEAN NA
    train_data_clean = drop_all_na(train_data)
    test_data_clean = drop_all_na(test_data)

    # ENCODE GENRE
    train_data_encoded = train_data_clean.copy()
    test_data_encoded = test_data_clean.copy()

    unique_genre = test_data_clean.playlist_genre.unique()
    for i, val in enumerate(unique_genre):
        train_data_encoded = train_data_encoded.replace(val, i)
        test_data_encoded = test_data_encoded.replace(val, i)

    # OUTLIERS TODO
    # fig, axs = plt.subplots(5)
    # axs[0].scatter(train_data_encoded.playlist_genre, train_data_encoded.danceability)
    # axs[0].set_xlabel('Genre')
    # axs[0].set_ylabel('Danceability')
    # plt.show()

    # NORMALIZE
    sc = StandardScaler()
    train_data_normalized = train_data_encoded[['playlist_genre', 'danceability', 'energy', 'key', 'loudness', 'mode', 'speechiness', 'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo', 'duration_ms']].copy(deep=True)
    train_data_normalized[['danceability', 'energy', 'key', 'loudness', 'mode', 'speechiness', 'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo', 'duration_ms']] = sc.fit_transform(train_data_normalized[['danceability', 'energy', 'key', 'loudness', 'mode', 'speechiness', 'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo', 'duration_ms']])

    test_data_normalized = test_data_encoded[['playlist_genre', 'danceability', 'energy', 'key', 'loudness', 'mode', 'speechiness', 'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo', 'duration_ms']].copy(deep=True)
    test_data_normalized[['danceability', 'energy', 'key', 'loudness', 'mode', 'speechiness', 'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo', 'duration_ms']] = sc.fit_transform(test_data_normalized[['danceability', 'energy', 'key', 'loudness', 'mode', 'speechiness', 'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo', 'duration_ms']])

    # MLP - PERCEPTRON
    x = train_data_normalized.drop(['playlist_genre'], axis=1)
    y = train_data_normalized.playlist_genre

    # x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=1)
    # TODO validation and batch
    mlp = MLPClassifier(hidden_layer_sizes=(12, 12, 12), max_iter=500, verbose=True)
    mlp.fit(x, y)

    # accuracy on test data
    predict_test = mlp.predict(test_data_normalized.drop(['playlist_genre'], axis=1))
    cm_test = confusion_matrix(test_data_normalized.playlist_genre, predict_test)
    acc_test = accuracy_score(test_data_normalized.playlist_genre, predict_test)

    # accuracy on train data
    predict_train = mlp.predict(x)
    cm_train = confusion_matrix(y, predict_train)
    acc_train = accuracy_score(y, predict_train)

    # show training plot


    end = 1
