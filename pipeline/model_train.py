from sklearn.ensemble import RandomForestClassifier
import os
import pandas as pd
import pickle

def preprocess_data(file_path, label_encoders_path):
    # Завантаження датасету
    ds = pd.read_csv(file_path)

    # Заповнення пропущених значень медіаною
    median_arrival_delay = ds['Arrival Delay'].median()
    ds['Arrival Delay'] = ds['Arrival Delay'].fillna(median_arrival_delay)

    median_departure_delay = ds['Departure Delay'].median()
    ds['Departure Delay'] = ds['Departure Delay'].fillna(median_departure_delay)

    # Відновлення словника з файлу
    with open(label_encoders_path, 'rb') as f:
        label_encoders_restored = pickle.load(f)

    # Кодування категоріальних ознак за допомогою відновленого словника
    for column, encoder in label_encoders_restored.items():
        ds[column] = encoder.transform(ds[column])

    return ds


def train_model(file_name):
    if not os.path.isfile('../data/' + file_name):
        print(f"File '{file_name}' not found.")
        return

    data = pd.read_csv('../data/' + file_name)

    X = data.drop(columns=['Satisfaction'])
    Y = data['Satisfaction']

    best_gbr = RandomForestClassifier(max_depth= 20, min_samples_leaf= 1, min_samples_split= 2, n_estimators= 100)
    best_gbr.fit(X, Y)

    with open(f'../model/RandomForestClassifier.pkl', 'wb') as f:
        pickle.dump(best_gbr, f)


file_path = "../data/airline_passenger_satisfaction.csv"
label_encoders_path = '../data/label_encoders.pkl'
preprocessed_ds = preprocess_data(file_path, label_encoders_path)
preprocessed_ds.to_csv('../data/preprocessed_train_data.csv', index=False)
train_model('train.csv')
