import pandas as pd
import pickle


def test_model(file_name):
    data = pd.read_csv('../data/' + file_name)

    X = data.drop(columns=['Satisfaction'])
    Y = data['Satisfaction']

    with open(f'../model/RandomForestClassifier.pkl', 'rb') as f:
        model = pickle.load(f)

    predictions = model.predict(X)

    pd.DataFrame(predictions).to_csv('../data/predictions.csv', index=False)

    accuracy = (predictions == Y).mean()
    print(f'Accuracy: {accuracy}')


test_model('test.csv')
