#!/usr/bin/env python
# coding: utf-8


import os.path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import random



import warnings
warnings.simplefilter('ignore')
ds = pd.read_csv("airline_passenger_satisfaction.csv")
print('columns count - ',len(ds.columns), '\n')
print('columns: ',list(ds.columns))

# Видалення дублікатів
ds = ds.drop_duplicates()

# Видалення рядків з відсутніми значеннями
ds = ds.dropna(how='all')

# Видалення непотрібних ознак
columns_to_drop = ['ID']
ds = ds.drop(columns=columns_to_drop)

# Перевірка розміру датасету після видалення
print("Розмір датасету після видалення:", ds.shape)

print("Any missing sample in training set:",ds.isnull().values.any())

for col in ds.columns:
    if ds[col].isnull().values.any():
        print("Missing data in: ", col)
        num_m = ds[col].isnull().values.sum()
        print("Number of missing values: ", num_m)
        print("Percentage of missing values: ", round(num_m/ds.shape[0]*100,2), "%")


def impute_na(df, variable, value):
    return df[variable].fillna(value)

median = ds['Arrival Delay'].median()
median
#  replace with the median
ds['Arrival Delay'] = impute_na(ds, 'Arrival Delay', median)


# function to create histogram, Q-Q plot and
from scipy import stats
def diagnostic_plots(df, variable, distribution="norm"):
    # function takes a dataframe (df) and
    # the variable of interest as arguments

    # define figure size
    plt.figure(figsize=(16, 4))

    # histogram
    plt.subplot(1, 3, 1)
    sns.histplot(df[variable], bins=30)
    plt.title('Histogram')

    # Q-Q plot
    plt.subplot(1, 3, 2)
    stats.probplot(df[variable], dist=distribution, sparams=(0.25,), plot=plt)
    plt.ylabel('Variable quantiles')

    # boxplot
    plt.subplot(1, 3, 3)
    sns.boxplot(y=df[variable])
    plt.title('Boxplot')

    plt.show()


diagnostic_plots(ds, 'Arrival Delay', distribution='expon')
diagnostic_plots(ds, 'Departure Delay', distribution='expon')
diagnostic_plots(ds, 'Flight Distance')

def remove_outliers(df, column_name, z_threshold=3):
    initial_rows = df.shape[0]
    
    outliers = df[np.abs(stats.zscore(df[column_name])) >= z_threshold]
    df = df[(np.abs(stats.zscore(df[column_name])) < z_threshold)]
    
    removed_rows = initial_rows - df.shape[0]
    removed_percentage = (removed_rows / initial_rows) * 100
    
    print("\nПочаткова кількість рядків:", initial_rows)
    print("Кількість видалених рядків:", removed_rows)
    print("Відсоток видалених рядків:", removed_percentage, "%")

    return df


ds = remove_outliers(ds, 'Arrival Delay', z_threshold=2)
ds = remove_outliers(ds, 'Departure Delay', z_threshold=2)
ds = remove_outliers(ds, 'Flight Distance', z_threshold=2)

diagnostic_plots(ds, 'Arrival Delay', distribution='expon')
diagnostic_plots(ds, 'Departure Delay', distribution='expon')
diagnostic_plots(ds, 'Flight Distance')

categorical_columns = ds.select_dtypes(include=['object']).columns
print(categorical_columns)


from sklearn.preprocessing import LabelEncoder
import pickle

# Створення екземпляра LabelEncoder
label_encoders = {}

# Кодування категоріальних стовпців
for column in categorical_columns:
    label_encoders[column] = LabelEncoder()
    ds[column] = label_encoders[column].fit_transform(ds[column])

# Збереження словника у файл
with open('label_encoders.pkl', 'wb') as f:
    pickle.dump(label_encoders, f)


# Якшо необхідно відновити словник з файлу
#with open('label_encoders.pkl', 'rb') as f:
#    label_encoders_restored = pickle.load(f)

# Використання відновленого словника
#for column, encoder in label_encoders_restored.items():
#    ds[column] = encoder.transform(ds[column])


floatColumns = ds.select_dtypes(include = ['float64']).columns
ds[floatColumns] = ds[floatColumns].astype(int)
ds.info()

ds.to_csv('preprocessed_data.csv', index=False)
display(ds.sample(15))
