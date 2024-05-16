#!/usr/bin/env python
# coding: utf-8

# In[79]:


import os.path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import random


# In[80]:


import warnings
warnings.simplefilter('ignore')


# In[81]:


print(os.path.exists("airline_passenger_satisfaction.csv"))


# In[82]:


ds = pd.read_csv("airline_passenger_satisfaction.csv")


# In[83]:


print('columns count - ',len(ds.columns), '\n')
print('columns: ',list(ds.columns))


# In[84]:


print('Samples count: ',ds.shape[0])


# In[85]:


display(ds.head(5))


# In[86]:


print("Any missing sample in training set:",ds.isnull().values.any())


# In[169]:


for col in ds.columns:
    if ds[col].isnull().values.any():
        print("Missing data in: ", col)
        num_m = ds[col].isnull().values.sum()
        print("Number of missing values: ", num_m)
        print("Percentage of missing values: ", round(num_m/ds.shape[0]*100,2), "%")


# In[88]:


ds.nunique()


# In[89]:


ds.describe()


# In[90]:


ds.info()


# In[91]:


import matplotlib.pyplot as plt
import seaborn as sns

# Виберемо лише числові дані з DataFrame
numerical_columns = ds.select_dtypes(include=['int64', 'float64'])

plt.figure(figsize=(15, 12))
correlation_matrix = numerical_columns.corr()
sns.heatmap(
    correlation_matrix,
    vmax=1,
    square=True,
    annot=True,
    fmt='.2f',
    cmap='GnBu',
    cbar_kws={"shrink": .5},
    robust=True)
plt.title('Correlation Matrix of features', fontsize=8)
plt.show()


# In[196]:


# Виберемо лише числові дані з DataFrame
numerical_columns = ds.select_dtypes(include=['int64', 'float64'])

# Створимо матрицю кореляції
correlation_matrix = numerical_columns.corr()

# Виберемо тільки значення кореляції, що перевищують 0.5 або -0.5
high_correlation = correlation_matrix[((correlation_matrix > 0.5) | (correlation_matrix < -0.5)) & (correlation_matrix != 1.0)].stack()

# Виведемо результат
print(high_correlation)


# In[218]:


columns = ['Age', 'Flight Distance', 'Departure Delay', 'Arrival Delay']

# Розміщення графіків у вигляді двох рядків по чотири графіки
plt.figure(figsize=(12, 8))

# Графіки гістограм
for i, col in enumerate(columns):
    plt.subplot(2, 4, i + 1)
    sns.histplot(data=ds, x=col, color='lightblue', bins=20)
    plt.title(f'{col} Histogram')
    plt.xlabel('')
    plt.ylabel('')
    plt.tight_layout()

# Графіки з вусами
for i, col in enumerate(columns):
    plt.subplot(2, 4, i + 5)
    sns.boxplot(data=ds, x=col, color='lightgreen')
    plt.title(f'{col} Boxplot')
    plt.xlabel('')
    plt.ylabel('')
    plt.tight_layout()

plt.show()


# In[201]:


def plot_pie_chart(data, column_name, colors=['#66c2a5', '#8da0cb','#fc8d62', '#e78ac3', '#a6d854', '#ffd92f']):
    plt.figure(figsize=(8, 8))
    counts = data[column_name].value_counts()
    explode = [0 if len(counts) > 2 and i == max(counts) else 0 for i in counts]  # Розсування сегментів
    plt.pie(counts, labels=counts.index, startangle=90, colors=colors, shadow=True, explode=explode, autopct='%1.1f%%', pctdistance=0.85)
    # Додаємо контур для кожного сегменту
    for w in plt.gca().patches:
        w.set_linewidth(0.5)
        w.set_edgecolor('black')
    plt.title(f'Pie Chart of {column_name}')
    plt.show()


# In[219]:


plot_pie_chart(ds, 'Satisfaction')
plot_pie_chart(ds, 'Gender',colors=['#fc8d62','#a6d854'])
plot_pie_chart(ds, 'Customer Type',colors=['#ffd92f','#a6d854'])
plot_pie_chart(ds, 'Type of Travel',colors=['#e78ac3','#8da0cb'])
plot_pie_chart(ds, 'Class',colors=['#fc8d62','#a6d854','#ffd92f'])


# In[221]:


cat_column = ds.select_dtypes(include = object).columns.tolist()
cat_column.remove('Gender')


# In[227]:


def create_kdeplot(x_axis, columns):
    # 1- set figure size
    plt.figure(figsize=(15, 10))

    # Define pastel colors
    pastel_colors = ['#FF99CC', '#a6d854','#FFCC99','#FF9999', '#99FF99']

    # 2- loop over categorical column list to plot columns
    for index, col in enumerate(columns):
        plt.subplot((len(columns) + 1) // 2, 2, index + 1) # create sub-plot
        sns.kdeplot(x=x_axis, hue=col, data=ds, fill=True, palette=pastel_colors)

        plt.title(col) # set title to each plot
        plt.xlabel("") # replace x label with empty string
        plt.ylabel("") # replace y label with empty string
        plt.yticks([]) # Remove y-axis label
 
    # 3- set layout between two plots
    plt.tight_layout(pad=2)

    plt.show()


# In[228]:


create_kdeplot("Age", cat_column) 


# In[229]:


create_kdeplot("Flight Distance", cat_column)


# In[132]:


# Обчислюємо середні значення кожного сервісу
service_means = ds[['Departure and Arrival Time Convenience', 'Ease of Online Booking', 'Check-in Service', 'Online Boarding',
                     'Gate Location', 'On-board Service', 'Seat Comfort', 'Leg Room Service', 'Cleanliness', 'Food and Drink',
                     'In-flight Service', 'In-flight Wifi Service', 'In-flight Entertainment', 'Baggage Handling']].mean().sort_values()

# Побудова графіку
plt.figure(figsize=(12, 6))
sns.barplot(x=service_means.index, y=service_means.values)
plt.xticks(rotation=45, ha='right')  # ha='right' для правильного вирівнювання тексту
plt.xlabel('Service')
plt.ylabel('Average Rating')
plt.title('Average Ratings for Each Service')
plt.show()


# In[230]:


def plot_services(column_name):
    # Обчислюємо середні значення кожного сервісу
    service_means = ds[['Departure and Arrival Time Convenience', 'Ease of Online Booking', 'Check-in Service', 'Online Boarding',
                         'Gate Location', 'On-board Service', 'Seat Comfort', 'Leg Room Service', 'Cleanliness', 'Food and Drink',
                         'In-flight Service', 'In-flight Wifi Service', 'In-flight Entertainment', 'Baggage Handling']].mean()

    services_columns = service_means.index # колонки для графіку

    services = pd.DataFrame(ds.groupby(column_name)[services_columns].mean().round(1))
    services = services.T  # Транспонуємо датафрейм для вертикального вигляду

    # Створюємо графік
    plt.figure(figsize=(15, 4))  # задаємо розмір графіку
    sns.lineplot(data=services, markers=True, palette='pastel')  # встановлюємо пастельні кольори

    # Встановлюємо нове положення міток на вісі x
    plt.xticks(rotation=45, ha='right')  # ha='right' для правильного вирівнювання тексту

    # Встановлюємо заголовок
    plt.title(f'Service Rating With {column_name}')

    # Показуємо графік
    plt.show()


# In[231]:


plot_services('Type of Travel')
plot_services('Gender')
plot_services('Class')
plot_services('Satisfaction')
plot_services('Customer Type')
plot_services('Age Group')


# In[145]:


# Розділити вік на 5 груп
ds['Age Group'] = pd.cut(ds['Age'], bins=5, labels=['0-20', '20-40', '40-60', '60-80', '80-100'])


# In[ ]:




