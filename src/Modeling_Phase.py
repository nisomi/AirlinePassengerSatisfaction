#!/usr/bin/env python
# coding: utf-8

# In[35]:


import os.path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import random
from sklearn.model_selection import train_test_split


# In[36]:


import warnings
warnings.simplefilter('ignore')


# In[37]:


ds = pd.read_csv("preprocessed_data.csv")


# In[38]:


ds.shape


# In[39]:


from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.linear_model import RidgeClassifier


from sklearn import metrics


# In[40]:


X = ds.drop(columns=['Satisfaction'])
y = ds['Satisfaction']


# In[65]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=91)


# In[66]:


# Define a pastel color palette
pastel_colors = ['#A5DD9B', '#E1AFD1']

for i, y_set in enumerate([y_train, y_test]):
    plt.rcParams["figure.figsize"] = (80, 80)
    plt.subplot(6, 3, i + 1)
    plt.rcParams.update({'font.size': 30})
    labels = y_set.astype('str').unique().tolist()
    slices = [y_set.astype('str').value_counts()[i] for i in labels]
    plt.pie(slices, labels=labels, explode=[0.01 for i in range(len(labels))], colors=pastel_colors, wedgeprops={'edgecolor':'black'}, shadow=True, autopct='%1.1f%%')
    plt.tight_layout()


# In[43]:


models = {
    'Logistic Regression': LogisticRegression(),
    'GaussianNB': GaussianNB(),
    'SGDClassifier': SGDClassifier(),
    'DecisionTreeClassifier': DecisionTreeClassifier(),
    'RandomForestClassifier': RandomForestClassifier(),
    'GradientBoostingClassifier': GradientBoostingClassifier(),
    'XGBClassifier': XGBClassifier(),
    'AdaBoostClassifier': AdaBoostClassifier(),
    'BaggingClassifier': BaggingClassifier(),
    'RidgeClassifier': RidgeClassifier()
}


# In[44]:


from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

results = []

for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')
    results.append({'Model': name, 'Accuracy': accuracy, 'Precision': precision, 'Recall': recall, 'F1 Score': f1})
    
    print(f"Results for {name}:")
    print(f"Accuracy: {accuracy:.3f}, Precision: {precision:.3f}, Recall: {recall:.3f}, F1 Score: {f1:.3f}")
    print("-" * 50)

results_df = pd.DataFrame(results)
results_df = results_df.sort_values(by='Accuracy', ascending=False)
print("\nFinal Results:")
print(results_df)


# In[45]:


importances = models['RandomForestClassifier'].feature_importances_
indices = np.argsort(importances)
plt.figure(figsize=(12, 10)) 
plt.title('Feature Importances', fontsize=12)
plt.barh(range(len(indices)), importances[indices], color='#9370DB', edgecolor='black', align='center')  
plt.yticks(range(len(indices)), [X.columns[i] for i in indices], fontsize=10)
plt.xticks(fontsize=10)
plt.xlabel('Relative Importance', fontsize=8)
plt.show()


# In[46]:


from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Параметри, які бажаємо перевірити
param_grid = {
    'n_estimators': [10, 50, 100],
    'max_depth': [None, 10, 20],
    'min_samples_split': [1, 2, 5],
    'min_samples_leaf': [1, 2, 5]
}

# Створюємо класифікатор RandomForestClassifier
rf = RandomForestClassifier()

# Ініціалізуємо GridSearchCV
grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=2, n_jobs=-1, verbose=2)

# Проводимо пошук по сітці
grid_search.fit(X_train, y_train)

# Отримуємо найкращі параметри та виводимо їх
best_params = grid_search.best_params_
print("Найкращі параметри:", best_params)

# Оцінюємо точність на тестових даних з найкращими параметрами
best_rf = RandomForestClassifier(**best_params)
best_rf.fit(X_train, y_train)
y_pred = best_rf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("Точність на тестових даних з найкращими параметрами:", round(accuracy,3))


# In[69]:


rf_classifier = RandomForestClassifier(max_depth= 20, min_samples_leaf= 1, min_samples_split= 2, n_estimators= 100)
rf_classifier.fit(X_train, y_train)
y_pred = rf_classifier.predict(X_test)
print(metrics.classification_report(y_test, y_pred))


# In[90]:


y.value_counts()


# In[91]:


from imblearn.over_sampling import RandomOverSampler

ros = RandomOverSampler(
    sampling_strategy='auto', # samples only the minority class
    random_state=0,  # for reproducibility
)  

X_res, Y_res = ros.fit_resample(X, y)


# In[92]:


Y_res.value_counts()


# In[93]:


X_train, X_test, y_train, y_test = train_test_split(X_res, Y_res, test_size=0.3, random_state=19)


# In[94]:


# Define a pastel color palette
pastel_colors = ['#A5DD9B', '#E1AFD1']

for i, y_set in enumerate([y_train, y_test]):
    plt.rcParams["figure.figsize"] = (80, 80)
    plt.subplot(6, 3, i + 1)
    plt.rcParams.update({'font.size': 30})
    labels = y_set.astype('str').unique().tolist()
    slices = [y_set.astype('str').value_counts()[i] for i in labels]
    plt.pie(slices, labels=labels, explode=[0.01 for i in range(len(labels))], colors=pastel_colors, wedgeprops={'edgecolor':'black'}, shadow=True, autopct='%1.1f%%')
    plt.tight_layout()


# In[95]:


rf_classifier = RandomForestClassifier()
rf_classifier.fit(X_train, y_train)
y_pred = rf_classifier.predict(X_test)
print(metrics.classification_report(y_test, y_pred))


# In[97]:


import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import metrics

plt.rcParams.update({'font.size': 16})

fig, ax = plt.subplots(figsize=(7, 6))
confusion_matrix = metrics.confusion_matrix(y_test, y_pred)
ax = sns.heatmap(confusion_matrix,
                 annot=True,
                 cbar=True,
                 fmt='g',
                 cmap='viridis')  # Змінено параметр cmap тут
plt.xlabel('True Label')
plt.ylabel('Predicted Label')
plt.title('Confusion-Matrix')
plt.tight_layout()
plt.show()


# In[98]:


from sklearn.metrics import roc_curve, roc_auc_score

# Отримання ймовірностей для позитивного класу
y_pred_prob = rf_classifier.predict_proba(X_test)[:, 1]

# Обчислення ROC кривої
fpr, tpr, _ = roc_curve(y_test, y_pred_prob)

# Обчислення AUC
auc = roc_auc_score(y_test, y_pred_prob)

# Виведення результатів
print(f"ROC AUC: {auc}")


# In[99]:


import matplotlib.pyplot as plt

# Встановлення розміру шрифту
plt.rcParams.update({'font.size': 14})

# Побудова ROC кривої
plt.figure(figsize=(8, 8))
plt.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % auc)
plt.plot([0, 1], [0, 1], 'k--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc="lower right")
plt.show()


# In[ ]:




