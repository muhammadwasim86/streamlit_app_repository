# import libraries
import streamlit as st  # For web application
import pandas as pd  # Python data analysis library, Name derived from panel data
import numpy as np  # For arithmatic operations
import matplotlib.pyplot as plt  # For plotting
import seaborn as sns  # For datasets
import sklearn  # For ml models
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

#Step 1
# app ki heading kasy deni ha with multiline string
st.write('''
### Explore different ML Models & Datasets
**Let see which one is better...**?
''')

# Step 2
# How to add a side-bar, a box on the side bar(inside box we will show diff. datasets)
# For this porpose select the st.siderbar.select box
dataset_name = st.sidebar.selectbox('Choose Dataset',
                ('Iris','Breast Cancer', 'Wine'))  

# Add another side-bar with box on the side-bar. for ML Algo's
classifier_name = st.sidebar.selectbox('Choose the Classifier',
                ('KNN','SVM', 'Random Forest')) 

# Step 3
# Import datasets; we have multiple datasets here.
# That is why we will define a function with name [get_dataset].
def get_dataset(dataset_name):
    data = None
    if dataset_name == 'Iris':
        data = datasets.load_iris()
    elif dataset_name == 'Wine':
        data = datasets.load_wine()
    else:
        data = datasets.load_breast_cancer()
    X = data.data # what is the purpose of this command. I need to understand
    y = data.target
    return X, y
# Now call the function & X, y k equal rakh lain gay.
X, y = get_dataset(dataset_name)

# Step 4
# Print the shape of dataset
st.write("Shape of dataset: ", X.shape)
st.write("Number of classes: ", len(np.unique(y))) # target variable is y and its type is class. unique values ki length lay lain gay.

# Step 5
# Define parameter of all classifiers. Here are three classifiers.
def add_parameter_ui(classifier_name): # ui means user input
    params = dict() # create an empty dictionary
    if classifier_name == 'SVM':
        C = st.sidebar.slider('C', 0.01, 10.0)
        params['C'] = C  # its the degree of correct classification
    elif classifier_name == 'KNN':
        K = st.sidebar.slider('K', 1, 15)
        params['K'] = K # no of nearest neighbours
    else:
        max_depth = st.sidebar.slider('max_depth', 2, 15)
        params['max_depth'] = max_depth # depth of each subtree in Random Forest
        n_estimators = st.sidebar.slider('n_estimators', 1, 100)
        params['n_estimators'] = n_estimators # no of trees
    return params
# Now call this function
params = add_parameter_ui(classifier_name)

# Step 6
def get_classifier(classifier_name, params): # ui means user input
    clf = None
    if classifier_name == 'SVM':
        clf = SVC(C=params['C'])
    elif classifier_name == 'KNN':
        clf = KNeighborsClassifier(n_neighbors=params['K'])
    else:
        clf = RandomForestClassifier(n_estimators=params['n_estimators'],
        max_depth = params['max_depth'], random_state= 1234)
    return clf

clf = get_classifier(classifier_name, params)

# Split data into test and train
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1234)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

# check the accuracy
acc = accuracy_score(y_test, y_pred)
st.write(f'Classifier = {classifier_name}')
st.write(f'Accuracy = ', acc)

# Step 8
# Scatter Plot based on PCA
pca = PCA(2)
X_projected = pca.fit_transform(X)

x1 = X_projected[:, 0]
x2 = X_projected[:, 1]

fig = plt.figure()
plt.scatter(x1, x2, c= y, alpha=0.8, cmap='viridis')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.colorbar()

st.pyplot(fig)