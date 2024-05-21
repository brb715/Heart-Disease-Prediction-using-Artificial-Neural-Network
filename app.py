# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.16.2
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

import streamlit as st
st.title('Heart Disease Prediction using :blue[Artificial Neural Network]')

# ## Importing the libraries

import numpy as np
import pandas as pd
import tensorflow as tf

# # Data Preprocessing


dataset = pd.read_csv('heart.csv')

st.header('Dataset', divider='rainbow')
with st.container(height=200):
    st.table(dataset)
    

X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values


st.header('Exploratory Data Analysis', divider ='rainbow')

import seaborn as sns
import matplotlib.pyplot as plt

st.subheader('Histogram')

option = st.selectbox(
    "Which feature histogram you want to see?",
    ('', 'age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach',
       'exang', 'oldpeak', 'slope', 'ca', 'thal', 'target'))
if option != '':
    fig = plt.figure()
    sns.histplot(data = dataset, x = dataset[option])
    st.pyplot(fig)



st.subheader('Features having Outliers')

option = st.selectbox(
    "Which feature having outliers you want to see?",
    ('', 'trestbps', 'chol', 'thalach', 'oldpeak', 'fbs', 'ca', 'thal'))
if option != '':
    fig = plt.figure()
    sns.boxplot(data = dataset, x = dataset[option])
    st.pyplot(fig)



st.subheader('Outlier Treatment')

def wisker(col):
    q1, q3 = np.percentile(col, [25, 75])
    iqr = q3 - q1
    lw = q1 - 1.5 * iqr
    uw = q3 + 1.5 * iqr
    return lw, uw

for i in ['trestbps', 'chol', 'thalach', 'oldpeak']:
    lw, uw = wisker(dataset[i])
    dataset[i] = np.where(dataset[i] < lw, lw, dataset[i])
    dataset[i] = np.where(dataset[i] > uw, uw, dataset[i])

option = st.selectbox(
    "Which feature you want to see after outlier treatment?",
    ('', 'trestbps', 'chol', 'thalach', 'oldpeak'))
if option != '':
    fig = plt.figure()
    sns.boxplot(data = dataset, x = dataset[option])
    st.pyplot(fig)
    

st.subheader('Scatter Plot')

option = st.selectbox(
    "Which feature's reltionship with the target you want to see?",
    ('', 'age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach',
       'exang', 'oldpeak', 'slope', 'ca', 'thal'))
if option != '':
    fig = plt.figure()
    sns.scatterplot(data = dataset, x = dataset[option], y = 'target')
    st.pyplot(fig)


st.subheader('Correlation Matrix')

s = dataset.iloc[:-1, :-1].corr()
fig, ax = plt.subplots(figsize = (10, 10))
sns.heatmap(s, annot = True, ax=ax)
st.write(fig)




# # Splitting the dataset into the Training set and Test set

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)



# # Feature Scaling

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)



# # Building the ANN

## ANN1
def fir(P):
    st.subheader('ANN 1')
    
    import time
    with st.spinner('Model Training...'):
    
        ann1 = tf.keras.models.Sequential()
        
        # ## Adding the input layer and the first hidden layer
        
        ann1.add(tf.keras.layers.Dense(units = 6, activation ='relu'))
        
        # ## Adding the second hidden layer
        
        ann1.add(tf.keras.layers.Dense(units = 6, activation ='relu'))
        
        # ## Adding the output layer
        
        ann1.add(tf.keras.layers.Dense(units = 1, activation ='sigmoid'))
        
        # ## Compiling the ANN
        
        ann1.compile(loss = 'binary_crossentropy', metrics = ['accuracy'], optimizer = 'adam')
        
        # ## Training the ANN on the Training set
        
        history1 = ann1.fit(X_train, y_train, validation_data=(X_test, y_test), batch_size = 32, epochs = 100)
    st.success('Training Done!')

    with st.spinner('Model Testing...'):   
        y_pred1 = ann1.predict(X_test)
        y_pred1 = (y_pred1 > 0.5)
        
        from sklearn.metrics import accuracy_score
        score = accuracy_score(y_test, y_pred1)
        time.sleep(5)
    st.success('Testing Done!')
    container = st.container(border=True)
    container.write(f'Accuracy = {score*100}')


    fig = plt.figure()
    plt.plot(history1.history['accuracy'])
    plt.plot(history1.history['val_accuracy'])
    plt.title('Model Accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'])
    st.pyplot(fig)
    
    fig = plt.figure()
    plt.plot(history1.history['loss'])
    plt.plot(history1.history['val_loss'])
    plt.title('Model Loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'])
    st.pyplot(fig)

    pred = ann1.predict(sc.transform([[P['age'], P['sex'], P['cp'], P['trestbps'], P['chol'], P['fbs'], P['restecg'], P['thalach'], P['exang'], P['oldpeak'], P['slope'], P['ca'], P['thal']]]))
    return pred

# ## ANN 2
def sec(P):
    st.subheader('ANN 2')
    
    import time
    with st.spinner('Model Training...'):
        ann2 = tf.keras.models.Sequential()
        ann2.add(tf.keras.layers.Dense(units = 32, activation ='relu'))
        ann2.add(tf.keras.layers.Dense(units = 1, activation ='sigmoid'))
        ann2.compile(loss = 'binary_crossentropy', metrics = ['accuracy'], optimizer = 'adam')
        history2 = ann2.fit(X_train, y_train, validation_data=(X_test, y_test), batch_size = 32, epochs = 100)
    st.success('Training Done!')
        
       
        
    with st.spinner('Model Testing...'): 
        y_pred2 = ann2.predict(X_test)
        y_pred2 = (y_pred2 > 0.5)

        from sklearn.metrics import accuracy_score
        score = accuracy_score(y_test, y_pred2)
        time.sleep(5)
    st.success('Testing Done!')
    container = st.container(border=True)
    container.write(f'Accuracy = {score*100}')

    
    fig = plt.figure()
    plt.plot(history2.history['accuracy'])
    plt.plot(history2.history['val_accuracy'])
    plt.title('Model Accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'])
    st.pyplot(fig)
    
    fig = plt.figure()
    plt.plot(history2.history['loss'])
    plt.plot(history2.history['val_loss'])
    plt.title('Model Loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'])
    st.pyplot(fig)

    pred = ann2.predict(sc.transform([[P['age'], P['sex'], P['cp'], P['trestbps'], P['chol'], P['fbs'], P['restecg'], P['thalach'], P['exang'], P['oldpeak'], P['slope'], P['ca'], P['thal']]]))
    return pred




# ## ANN 3
def thr(P):
    st.subheader('ANN 3')

    import time
    with st.spinner('Model Training...'):
        ann3 = tf.keras.models.Sequential()
        ann3.add(tf.keras.layers.Dense(units = 16, activation ='relu'))
        ann3.add(tf.keras.layers.Dropout(0, 0.3))
        ann3.add(tf.keras.layers.Dense(units = 4, activation ='relu'))
        ann3.add(tf.keras.layers.Dropout(0, 0.3))
        ann3.add(tf.keras.layers.Dense(units = 1, activation ='sigmoid'))
        ann3.compile(loss = 'binary_crossentropy', metrics = ['accuracy'], optimizer = 'adam')
        history3 = ann3.fit(X_train, y_train, validation_data=(X_test, y_test), batch_size = 32, epochs = 100)
    st.success('Training Done!')
        
       
        
    with st.spinner('Model Testing...'): 
        y_pred3 = ann3.predict(X_test)
        y_pred3 = (y_pred3 > 0.5)

        from sklearn.metrics import accuracy_score
        score = accuracy_score(y_test, y_pred3)
        time.sleep(5)
    st.success('Testing Done!')
    container = st.container(border=True)
    container.write(f'Accuracy = {score*100}')


    fig = plt.figure()
    plt.plot(history3.history['accuracy'])
    plt.plot(history3.history['val_accuracy'])
    plt.title('Model Accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'])
    st.pyplot(fig)
    
    fig = plt.figure()
    plt.plot(history3.history['loss'])
    plt.plot(history3.history['val_loss'])
    plt.title('Model Loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'])
    st.pyplot(fig)
    
    pred = ann3.predict(sc.transform([[P['age'], P['sex'], P['cp'], P['trestbps'], P['chol'], P['fbs'], P['restecg'], P['thalach'], P['exang'], P['oldpeak'], P['slope'], P['ca'], P['thal']]]))
    return pred



st.header('Predicting the result of a single observation', divider = 'rainbow')
P = []
with st.container(border=True):
    age = st.number_input("Age", value = None, placeholder ='Enter your age in years')
    sex = st.number_input("Sex", value = None, placeholder = '1 = male, 0 = female')
    cp = st.number_input("Chest Pain", value = None, placeholder = '1 = typical angina, 2 = atypical angina, 3 = non-anginal pain, 4 = asymptomatic')
    trestbps = st.number_input("Resting Blood Pressure", value = None, placeholder = "Resting Blood Pressure in mm")
    chol = st.number_input("Serum Cholestoral", value = None, placeholder = 'Serum Cholestoral in mg/dl')
    fbs = st.number_input("Fasting Blood Sugar > 120 mg/dl", value = None, placeholder = '1 = true, 0 = false')
    restecg = st.number_input("Resting Electrocardiographic", value = None, placeholder = "0 = normal, 1 = having ST-T wave abnormality, 2 =  left ventricular hypertrophy")
    thalach = st.number_input("Maximum Heart Rate Achieved", value = None, placeholder = "Maximum Heart Rate Achieved")
    exang = st.number_input("Exercise Induced Angina", value = None, placeholder = '1 = yes, 0 = no')
    oldpeak = st.number_input("ST depression induced by exercise relative to rest", value = None, placeholder = "ST depression induced by exercise relative to rest")
    slope = st.number_input("Slope of the peak exercise ST segmen", value = None, placeholder = '1 = upsloping, 2 = flat, 3 = downsloping')
    ca = st.number_input("Coronary Arteries", value = None, placeholder = 'Number of major vessels (0-3) colored by Flourosopy')
    thal = st.number_input('Thalassemia', value = None ,placeholder = "3 = normal, 6 = fixed defect, 7 = reversable defect")
    P = {'age': age, 'sex': sex, 'cp': cp, 'trestbps': trestbps, 'chol': chol, 'fbs': fbs, 'restecg': restecg, 'thalach': thalach, 'exang': exang, 'oldpeak': oldpeak, 'slope': slope, 'ca': ca, 'thal': thal}

model = st.selectbox(
        "Which ANN model you want to use?",
        ('', 'ANN1', 'ANN2', 'ANN3'))


if model != '':
    if model == 'ANN1':
        pred = fir(P)
    elif model == 'ANN2':
        pred = sec(P)
    else:
        pred = thr(P)
    st.subheader('Prediction:')
    with st.container(border=True):
        if pred>0.5:
            st.write('You have a heart disease')
        else:
            st.write("You don't have a heart disease")
