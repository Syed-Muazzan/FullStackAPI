
import pandas as pd
import matplotlib.pyplot as plt


# read data
data=pd.read_excel('C:/Users/Syed Muazzan/Desktop/Model/Companies Status.xlsx')
df=pd.DataFrame(data)
df.isna().sum()
df['Risk capital'].median()
#df['Risk capital'] = pd.fillna(df['Risk capital'].median())

#Finding Null values

df.isna().sum()

#Filling Null Values using Median

col_list=['Risk capital','At-risk purchase price ($)','Sponsor promote shares',
      'Promote','Total warrants','Total shares','Total warrants / total shares']

for i in col_list:
  df[i].fillna(df[i].median(),inplace=True)

df.isna().sum()

df.columns

#Visualizing Status Values

df['STATUS'].value_counts().sort_values().plot(kind='bar')

#Printing Catagorical Values

print("Categorical Features")
df.select_dtypes('object').columns

print("Numerical Features")
df.select_dtypes('float64').columns

plt.subplot(1,2,1)
df['Sector Target'].value_counts().plot(kind='bar')
plt.tight_layout()
plt.subplot(1,2,2)
df['Sector Target'].value_counts().plot(kind='bar',color='green')
plt.tight_layout()

#Converting Catagorical Values into Numerical

df['Sector Target'].unique()

#Scikit Learn or Sklearn

import sklearn
from sklearn.preprocessing import LabelEncoder

encoder=LabelEncoder()
cat_list=['Sector Target', 'Sponsor type', 'STATUS']
for i in cat_list:
  print("Before Label Encoding:")
  print(df[i].unique())
  df[i]=encoder.fit_transform(df[i])
  print("After Label Encoding:")
  print(df[i].unique())
  print()

df.columns

X=df[['Sector Target', 'Initial Size', 'Risk capital',
       'Sponsor type', 'At-risk purchase price ($)', 'Sponsor promote shares',
       'Promote', 'Total warrants', 'Total shares',
       'Total warrants / total shares', 'Trust']]

Y=df['STATUS']

#Data Spliting

from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test =train_test_split(X,Y,test_size=0.2)

from sklearn .preprocessing import MinMaxScaler

normalize=MinMaxScaler(feature_range=(0,1))
X_train=normalize.fit_transform(X_train)
X_test=normalize.transform(X_test)

X_train.shape

X_test.shape

Y_train.shape

Y_test.shape

#Machine Learning Model
import pickle
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC

# Train the model
model = SVC(kernel='poly', C=1.0)
model.fit(X_train, Y_train)

# Make predictions
Y_predict = model.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(Y_test, Y_predict)
print(f'Model Accuracy is :{round(accuracy * 100, 2)}%')

# Save the model to a .pkl file
model_filename = 'predict.pkl'
with open(model_filename, 'wb') as file:
    pickle.dump(model, file)

print(f"Model saved as {model_filename}")