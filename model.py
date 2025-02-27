import pickle
import numpy as np
# import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import sklearn 
from sklearn.linear_model import LogisticRegression
data=pd.read_csv('data.csv')
selected_features = [
    "concave points_worst", "perimeter_worst", "concave points_mean",
    "radius_worst", "perimeter_mean", "area_worst",
    "radius_mean", "area_mean", "concavity_mean", "concavity_worst"
]
X = data[selected_features]  # Independent variables
Y = data["diagnosis"]  # Target variable

from sklearn.preprocessing import LabelEncoder
label_encoder = LabelEncoder()
Y = label_encoder.fit_transform(Y) 


from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)


from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test) 

from sklearn.linear_model import LogisticRegression
log_model = LogisticRegression()
log_model.fit(X_train, Y_train)


pickle.dump(log_model, open("model.pkl", "wb"))

