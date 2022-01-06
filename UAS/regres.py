#library
import numpy as np
import joblib
from sklearn.linear_model import LinearRegression
#dataset 
X= np.array([400,350,1000,200,560,430,1500,780,670,480]).reshape((-1, 1))
Y= np.array([1500,750,1760,500,800,900,890,1600,2000,1970])
#call model regression
model = LinearRegression().fit(X,Y)
#save model
filename = 'model.sav'
joblib.dump(model, filename)
#load model
loaded_model = joblib.load(filename)
#prediction model
loaded_model.predict(20)
