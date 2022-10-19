import warnings
warnings.filterwarnings('ignore')
import pandas as pd
import numpy as np
from sklearn.feature_selection import chi2 
from sklearn.feature_selection import SelectKBest 
from sklearn.ensemble import ExtraTreesClassifier
import imblearn
from sklearn.model_selection import train_test_split # train-test split
from sklearn.metrics import confusion_matrix, classification_report # classification metrics
from imblearn.over_sampling import SMOTE # SMOTE
from sklearn.preprocessing import RobustScaler, MinMaxScaler, StandardScaler # scaling methods

from sklearn.model_selection import GridSearchCV # grid search cross validation
from sklearn.model_selection import RandomizedSearchCV # randomized search cross validation
import pickle
#selecting 20 best features
# select_best= SelectKBest(chi2, k=20)
# X_feat_20 = select_best.fit_transform(data_X, data_y_trans)
# X_feat_20.shape


df = pd.read_csv("data.csv")
data_X=df.drop(['Bankrupt?',' Net Value Per Share (A)' ,' Revenue Per Share (Yuan ¥)',' Operating Profit Per Share (Yuan ¥)',' Per Share Net profit before tax (Yuan ¥)' ],axis=1)
data_y=df['Bankrupt?']
# print(data_X)

model = ExtraTreesClassifier(random_state=42)
model.fit(data_X, data_y)
feature_importance_std = pd.Series(model.feature_importances_, index=data_X.columns)
#feature_importance_std.nlargest(40).plot(kind='bar', title='Standardised Dataset Feature',figsize=(20,5),colormap='RdBu')
cols=[' Debt ratio %', ' Net Income to Total Assets',  
 ' Borrowing dependency',                           
 ' Net worth/Assets',           
 ' ROA(B) before interest and depreciation after tax',  
 ' Persistent EPS in the Last Four Seasons',          
 ' ROA(C) before interest and depreciation before interest',   
 ' Liability to Equity',                                      
 ' Retained Earnings to Total Assets',                         
 ' Net profit before tax/Paid-in capital',                    
 ' ROA(A) before interest and % after tax',   
 ' Fixed Assets Turnover Frequency',                          
 ' Degree of Financial Leverage (DFL)',                       
 ' Current Liabilities/Equity',                               
 ' Current Liability to Assets',                               
 ' Cash/Current Liability',                                   
 ' Interest Coverage Ratio (Interest expense to EBIT)',       
 ' Interest Expense Ratio',                                    
 ' Working Capital to Total Assets',                          
 ' Total debt/Total net worth',                                
 ' Current Liability to Equity',                               
 ' Interest-bearing debt interest rate',                                                          
 ' Equity to Liability',                                      
 ' Equity to Long-term Liability',                             
 ' Total income/Total expense',                               
  ' Allocation rate per person',                               
 ' Non-industry income and expenditure/revenue',              
 ' Inventory/Working Capital',                                 
 ' Current Liability to Current Assets',                       
 ' After-tax net Interest Rate',                              
 ' Continuous interest rate (after tax)',                      
 ' Average Collection Days',                                                                   
 ' Working Capital/Equity',                                    
 ' Total Asset Return Growth Rate Ratio' ,                     
 ' Operating profit per person' ,                               
 ' Total assets to GNP price']       

data_X_new=data_X[cols]
from sklearn.model_selection import train_test_split
X_train_40, X_test_40, y_train_40, y_test_40 = train_test_split(data_X_new, data_y, test_size = 0.30, random_state = 42)
over = SMOTE(sampling_strategy=0.2)
X_train_40,y_train_40 = over.fit_resample(X_train_40.astype('float'),y_train_40.ravel())
from sklearn.preprocessing import StandardScaler 
ss_20 = StandardScaler()
X_train_std_40 = ss_20.fit_transform(X_train_40)
X_test_std_40 = ss_20.fit_transform(X_test_40)
import numpy as np
import pandas as pd
import tensorflow as tf
# ann = tf.keras.models.Sequential()
# ann.add(tf.keras.layers.Dense(units=6, activation='relu'))
# ann.add(tf.keras.layers.Dense(units=6, activation='relu'))
# ann.add(tf.keras.layers.Dense(units=1, activation='sigmoid'))
# ann.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
# ann.fit(X_train_40, y_train_40, batch_size = 32, epochs = 100)
# b= ann.predict(X_test_40)
from sklearn.linear_model import LogisticRegression
lr_under = LogisticRegression(C= 100, penalty= 'l2', solver= 'newton-cg')
lr_under.fit(X_train_40,y_train_40)
pickle.dump(lr_under,open('model.pkl','wb'))
model=pickle.load(open('model.pkl','rb'))
