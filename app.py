from flask import Flask,render_template,request
import pickle
import numpy as np
import pandas as pd
import json
from flask import Flask,render_template

app=Flask(__name__,template_folder='template')
model=pickle.load(open('model.pkl','rb'))
company= '{" Debt ratio %":0.11179974," Net Income to Total Assets":0.8201747078," Borrowing dependency":0.3753721345," Net worth\/Assets":0.88820026," ROA(B) before interest and depreciation after tax":0.5537769688," Persistent EPS in the Last Four Seasons":0.2213292994," ROA(C) before interest and depreciation before interest":0.5009506167," Liability to Equity":0.2788109421," Retained Earnings to Total Assets":0.9393391394," Net profit before tax\/Paid-in capital":0.1822109946," ROA(A) before interest and % after tax":0.5841692106," Fixed Assets Turnover Frequency":0.0003366415," Degree of Financial Leverage (DFL)":0.0268549274," Current Liabilities\/Equity":0.3304650747," Current Liability to Assets":0.1009746235," Cash\/Current Liability":0.002244794," Interest Coverage Ratio (Interest expense to EBIT)":0.5654451626," Interest Expense Ratio":0.6308218381," Working Capital to Total Assets":0.7827919893," Total debt\/Total net worth":0.0055590512," Current Liability to Equity":0.3304650747," Interest-bearing debt interest rate":0.0002210221," Equity to Liability":0.0336727066," Equity to Long-term Liability":0.1133919473," Total income\/Total expense":0.0027190165," Allocation rate per person":0.01127465," Non-industry income and expenditure\/revenue":0.3036504512," Inventory\/Working Capital":0.2772535851," Current Liability to Current Assets":0.0355712951," After-tax net Interest Rate":0.8094379537," Continuous interest rate (after tax)":0.7816268167," Average Collection Days":0.0084383286," Working Capital\/Equity":0.7340604254," Total Asset Return Growth Rate Ratio":0.2643619212," Operating profit per person":0.3975939033," Total assets to GNP price":0.0022800252}'
@app.route('/')
def hello():
    return render_template('index.html',company=json.dumps(company))
    
@app.route('/predict',methods=['POST','GET'])
def predict():
    cols=[' Debt ratio %', ' Net Income to Total Assets',  ' Borrowing dependency',                           
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
    dict={}
    # data=request.get_data()
    # data=json.loads(data.decode('utf-8'))
    print("**")
    #print(data)
    data= {' Debt ratio %': 0.11179974, ' Net Income to Total Assets': 0.8201747078, ' Borrowing dependency': 0.3753721345, ' Net worth/Assets': 0.88820026, ' ROA(B) before interest and depreciation after tax': 0.5537769688, ' Persistent EPS in the Last Four Seasons': 0.2213292994, ' ROA(C) before interest and depreciation before interest': 0.5009506167, ' Liability to Equity': 0.2788109421, ' Retained Earnings to Total Assets': 0.9393391394, ' Net profit before tax/Paid-in capital': 0.1822109946, ' ROA(A) before interest and % after tax': 0.5841692106, ' Fixed Assets Turnover Frequency': 0.0003366415, ' Degree of Financial Leverage (DFL)': 0.0268549274, ' Current Liabilities/Equity': 0.3304650747, ' Current Liability to Assets': 0.1009746235, ' Cash/Current Liability': 0.002244794, ' Interest Coverage Ratio (Interest expense to EBIT)': 0.5654451626, 
    ' Interest Expense Ratio': 0.6308218381, ' Working Capital to Total Assets': 0.7827919893, ' Total debt/Total net worth': 0.0055590512, ' Current Liability to Equity': 0.3304650747, ' Interest-bearing debt interest rate': 0.0002210221, ' Equity to Liability': 0.0336727066, ' Equity to Long-term Liability': 0.1133919473, ' Total income/Total expense': 0.0027190165, ' Allocation rate per person': 0.01127465, ' Non-industry income and expenditure/revenue': 0.3036504512, ' Inventory/Working Capital': 0.2772535851, ' Current Liability to Current Assets': 0.0355712951, ' After-tax net Interest Rate': 0.8094379537, ' Continuous interest rate (after tax)': 0.7816268167, ' Average Collection Days': 0.0084383286, ' Working Capital/Equity': 0.7340604254, ' Total Asset Return Growth Rate Ratio': 0.2643619212, ' Operating profit per person': 0.3975939033, ' Total assets to GNP price': 0.0022800252}
    dict = { your_key: data[your_key] for your_key in cols }
    dict=  {' Debt ratio %': 0.11179974, ' Net Income to Total Assets': 0.8201747078, ' Borrowing dependency': 0.3753721345, ' Net worth/Assets': 0.88820026, ' ROA(B) before interest and depreciation after tax': 0.5537769688, ' Persistent EPS in the Last Four Seasons': 0.2213292994, ' ROA(C) before interest and depreciation before interest': 0.5009506167, ' Liability to Equity': 0.2788109421, ' Retained Earnings to Total Assets': 0.9393391394, ' Net profit before tax/Paid-in capital': 0.1822109946, ' ROA(A) before interest and % after tax': 0.5841692106, ' Fixed Assets Turnover Frequency': 0.0003366415, ' Degree of Financial Leverage (DFL)': 0.0268549274, ' Current Liabilities/Equity': 0.3304650747, ' Current Liability to Assets': 0.1009746235, ' Cash/Current Liability': 0.002244794, ' Interest Coverage Ratio (Interest expense to EBIT)': 0.5654451626,
     ' Interest Expense Ratio': 0.6308218381, ' Working Capital to Total Assets': 0.7827919893, ' Total debt/Total net worth': 0.0055590512, ' Current Liability to Equity': 0.3304650747, ' Interest-bearing debt interest rate': 0.0002210221, ' Equity to Liability': 0.0336727066, ' Equity to Long-term Liability': 0.1133919473, ' Total income/Total expense': 0.0027190165, ' Allocation rate per person': 0.01127465, ' Non-industry income and expenditure/revenue': 0.3036504512, ' Inventory/Working Capital': 0.2772535851, ' Current Liability to Current Assets': 0.0355712951, ' After-tax net Interest Rate': 0.8094379537, ' Continuous interest rate (after tax)': 0.7816268167, ' Average Collection Days': 0.0084383286, ' Working Capital/Equity': 0.7340604254, ' Total Asset Return Growth Rate Ratio': 0.2643619212, ' Operating profit per person': 0.3975939033, ' Total assets to GNP price': 0.0022800252}
    #print(dict)
    li=list(data.values())
    processedData = pd.DataFrame([li],columns=cols)
    # features=[float(x) for x in request.get_data().values()]
    # final=[np.array(features)]
    print("the model started");
    res = model.predict_proba(processedData)*100
    res='{0:.{1}f}'.format(res[0][1],2)
    res=str(res)
    print(res)
    return render_template('ind.html',pred=res);

# main driver function
if __name__ == '__main__':

	# run() method of Flask class runs the application
	# on the local development server.
	app.run(debug=True,port=5050)