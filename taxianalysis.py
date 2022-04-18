import streamlit as st
import pandas as pd
# import seaborn as sns
import seaborn as sns
import matplotlib.pyplot as plt


from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

from PIL import Image

#containers
header = st.container()
dataset = st.container()
features = st.container()
model_Traning = st.container()
end=st.container()

with header:
    st.title('🚕  🚕  TAXI DATA ANALYSIS 🚕  🚕 ')
    st.text('In this project we are predicting the  fare amount')



with dataset:
    st.header('Taxi dataset')
    st.text('dataset we are taken to do analysis (head of dataset)..... ')
    taxi_data=pd.read_csv('tdata.csv')
    st.write(taxi_data.head())

    taxi_data.dropna(how='all')

    st.subheader('Dealing with duplicate data')
    st.write("**Click is you want to check duplicate data**")
    if st.button('Check'):
            duplicate_rows_df = taxi_data[taxi_data.duplicated()]
            st.write("number of duplicate rows: ", duplicate_rows_df.shape[0])
            st.write("Here are the duplicate rows", duplicate_rows_df)
    st.write('**Click if you want to delete the duplicants rows**')
    if st.button('Delete duplicated'):
        taxi_data.drop_duplicates(inplace=True)
        st.success('Duplicates deleted')
        duplicate_rows_df = taxi_data[taxi_data.duplicated()]
        st.write("number of duplicate rows: ", duplicate_rows_df.shape[0])
        st.write("Here are the duplicate rows", duplicate_rows_df)


    st.subheader('Pick-up Location id distribution on the dataset')
    pulocation_dist=pd.DataFrame(taxi_data['PULocationID'].value_counts())
    st.bar_chart(pulocation_dist)




with features:
    st.header('The features I created')
    image = Image.open('image.png')

    st.image(image, caption='Correlational matrix to check relations between variables')
   
    st.markdown('* **first feature:** From above correlational matrix we clearly know that our target variable is fare_amount ')
    st.markdown('* **second feature:** fare_amount  is depend on DOLocationID ,payment_type,extra,mta_tax,tip_amount,tolls_amount,improvement_surcharge,total_amount,congestion_surcharge')







with model_Traning:
    st.header('Time to train model')
    st.text('Here you get to choose the hyperparameter of model and see how the performance change')
    sel_col,disp_col=st.columns(2)
    
    max_depth=sel_col.slider('what sholud be the max_depth of the model?',min_value=10,max_value=100,value=20,step=10)
    n_estimators=sel_col.selectbox('how many trees should there be?', options=[100,200,300, ' No limit'], index=0)
    
    sel_col.text('here is a list of features in my data')
    sel_col.write(taxi_data.columns)
    input_feature=sel_col.text_input('which feature should be used as input feature?', 'PULocationID')
    
    
    if n_estimators==' No limit':
        regr=RandomForestRegressor(max_depth=max_depth)
    else:
        regr=RandomForestRegressor(max_depth=max_depth,n_estimators=n_estimators)
    
    x=taxi_data[[input_feature]]
    y=taxi_data[['fare_amount']]
    
    x_train,x_text,y_train,y_test=train_test_split(x,y,test_size=0.3)
    regr.fit(x_train,y_train)
    prediction=regr.predict(y_test)
    st.write(prediction)

    disp_col.subheader('mean absolute error of model is')
    disp_col.write(mean_absolute_error(y_test, prediction))

    disp_col.subheader('mean squared error of model is')
    disp_col.write(mean_squared_error(y_test, prediction))

    disp_col.subheader('r2 score of model is')
    disp_col.write(r2_score(y_test, prediction))

    # disp_col.subheader('accuracy  score of model is')
    # disp_col.write(accuracy_score(y_test, prediction))
    # accuracy_score(y_test, predict_y)
    
with end:
    st.text(' **Developed by - Mission proplacement -- Payal || Pooja || Sanyogita || Pratiksha')
