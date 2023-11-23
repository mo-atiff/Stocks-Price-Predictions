import pandas as pd
import streamlit as st
import numpy as np
from pymongo import MongoClient

from tensorflow.keras.models import model_from_json
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.layers import LSTM
import pandas_datareader as pdr
import plotly.express as px
from sklearn.preprocessing import MinMaxScaler
from datetime import datetime, timedelta
import plotly.graph_objects as go


key = st.secrets["api_key"]

# global client

ticker_symbols = []
nasdaq = pd.read_csv("nasdaq.csv")

ticker_symbols = nasdaq['Symbol']

st.set_page_config(
    page_title="Stocks Predictions",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("<h1 style='text-align: centre; color: #39FF14;'>STOCK PRICE PREDICTOR📈</h1>",
                unsafe_allow_html=True)

with st.sidebar:
    ml = st.selectbox('Select from below', ['Signup', 'Login', 'Forecast'])

# @st.cache_data
# def connect(ab):
#     # global client
#     with st.spinner('Hold on...'):
#         client = MongoClient(ab)

#         database = client.Imstock
#         collection = database.stocks

#         collection2 = database.userdata
#     return database, collection, collection2
#                 # st.error('😲An Unfortunate Error Occurred...check your connection')

# database, collection, collection2 = connect('mongodb+srv://Atif:XHMswoIrHKVzfIjo@stockcluster.m5bop17.mongodb.net/')

# database = client.Imstock
# collection = database.stocks

# collection2 = database.userdata


if ml=='Forecast':
    col1, col2 = st.columns(2)
    col3, col4, col5, col6, col7 = st.columns(5)

    with col1:
        st.markdown("<h3 style='text-align: centre; color: #BF3EFF;'>Select Ticker Symbol</h3>",
                    unsafe_allow_html=True)
        ticker_symbol = st.selectbox("Select Ticker Symbol", ticker_symbols)

    with col2:
        st.markdown("<h3 style='text-align: centre; color: #BF3EFF;'>No of Days to Forecast</h3>",
                    unsafe_allow_html=True)
        forecast_days = st.slider("", min_value=1, max_value=30)

    print('Connecting.....')
    with st.spinner('Hold on...'):
        try:
            client = MongoClient('mongodb+srv://Atif:XHMswoIrHKVzfIjo@stockcluster.m5bop17.mongodb.net/') 
        except Exception:
            st.error('😲An Unfortunate Error Occurred...check your connection')

    print('connection sucessful...')
    database = client.Imstock
    collection = database.stocks # cluster stocks stored in company

    with col5:
        st.write(' ')
        st.write(' ')
        st.write(' ')

        st.markdown(
            """
            <style>
            .stButton button {
                background-color: black;
                color: white;
                font-size: 16px;
                padding: 10px 20px;
                border-radius: 5px;
                border: none;
            }
            </style>
            """,
            unsafe_allow_html=True
        )

        predict_button = st.button("Forecast")


    if predict_button:
        query = {"tickerSym": ticker_symbol}
        results = collection.find(query)

        data = []

        for i in results:
            data.append(i)

        if len(data) == 0:
            # st.write("Model was not found")
            data = pdr.get_data_tiingo(ticker_symbol, api_key = key)
            data = pd.concat(objs=[data], axis=0)
            data = data.reset_index()
            last_date = data.iloc[-1]['date']

            data = data.drop(['symbol', 'date', 'divCash', 'splitFactor'], axis = 1)

            scaler = MinMaxScaler(feature_range=(0, 1))

            scaled_data = scaler.fit_transform(data)
            tx = [] 
            ty = []
            n_past = 100
            for i in range(100, len(scaled_data)-1+1):
                tx.append(scaled_data[i-n_past:i, 0:data.shape[1]])
                ty.append(scaled_data[i+1-1:i+1, 1]) 

            tx = np.array(tx)
            ty = np.array(ty)

            model = Sequential()
            model.add(LSTM(64, activation='relu', input_shape=(tx.shape[1], tx.shape[2]), return_sequences=True))
            model.add(LSTM(32, activation='relu', return_sequences=False))
            model.add(Dropout(0.2))
            model.add(Dense(ty.shape[1]))

            model.compile(optimizer='adam', loss='mse')
            with st.spinner('Model is predicting have patience... 🤫About 2-3 mins...'):
                model.fit(tx, ty, epochs=10, batch_size=65, validation_split=0.1, verbose=1)

            sc = scaler.transform(data[-100:]) 

            sc = np.reshape(sc, (1, 100, 10))

            predictions = []
            for i in range(30):
                prediction = model.predict(sc) 

                next_day_prediction = prediction[0][0]  
                
                predictions.append(next_day_prediction)
                
                sc = np.roll(sc, -1, axis=1)
                sc[0, -1, :] = next_day_prediction

            predictions = np.array(predictions)
            predictions = predictions.reshape(-1, 1)

            prediction_copies = np.repeat(predictions, data.shape[1], axis=-1)
            preds = scaler.inverse_transform(prediction_copies)[:, 0].tolist()  

            dataT = {
                "tickerSym": ticker_symbol,
                "last_date" : last_date,
                "last_value" : data.iloc[-1][1],
                "forecast" : preds,
            }

            result = collection.insert_one(dataT)

            now = datetime(last_date.year, last_date.month, last_date.day)

            future_dates = []

            for i in range(forecast_days):  
                future_date = now + timedelta(days=i)
                future_dates.append(future_date.date())

            to_plot = pd.DataFrame({'DATES' : future_dates, 'FUTURE PREDICTION OF HIGHEST STOCK PRICE' : preds[:forecast_days]})
            col11, col12 = st.columns(2)
            
            with col11:
                line = px.line(to_plot, x = 'DATES', y = 'FUTURE PREDICTION OF HIGHEST STOCK PRICE', title = 'PREDICTION FOR HIGH PRICE')
                st.plotly_chart(line)

            with col12:
                area = px.area(to_plot, x = 'DATES', y = 'FUTURE PREDICTION OF HIGHEST STOCK PRICE')
                st.plotly_chart(area)

            col555, col566, col577 = st.columns(3)
            with col566:
                fig = go.Figure(go.Indicator(
                    mode="number+delta",
                    value=preds[0],
                    delta={'position': "top", 'reference': data.iloc[-1][1]},
                    domain={'x': [0, 1], 'y': [0, 1]},
                    title = {'text': 'HIGH VALUE FROM THE LAST DAY'}
                ))

                fig.update_layout(
                    width=350,  
                    height=350  
                )

                st.plotly_chart(fig)
            
            col444, col455, col466 = st.columns(3)
            with col455:
                st.write("HIGHEST PRICE OF A STOCK FORECAST : ")
                st.dataframe(to_plot)

            
            st.write(f'LINK TO YOUR STOCK : https://finance.yahoo.com/quote/{ticker_symbol}')


        else:
            document = collection.find_one()

            last_date = document['last_date']
            pred_vals = document['forecast']
            last_value = document['last_value']
            now = datetime(last_date.year, last_date.month, last_date.day)

            future_dates = []

            for i in range(forecast_days):
                future_date = now + timedelta(days=i)
                future_dates.append(future_date.date())
            
            to_plot = pd.DataFrame({'DATES' : future_dates, 'FUTURE PREDICTION OF HIGHEST STOCK PRICE' : pred_vals[:forecast_days]})
            col77, col88 = st.columns(2)

            with col77:
                line = px.line(to_plot, x = 'DATES', y = 'FUTURE PREDICTION OF HIGHEST STOCK PRICE', title = 'PREDICTION FOR HIGH PRICE')
                st.plotly_chart(line)

            with col88:
                area = px.area(to_plot, x = 'DATES', y = 'FUTURE PREDICTION OF HIGHEST STOCK PRICE')
                st.plotly_chart(area) 

            col55, col56, col57 = st.columns(3)
            with col56:
                fig = go.Figure(go.Indicator(
                    mode="number+delta",
                    value=pred_vals[0],
                    delta={'position': "top", 'reference': last_value},
                    domain={'x': [0, 1], 'y': [0, 1]},
                    title = {'text' : 'HIGH VALUE FROM THE LAST DAY'}
                ))

                fig.update_layout(
                    width=350,  
                    height=350  
                )

                st.plotly_chart(fig)


            col44, col45, col46 = st.columns(3)
            with col45:
                st.write("HIGHEST PRICE OF A STOCK FORECAST : ")
                st.dataframe(to_plot)

            st.write(f'URL TO YOUR STOCK : https://finance.yahoo.com/quote/{ticker_symbol}')




elif ml=='Signup':
    st.title("Sign Up")
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")
    conf = st.text_input("Password", type="password", key="login-password")
    email = st.text_input("Email")

    if password == conf:
        pass
    elif password != conf:
        st.error("Password's do not match")

    if st.button("Sign Up"):
        with st.spinner('Signing u up...'):
            try:
                client = MongoClient('mongodb+srv://Atif:XHMswoIrHKVzfIjo@stockcluster.m5bop17.mongodb.net/') 
            except Exception:
                st.error('😲An Unfortunate Error Occurred...check your connection')

        print('connection sucessful...')
        database = client.Imstock
        collection = database.userdata 
        user_to_ins = {'Username':username, 'Password':password, 'Email':email}

        done = collection.insert_one(user_to_ins)
        st.success('Signed In sucessfully')

else:
    st.title("Log In")
    usernamel = st.text_input("Username")
    passwordl = st.text_input("Password", type="password")
    # database3 = client.Imstock
    # collection3 = database3.userdata
    logbut = st.button('Log In')
    if logbut:
        with st.spinner('Connecting to database...'):
            try:
                client = MongoClient('mongodb+srv://Atif:XHMswoIrHKVzfIjo@stockcluster.m5bop17.mongodb.net/') 
            except Exception:
                st.error('😲An Unfortunate Error Occurred...check your connection')
        
        to_check = {'Username':usernamel, 'Password':passwordl}

        print('connection sucessful...')
        database = client.Imstock
        collection = database.userdata 

        if collection.find_one(to_check):
            st.success('Sucessfully Logged In')
        else:
            st.error('Invalid Username or Password')

