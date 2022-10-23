import streamlit as st
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import pandas_datareader as pdr
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM
from datetime import datetime
from datetime import timedelta

# ticker_symbols = ['GOOG', 'TSLA', 'AMZN', 'MSFT', 'AAPL']
ticker_symbols = []
nasdaq = pd.read_csv("nasdaq.csv")

st.set_page_config(
    page_title="Stock Price Predictor",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    [data-testid="stSidebar"][aria-expanded="true"] > div:first-child {
        width: 190px;
    }
    [data-testid="stSidebar"][aria-expanded="false"] > div:first-child {
        width: 100px;
        margin-left: -500px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown("<h1 style='text-align: centre; color: purple;'>STOCK PRICE PREDICTIONS</h1>",
                unsafe_allow_html=True)

ticker_symbols = nasdaq['Symbol']

with st.sidebar:
    st.markdown("<h3 style='text-align: centre; color: red;'>NO PRICE IS TOO LOW FOR A BEAR OR TOO HIGH FOR A BULL</h3>",
                    unsafe_allow_html=True)
    st.image('bear_bull.png')
    tick = st.selectbox("Select Ticker Symbol", ticker_symbols)
    future_days = st.number_input(
        'How Many Future Days Forecast You Wanna See', min_value=1, max_value=1000, step=1)

try:
    key = 'd1e0bf0b26e537200ebc6fce031449455a3f44e9'
    amz = pdr.get_data_tiingo(tick, api_key = key)
except Exception:
    st.error("Oop's This Error Wasn't Supposed to Happen Try Again after some Time")
    exit()

amz.to_csv('apl.csv')
amz = pd.read_csv('apl.csv')
print("csv-created\n")

low = amz['close']

scaler = MinMaxScaler(feature_range=(0, 1))
low = np.array(low).reshape(-1, 1)

scaler.fit(low)
low = scaler.transform(low)

def create_dataset(dataset, time_step=1):
    dataX, dataY = [], []
    for i in range(len(dataset)-time_step-1):
        a = dataset[i:(i+time_step), 0]   ###i=0, 0,1,2,3-----99   100 
        dataX.append(a)
        dataY.append(dataset[i + time_step, 0])
    return np.array(dataX), np.array(dataY)


time_step = 100
X_train, y_train = create_dataset(low, time_step)

X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)

print("dataset created\n")

col1, col2, col3 = st.columns(3)

with col2:
    st.write('')
    st.write('')
    st.write('')
    st.write(f"YOU SELECTED : {tick}")
st.write('')

m = st.markdown("""
<style>
div.stButton > button:first-child {
    background-color: purple; color : white
}
</style>""", unsafe_allow_html=True)

stocks = st.button("Predict Stocks")

if stocks:
    model=Sequential()
    model.add(LSTM(50,return_sequences=True,input_shape=(100,1)))
    model.add(LSTM(50,return_sequences=True))
    model.add(LSTM(50))
    model.add(Dense(1))
    model.compile(loss='mean_squared_error',optimizer='adam')

    lst = []

    with st.spinner('Model is Predicting Have Patience... It Takes About 4-5 Mins at Max'):
        st.balloons()
        model.fit(X_train,y_train,epochs=100,batch_size=64,verbose=1)
        st.snow()

    print("Training Completed\n")

    data_len = len(low) - 100

    x_input = low[data_len:].reshape(1, -1)

    temp_input = list(x_input)
    temp_input = temp_input[0].tolist() 

    print(("Prediction starts\n"))
    lst_output=[]
    n_steps=100
    i=0
    while(i<future_days):
        if(len(temp_input)>100):
            #print(temp_input)
            x_input=np.array(temp_input[1:])
            print("{} day input {}".format(i,x_input))
            x_input = x_input.reshape(1,-1)
            x_input = x_input.reshape((1, n_steps, 1))
            #print(x_input)
            yhat = model.predict(x_input, verbose=0)
            print("{} day output {}".format(i, yhat))
            temp_input.extend(yhat[0].tolist())
            temp_input=temp_input[1:]
            #print(temp_input)
            lst_output.extend(yhat.tolist())
            i=i+1
        else:
            x_input = x_input.reshape((1, n_steps,1))
            yhat = model.predict(x_input, verbose=0)
    #         print(yhat[0]) 
            temp_input.extend(yhat[0].tolist())
    #         print(len(temp_input))
            lst_output.extend(yhat.tolist())
            i=i+1 


    apl = scaler.inverse_transform(lst_output)
    print("APL : ", apl)
    day_new = np.arange(1,101)
    diff = 101 + future_days
    day_pred = np.arange(101,diff)

    print("Have Future Values\n")

    last_date = amz.iloc[len(amz)-1]['date'].split()[0]

    date_last = datetime.strptime(last_date, "%Y-%m-%d")
    date_last_F = date_last.strftime("%d-%m-%Y")
    temp_l = date_last_F.split('-')

    date_last_Fq = datetime(int(temp_l[2]), int(temp_l[1]), int(temp_l[0]))
    final_dates = []

    for i in range(1, future_days+1):
        date_last_Fq += timedelta(days=1)
        print(date_last_Fq)
        final_dates.append(date_last_Fq)


    FINAL_DATES = []
    for i in final_dates:
        i = str(i)
        k = i.split()[0]
        FINAL_DATES.append(k)
    st.markdown("<h5 style='text-align: centre; color: yellow;'>Prediction on Closing Price</h5>",
                unsafe_allow_html=True)

    print("Dates are done\n")
    apl = apl.flatten()
    fig, x = plt.subplots()
    plt.gcf().autofmt_xdate()
    plt.legend('PREDICTION')
    x.plot_date(FINAL_DATES, apl)
    x.plot(apl, label = "PREDICTION")
    x.legend()
    st.pyplot(fig)
    print(apl)

    # show exact prices

    despo = pd.DataFrame({"DATES" : FINAL_DATES, "EXPECTED STOCK PRICE" : apl})
    col4, col5, col6 = st.columns(3)
    with col5:
        st.markdown("<h3 style='text-align: centre; color: cyan;'>DATES vs STOCK PRICE</h3>",
                    unsafe_allow_html=True)
        st.dataframe(despo)

    fig2, x2 = plt.subplots()
    x2.plot(day_new,scaler.inverse_transform(low[data_len:]), label = "ActualStocks")
    x2.plot(day_pred,apl, label = "PredictedStocks")
    x2.legend()
    st.pyplot(fig2)
    cc1, cc2, cc3 = st.columns(3)
    with cc2:
        st.markdown("<h3 style='text-align: centre; color: cyan;'>ACTUAL vs PRECDICTED</h3>",
                        unsafe_allow_html=True)

#     st.dataframe(amz)

    print("sucess")


