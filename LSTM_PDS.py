# pip install streamlit fbprophet yfinance plotly
import streamlit as st
from datetime import date,timedelta

from bs4 import BeautifulSoup
import requests

import yfinance as yf
from fbprophet import Prophet
from fbprophet.plot import plot_plotly
from plotly import graph_objs as go

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np

tf.random.set_seed(42)
np.random.seed(42)

print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

df1 = pd.read_csv("top5_stock.csv", index_col = False)
#df2 = df1.copy()
#df2 = df2.iloc[:3]

pd.options.display.float_format = "{:,.2f}".format

def dow30():
    dow_30 = []
    url = "https://en.wikipedia.org/wiki/Dow_Jones_Industrial_Average"
    data = requests.get(url).text
    soup = BeautifulSoup(data, "html.parser")
    table = soup.find('table', class_='wikitable sortable')
    for row in table.tbody.find_all('tr'):
        columns = row.find_all('td')
        if (columns != []):
            st = columns[1].text.strip()
            dow_30.append(st)
    return dow_30

def thai_stock():
    url = "https://www.settrade.com/C13_MarketSummary.jsp?detail=SET100"
    data = requests.get(url).text
    soup = BeautifulSoup(data, "html.parser")
    selector = 'a.link-stt'
    symbols = soup.select(selector)
    set100 = list(map(lambda s: s.text, symbols[9:109]))
    set100BK = [i +".BK" for i in set100]
    return set100BK

def nasdaq100():
    nasdaq_100 = []
    url = "https://en.wikipedia.org/wiki/Nasdaq-100"
    data = requests.get(url).text
    soup = BeautifulSoup(data, "html.parser")
    table = soup.find('table', class_='wikitable sortable', id ="constituents")
    for row in table.tbody.find_all('tr'):
        columns = row.find_all('td')
        if (columns != []):
            st = columns[1].text.strip()
            nasdaq_100.append(st)
    return nasdaq_100

def selectionstock():
    list_stock = []
    if selected_market == "SET100" :
        list_stock = thai_stock()
    elif selected_market == "DOW30" :
        list_stock = dow30()
    elif selected_market == "NASDAQ100":
        list_stock = nasdaq100()
    return list_stock



# Predict section

time_step = 50
test_size = 7
def prep_data(data):
    series_1 = data.set_index('Date')['Close']
    series_1_log = np.log(series_1)
    df_log = pd.concat([series_1_log.to_frame()], axis=1)
    df_log['DiffLogClose'] = df_log['Close'].diff()
    sc_1 = StandardScaler()
    data_scaled = sc_1.fit_transform(df_log[['DiffLogClose']])
    df_log['ScaledLogReturn'] = data_scaled.flatten()
    return df_log, sc_1

def multi_output_dataset(dataset, time_step, test_size=7):
    Tx = time_step
    Ty = test_size
    X = []
    Y = []
    for t in range(len(dataset) - Tx - Ty + 1):
        x = dataset[t:t+Tx]
        X.append(x)
        y = dataset[t+Tx:t+Tx+Ty]
        Y.append(y)
    X_log_1 = np.array(X).reshape(-1, Tx)
    y_log = np.array(Y).reshape(-1, Ty)
    return X_log_1, y_log

def split_train_test_xy_log(X_log,y_log):
    X_train_log, y_train_log = X_log[:-1], y_log[:-1]
    X_test_log, y_test_log = X_log[-1:], y_log[-1:]
    return X_train_log, y_train_log,X_test_log, y_test_log

def reshape_test_log(X_train_log, X_test_log):
    X_train_log = X_train_log.reshape(X_train_log.shape[0], X_train_log.shape[1],1 ) # เพิ่มอีก 1 มิติเข้ามา(เป็นกฎ)
    X_test_log = X_test_log.reshape(X_test_log.shape[0], X_test_log.shape[1], 1)
    return X_train_log, X_test_log

def LSTM_model(X_train_log,y_train_log,X_test_log,y_test_log, time_step):
    tf.keras.backend.clear_session()
    tf.random.set_seed(42)
    model = Sequential()
    model.add(LSTM(128, return_sequences=True))
    model.add(LSTM(64))
    model.add(Dense(7))
    model.compile(loss='mse', optimizer='adam')
    return model

def multi_output_dataset_for_test(dataset, time_step):
    Tx = time_step
    X = dataset[len(dataset) - Tx:len(dataset)]
    X_prod_1 = np.array(X).reshape(-1,Tx, 1)
    return X_prod_1

def pred_sc(model,X_prod_1,sc_1,df,data):
  Pprod = model.predict(X_prod_1)
  Pprod = Pprod[0]
  Pprod = sc_1.inverse_transform(Pprod.reshape(-1, 1)).flatten()
  last_train = df.iloc[-1]['Close']
  pred = last_train + np.cumsum(Pprod)
  pred = np.exp(pred)
  pred = pd.Series(pred)
  pred = pred.to_frame(name='pred')
  price = data[['Close']]
  price = pd.concat([price, pred], ignore_index=True)
  return price


def daterange(date1, date2):
    for n in range(int ((date2 - date1).days)+1):
        yield date1 + timedelta(n)

def create_df(price, data):
  price["Date"] = data["Date"]
  date_fil = []
  start_dt = date.today()
  end_dt = date.today() + timedelta(days=14)
  weekdays = [6,7]
  for dt in daterange(start_dt, end_dt):
      if dt.isoweekday() not in weekdays:
          d=dt.strftime("%Y-%m-%d")
          date_fil.append(d)
  date7 = date_fil[:7]
  price.loc[len(data):len(data)+7,"Date"] = date7
  return price

def LSTM_process(*arg):
  df, sc_1 = prep_data(data)
  series_log_1 = df['ScaledLogReturn'].dropna().to_numpy()
  X_log_1, y_log = multi_output_dataset(series_log_1, time_step)
  X_train_log, y_train_log,X_test_log, y_test_log = split_train_test_xy_log(X_log_1,y_log)
  X_train_log, X_test_log = reshape_test_log(X_train_log, X_test_log)
  model = LSTM_model(X_train_log,y_train_log,X_test_log,y_test_log,time_step)
  history = model.fit(X_train_log, y_train_log, validation_data=(X_test_log, y_test_log), batch_size=32, epochs=50, verbose=0)
  X_prod_1 = multi_output_dataset_for_test(series_log_1, time_step)
  price = pred_sc(model,X_prod_1,sc_1,df,data)
  price_df = create_df(price, data)
  return price_df


START = "2015-01-01"
TODAY = date.today().strftime("%Y-%m-%d")

st.title('Stock Forecast App AI4BA')
markets = ["DOW30","NASDAQ100","SET100"]
selected_market = st.selectbox('Select Market', markets)

selected_stock = st.selectbox('Select dataset for prediction', selectionstock())

# n_years = st.slider('Years of prediction:', 1, 4)
# period = n_years*365

st.sidebar.title("7 days Recommend")
st.sidebar.subheader("Top 5 SET100")
st.sidebar.dataframe(df1)
#st.sidebar.write("Top XX DOW30")
#st.sidebar.dataframe(df2)

@st.cache
def load_data(ticker):
    data = yf.download(ticker, START, TODAY)
    data.reset_index(inplace=True)
    return data


data_load_state = st.text('Loading data...')
data = load_data(selected_stock)
data_load_state.text('Loading data... done!')

st.subheader('Raw data')
#st.write(data.tail())
data1 = data.copy()
data1["Date"] = pd.to_datetime(data1["Date"])
data1 = data1.sort_values(by="Date", ascending=False)
data1["Date"] = data1['Date'].dt.strftime("%Y-%m-%d")
st.dataframe(data1)


# Plot raw data
def plot_raw_data():
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=data['Date'], y=data['Open'], name="stock_open"))
    fig.add_trace(go.Scatter(x=data['Date'], y=data['Close'], name="stock_close"))
    fig.layout.update(title_text='Time Series data', xaxis_rangeslider_visible=True)
    st.plotly_chart(fig)


plot_raw_data()

st.subheader('Next 7 days forecast')
with st.spinner('Wait for train model...'):
    price_df = LSTM_process()
    price_df1 = price_df[["Date","Close","pred"]]
    price_df1 = price_df1.astype({'pred': 'float64'})
    price_df1 = price_df1.replace(np.nan, "-")
    pred_list = price_df["pred"].dropna().tolist()
    percent = ((pred_list[-1]/pred_list[0]) * 100) - 100

#price_df1 = price_df1.set_index("Date")
#price_df1 = price_df1.replace(np.nan, "-")
#price_df2 = price_df1.replace("-", 0)
#price_df2 = price_df1.astype('str').dtypes

st.write(price_df1.tail(7))
st.metric(label="Day 7", value="{:.2f}".format(pred_list[-1]), delta= "{:.2f}%".format(percent))

price_df_gr = price_df.iloc[-100:]

def plot_forecast_data():
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=price_df_gr['Date'], y=price_df_gr['pred'], name="Predict"))
    fig.add_trace(go.Scatter(x=price_df_gr['Date'], y=price_df_gr['Close'], name="Close actual"))
    fig.layout.update(title_text='Forcast ' + selected_stock, xaxis_rangeslider_visible=True)
    st.plotly_chart(fig)

plot_forecast_data()