# pip install streamlit fbprophet yfinance plotly
import streamlit as st
from datetime import date

from bs4 import BeautifulSoup
import requests

import yfinance as yf
from fbprophet import Prophet
from fbprophet.plot import plot_plotly
from plotly import graph_objs as go

import tensorflow as tf
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

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


#set100, set100BK = thai_stock()
#print(set100BK)

START = "2015-01-01"
TODAY = date.today().strftime("%Y-%m-%d")

st.title('Stock Forecast App Sira')
markets = ["DOW30","NASDAQ100","SET100"]
selected_market = st.selectbox('Select Market', markets)

stocks = ['GOOG', 'AAPL', 'MSFT', 'GME',"PTTGC.BK"]
selected_stock = st.selectbox('Select dataset for prediction', selectionstock())

n_years = st.slider('Years of prediction:', 1, 4)
period = n_years*365


@st.cache
def load_data(ticker):
    data = yf.download(ticker, START, TODAY)
    data.reset_index(inplace=True)
    return data


data_load_state = st.text('Loading data...')
data = load_data(selected_stock)
data_load_state.text('Loading data... done!')

st.subheader('Raw data')
st.write(data.tail())


# Plot raw data
def plot_raw_data():
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=data['Date'], y=data['Open'], name="stock_open"))
    fig.add_trace(go.Scatter(x=data['Date'], y=data['Close'], name="stock_close"))
    fig.layout.update(title_text='Time Series data with Rangeslider', xaxis_rangeslider_visible=True)
    st.plotly_chart(fig)


plot_raw_data()

# Predict forecast with Prophet.
df_train = data[['Date', 'Close']]
df_train = df_train.rename(columns={"Date": "ds", "Close": "y"})

m = Prophet()
m.fit(df_train)
future = m.make_future_dataframe(periods=period)
forecast = m.predict(future)

# Show and plot forecast
st.subheader('Forecast data')
st.write(forecast.tail())

st.write(f'Forecast plot for {n_years} years')
fig1 = plot_plotly(m, forecast)
st.plotly_chart(fig1)

st.write("Forecast components")
fig2 = m.plot_components(forecast)
st.write(fig2)


#แปลงรายชื่อ .BK ออก
#option ดำงหุ้นต่างประเทศ
#recommend 10 อันดับ