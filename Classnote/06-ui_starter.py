from datetime import date, timedelta
import pandas as pd
import plotly.express as px
import streamlit as st
import yfinance as yf
from matplotlib import ticker

# CONSTANTS
END = date.today()
START = END - timedelta(365)

# data handling
def get_stock_data(ticker, start, end):
    try:
        data = yf.download(ticker, start, end, auto_adjust=False)
        if data.empty:
            return None, f"No data found for {ticker}"
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.get_level_values(0)
        return data, f"Successfully loaded data for {ticker}"
    except Exception as e:
        return None, f"Error {e}"

# sidebar
st.sidebar.title("Inputs")
ticker= st.sidebar.text_input("Stock symbol",value="AAPL").upper()
col1, col2 = st.sidebar.columns(2)
start = col1.date_input("Start Date",value=START)
end = col2.date_input("End Date",value=END)
moving_average = st.sidebar.slider("Moving Average Window",
                                   min_value=10,
                                   max_value=100,
                                   value=50,
                                   step=5)

run = st.sidebar.button(
    "Run Analysis",
    type="primary")

#main UI
st.title("Stock Analyzer")
if run:
    with st.spinner(f"Fetching {ticker} data..."):
        df, message = get_stock_data(ticker, start, end)
    if df is not None:
        st.sidebar.success(message)
    else:
        st.sidebar.error(message)
        df['MA'] = df['Close'].pct_change() * 100

        #output
        tab1, tab2, tab3 = st.tabs(['Chart', 'Statistics', 'Raw Data'])
        with tab1:
            st.subheader(f"{ticker} High Level Analysis")
            col1, col2, col3 = st.columns(3)
            col1.metric(f'Latest Price', f'{df['Closed'].iloc[-1]:.2f}')
            col2.metric(f'Cum Change',
                        f'{((df['Closed'].iloc[-1]/df['Closed'].iloc[0]) -1):.2f}')
            col3.metric(f'Trading Days', f'{len(df)}')
            figure = px.line(df, y=['Closed', 'MA'])
            figure.update_layout(hovermode='x unified')
            with st.spinner('Creating chart'):
                st.plotly_chart(figure, use_container_width=True)

        with tab2:
            st.subheader("Summary Statistics")
            col1, col2 = st.columns(2)
            with col1:
                st.write("**Daily Change Statistics**")
                summary = df['pct_change'].describe()
                st.dataframe(summary)
            with col2:
                price_stats = pd.DataFrame(
                    {'Metric': ['High', 'Low', 'Mean', 'Volatility'],
                     'Values': [
                         f"{df["Closed"].max()}",
                         f"{df["Closed"].min()}",
                         f"{df["Closed"].mean()}",
                         f"{df["Closed"].std()}"
                     ]
                    }
                )
                st.dataframe(price_stats)


