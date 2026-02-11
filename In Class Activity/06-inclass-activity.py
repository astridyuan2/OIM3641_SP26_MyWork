from datetime import date, timedelta
import pandas as pd
import plotly.express as px
import streamlit as st
import yfinance as yf

st.set_page_config(layout='wide', page_title="Stock Analyzer")

# CONSTANTS
END = date.today()
START = END - timedelta(365)

# data handling
@st.cache_data
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
st.sidebar.title("🗠 Inputs")
ticker = st.sidebar.text_input("Stock symbol", value="AAPL").upper()
comparison_ticker = st.sidebar.text_input("Comparison Ticker", value="SPY").upper()
col1, col2 = st.sidebar.columns(2)
start = col1.date_input("Start Date", value=START)
end = col2.date_input("End Date", value=END)
moving_average = st.sidebar.slider("Moving Average Window",
                                   min_value=10,
                                   max_value=100,
                                   value=50,
                                   step=5)
run = st.sidebar.button("Run Analysis", type='primary')

#main UI
st.title("Stock Analyzer")

if run:
    with st.spinner(f"Fetching {ticker} data..."):
        df, message = get_stock_data(ticker, start, end)
    ### ICA 6
    with st.spinner(f"Fetching {comparison_ticker} data..."):
        comparison_df, comparison_message = get_stock_data(comparison_ticker, start, end)

    if df is not None:
        st.sidebar.success(message)
    else:
        st.sidebar.error(message)
    df['MA'] = df['Close'].rolling(window=moving_average).mean()
    df['pct_change'] = df['Close'].pct_change() * 100

    # Normalize both tickers to base 100
    df['normalized_close'] = (df['Close'] / df['Close'].iloc[0]) * 100
    comparison_df['normalized_close'] = (comparison_df['Close'] / comparison_df['Close'].iloc[0]) * 100


    if comparison_df is not None:
        st.sidebar.success(comparison_message)
    else:
        st.sidebar.error(comparison_message)

    if df is None or comparison_df is None:
        st.stop()

    # output
    tab1, tab2, tab3, tab4 = st.tabs(['📉 Chart', ' 📊Statistics', ' 🗄️Raw Data', '📈Comparison'])
    with tab1:
        st.subheader(f"{ticker} High Level Analysis")
        col1, col2, col3 = st.columns(3)
        col1.metric(f'Latest Price', f'{df['Close'].iloc[-1]:.2f}')
        col2.metric(f'Cum Change',
                    f'{((df['Close'].iloc[-1]/df['Close'].iloc[0]) -1) * 100:.2f}')
        col3.metric(f'Trading Days', f'{len(df)}')
        figure = px.line(df, y= ['Close', 'MA'])
        figure.update_layout(hovermode='x unified')
        with st.spinner("Creating chart"):
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
                     f"{df["Close"].max()}",
                     f"{df["Close"].min()}",
                     f"{df["Close"].mean()}",
                     f"{df["Close"].std()}"
                 ]
                }
            )
            st.dataframe(price_stats)

    with tab3:
        st.subheader("Raw Data")
        st.dataframe(df.tail(10))
        csv = df.to_csv()
        st.download_button(
                           label="Download Data",
                           data = csv,
                           file_name = f"{ticker}_{start}_{end}.csv",
                           mime="text/csv"
                            )

    with tab4:
        st.subheader("Comparison (Base 100)")
        plot_df = pd.DataFrame({
            ticker: df["normalized_close"],
            comparison_ticker: comparison_df["normalized_close"]
        }, index=df.index)

        fig = px.line(
            plot_df,
            x=plot_df.index,
            y=[ticker, comparison_ticker],
            title=f"{ticker} vs {comparison_ticker} Performance (Base 100)",
            labels={"value": "Normalized Close (Base 100)", "variable": "Ticker", "x": "Date"}
        )
        st.plotly_chart(fig, use_container_width=True)

        # summary table
        summary = pd.DataFrame({
            "Metric": ["Min", "Max", "Final"],
            ticker: [plot_df[ticker].min(), plot_df[ticker].max(), plot_df[ticker].iloc[-1]],
            comparison_ticker: [
                plot_df[comparison_ticker].min(),
                plot_df[comparison_ticker].max(),
                plot_df[comparison_ticker].iloc[-1]
            ]
        })
        st.dataframe(summary)











