import datetime as dt
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import numpy as np
import pandas as pd
import seaborn as sb
import yfinance as yf  # Assuming yfinance for data API

sb.set_theme()

"""
STUDENT CHANGE LOG & AI DISCLOSURE:
----------------------------------
1. Did you use an LLM (ChatGPT/Claude/etc.)? [Yes/No]
2. If yes, what was your primary prompt?
----------------------------------
"""

DEFAULT_START = dt.date.isoformat(dt.date.today() - dt.timedelta(365))
DEFAULT_END = dt.date.isoformat(dt.date.today())


class Stock:
    def __init__(self, symbol, start=DEFAULT_START, end=DEFAULT_END):
        """
        symbol: stock ticker symbol (e.g., "AAPL")
        start/end: ISO date strings "YYYY-MM-DD"
        """
        self.symbol = symbol
        self.start = start
        self.end = end
        self.data = self.get_data()

    def get_data(self):
        """Downloads data from yfinance and triggers return calculation."""
        # TODO: Use yf.download(self.symbol, start=self.start, end=self.end)
        # data = ...

        # self.calc_returns(data)
        # return data

        start_dt = pd.to_datetime(self.start)
        # yfinance end is exclusive, so add 1 day to include end date
        end_dt = pd.to_datetime(self.end) + pd.Timedelta(days=1)

        data = yf.download(
            tickers = self.symbol,
            start = start_dt,
            end = end_dt,
            auto_adjust=False,
            progress=False,
            threads=True,
        )
        if data is None or data.empty:
            raise ValueError(f"No data found for '{self.symbol}' in range {self.start} to {self.end}.")

        # Ensure the index is datetime and named "Date"
        data.index = pd.to_datetime(data.index)
        data.index.name = "Date"

        # Enrich with returns
        self.calc_returns(data)

        self.data = data
        return data

    def calc_returns(self, df):
        """Adds 'Change', close to close and 'Instant_Return' columns to the dataframe."""
        # Requirement: Use vectorized pandas operations, not loops.
        """
        Helper called by get_data(). Enrich df (in place) with:
        - change: Close-to-close difference relative to the previous day close
        - instant_return: daily instantaneous rate of return
            np.log([closing_price]).diff().round(4)
        """
        if "Close" not in df.columns:
            raise KeyError("Expected a 'Close' column in the downloaded data.")

        df["change"] = df["Close"].diff().round(4)
        df["instant_return"] = np.log(df["Close"]).diff().round(4)
        return df

    def add_technical_indicators(self, windows=[20, 50]):
        """
        Add Simple Moving Averages (SMA) for the given windows
        to the internal DataFrame. Produce a plot showing the closing price and SMAs. 
        """
        if self.data is None or self.data.empty:
            raise ValueError("No data available. Run get_data() first.")

        df = self.data

        windows = [int(w) for w in windows]
        for w in windows:
            df[f"sma_{w}"] = df["Close"].rolling(window=w).mean()

        # Plot Close + SMAs
        plt.figure(figsize=(12, 6))
        plt.plot(df.index, df["Close"], label="Close")
        for w in windows:
            plt.plot(df.index, df[f"sma_{w}"], label=f"SMA {w}")

        ax = plt.gca()
        ax.yaxis.set_major_formatter(mtick.StrMethodFormatter("${x:,.0f}"))
        plt.title(f"{self.symbol} Close Price with Moving Averages")
        plt.xlabel("Date")
        plt.ylabel("Price")
        plt.legend()
        plt.tight_layout()
        plt.show()

        return df

    def plot_return_dist(self, bins:int = 50):
        """
        Plot a well formatted histogram of instantaneous returns.
        """
        if self.data is None or self.data.empty:
            raise ValueError("No data available. Run get_data() first.")
        if "instant_return" not in self.data.columns:
            raise KeyError("Missing 'instant_return'. Run get_data() first.")

        rets = self.data["instant_return"].dropna()

        plt.figure(figsize=(10, 5))
        plt.hist(rets, bins=bins, edgecolor="black")

        ax = plt.gca()
        # log returns are decimals; showing as % often helps readability
        ax.xaxis.set_major_formatter(mtick.PercentFormatter(xmax=1.0))

        plt.title(f"{self.symbol} Distribution of Daily Instantaneous Returns")
        plt.xlabel("Instantaneous Return")
        plt.ylabel("Frequency")
        plt.tight_layout()
        plt.show()

    def plot_performance(self):
        """
        Plots cumulative growth of $1 investment.
        Plot stock performance over the range of data collected, as a percent gain/loss.
        """
        if self.data is None or self.data.empty:
            raise ValueError("No data available. Run get_data() first.")

        df = self.data
        base = df["Close"].iloc[0]
        perf = (df["Close"] / base) - 1.0

        plt.figure(figsize=(12, 6))
        plt.plot(df.index, perf)

        ax = plt.gca()
        ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))

        plt.title(f"{self.symbol} Performance (% Gain/Loss)")
        plt.xlabel("Date")
        plt.ylabel("Performance")
        plt.tight_layout()
        plt.show()


def main():
    # Example usage:
    stock = Stock("AAPL")
    stock.plot_performance()
    stock.add_technical_indicators() # SMA
    stock.plot_return_dist() # histogram


if __name__ == "__main__":
    main()