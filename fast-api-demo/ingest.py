import yfinance as yf
from supabase import create_client
from datetime import datetime, timedelta

# typically we would save these credentials in your .env file for security
supabase = create_client("https://afwcwevryixsjkzerzuu.supabase.co", "")


def ingest_stock_data(symbol: str):
    # 1. Setup Dates
    end_date = datetime.now()
    start_date = end_date - timedelta(days=30)

    # 2. Fetch from yfinance
    ticker = yf.Ticker(symbol)
    df = ticker.history(start=start_date, end=end_date)

    if not df.empty:
        initial_p = float(df['Close'].iloc[0])
        final_p = float(df['Close'].iloc[-1])

        # 3. TASK: Save to Supabase table 'stock_records'
        # Do NOT calculate the signal here. Just save the raw prices.
        data = {
            "ticker": symbol.upper(),
            "initial_price": initial_p,
            "final_price": final_p
        }
        supabase.table("stock_records").insert(data).execute()
        print(f"Successfully ingested {symbol}")


# Run it for a few stocks
ingest_stock_data("AAPL")
ingest_stock_data("TSLA")
