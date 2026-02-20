def get_ticker_metadata(ticker):
    try:
        info = yf.Ticker(ticker).info
        return {
            "symbol": ticker,
            "name": info.get("shortName") or info.get("longName") or ticker,
            "marketcap": info.get("marketCap", 0)
        }
    except Exception:
        return {
            "symbol": ticker,
            "name": ticker,
            "marketcap": 0
        }

