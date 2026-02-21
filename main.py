import torch
import torch.nn as nn
import yfinance as yf
import pandas_ta as ta
import numpy as np
import os
import json
import time
from datetime import datetime, timedelta
from sklearn.preprocessing import StandardScaler

# ==============================
# CONFIG
# ==============================
import pandas as pd
import requests
import io
from torch.utils.data import DataLoader, TensorDataset

# ==============================
# CONFIG & CONSTANTS
# ==============================
WINDOW_SIZE = 20          # Model looks at the last 20 trading days
MODEL_PATH = "stock_model.pth" 
STATE_PATH = "run_state.json"
RUN_HOUR = 16             # 4 PM
RUN_MINUTE = 2            # 4:02 PM
WEEKLY_STATE = "weekly_state.json"
SIGNALS_CSV_PATH = "buy_signals.csv"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"Using device: {DEVICE}")

# --- NEW CONFIG ---
BATCH_SIZE = 64
LR_GLOBAL = 0.001
LR_FINE_TUNE = 0.0001
INPUT_SIZE = 6  # Matches your features list: Log_Ret, RSI, ATR, EMA20, EMA50, Vol_Change


def get_ticker_metadata(ticker):
    """Fetch stock name and market cap for reporting."""
    try:
        info = yf.Ticker(ticker).info
        return {
            "ticker": ticker,
            "stock_name": info.get("shortName") or info.get("longName") or ticker,
            "marketcap": info.get("marketCap") or 0,
        }
    except Exception:
        return {
            "ticker": ticker,
            "stock_name": ticker,
            "marketcap": 0,
        }


def write_buy_signals_csv(signals, path=SIGNALS_CSV_PATH):
    """Write buy signals sorted by market cap to CSV."""
    if not signals:
        pd.DataFrame(columns=["percentage", "stock_name", "ticker", "marketcap"]).to_csv(path, index=False)
        print(f"No buy signals found. Wrote empty CSV to {path}.")
        return

    rows = []
    for signal in signals:
        metadata = get_ticker_metadata(signal["ticker"])
        rows.append({
            "percentage": round(signal["predicted_return"] * 100, 2),
            "stock_name": metadata["stock_name"],
            "ticker": metadata["ticker"],
            "marketcap": metadata["marketcap"],
        })

    report_df = pd.DataFrame(rows).sort_values(by="marketcap", ascending=False)
    report_df.to_csv(path, index=False)
    print(f"Wrote {len(report_df)} buy signals to {path} (sorted by market cap).")

def get_tickers():
    cache_file = "tickers_cache.json"
    
    # Check if cache exists and is fresh (less than 24h old)
    if os.path.exists(cache_file):
        mtime = os.path.getmtime(cache_file)
        if (time.time() - mtime) < 86400:
            with open(cache_file, "r") as f:
                print("Loading tickers from cache...")
                return json.load(f)

    headers = {'User-Agent': 'Mozilla/5.0'}
    all_symbols = []

    # 1. Russell 1000
    try:
        r_resp = requests.get("https://en.wikipedia.org/wiki/Russell_1000_Index", headers=headers)
        r_df = pd.read_html(io.StringIO(r_resp.text))[2] # Usually index 2
        all_symbols.extend(r_df['Symbol'].tolist())
    except Exception as e: print(f"Russell Error: {e}")

    # 2. TSX 60
    try:
        t_resp = requests.get("https://en.wikipedia.org/wiki/S%26P/TSX_60", headers=headers)
        t_df = pd.read_html(io.StringIO(t_resp.text))[0]
        tsx_list = [str(s).replace('.', '-') + ".TO" for s in t_df['Symbol'].tolist()]
        all_symbols.extend(tsx_list)
    except Exception as e: print(f"TSX Error: {e}")

    all_symbols = list(set([s.strip() for s in all_symbols]))
    with open(cache_file, "w") as f:
        json.dump(all_symbols, f)
    
    return all_symbols

# ==============================
# 1. IMPROVED DATA PROCESSING (No Leakage)
# ==============================
def process_dataframe(df):
    """Transforms raw OHLCV data into scaled tensors."""
    if len(df) < 150:
        return None, None, None

    # Technical Indicators
    df['RSI'] = ta.rsi(df['Close'], length=14)
    df['ATR'] = ta.atr(df['High'], df['Low'], df['Close'])
    df['EMA20'] = ta.ema(df['Close'], length=20)
    df['EMA50'] = ta.ema(df['Close'], length=50)
    df['Log_Ret'] = np.log(df['Close'] / df['Close'].shift(1))
    df['Vol_Change'] = df['Volume'].pct_change()
    df['Target'] = df['Close'].shift(-5) / df['Close'] - 1
    
    df = df.replace([np.inf, -np.inf], np.nan).dropna()
    features = ['Log_Ret', 'RSI', 'ATR', 'EMA20', 'EMA50', 'Vol_Change']

    # Split before scaling to prevent leakage
    train_df = df.iloc[:-WINDOW_SIZE].copy()
    predict_df = df.iloc[-WINDOW_SIZE:].copy()

    scaler = StandardScaler()
    X_train_raw = scaler.fit_transform(train_df[features])
    X_pred_raw = scaler.transform(predict_df[features])

    X, y = [], []
    for i in range(WINDOW_SIZE, len(X_train_raw)):
        X.append(X_train_raw[i-WINDOW_SIZE:i])
        y.append(train_df['Target'].iloc[i-1])

    return (
        torch.tensor(np.array(X), dtype=torch.float32),
        torch.tensor(np.array(y), dtype=torch.float32).unsqueeze(1),
        torch.tensor(np.array(X_pred_raw), dtype=torch.float32)
    )

# ==============================
# 2. BATCHED GLOBAL TRAINING
# ==============================
def train_global_model(tickers):
    print(f"Downloading data for {len(tickers)} symbols...")
    # Bulk download is 10x faster than looping yf.Ticker
    raw_data = yf.download(tickers, period="2y", group_by='ticker', threads=True)
    
    all_X, all_y = [], []
    for t in tickers:
        try:
            ticker_df = raw_data[t] if len(tickers) > 1 else raw_data
            X, y, _ = process_dataframe(ticker_df)
            if X is not None:
                all_X.append(X)
                all_y.append(y)
        except Exception: continue

    if len(all_X) == 0:
        raise RuntimeError("No valid training data found.")

    X_total = torch.cat(all_X)
    y_total = torch.cat(all_y)
    
    dataset = TensorDataset(X_total, y_total)
    loader = DataLoader(dataset, batch_size=64, shuffle=True)

    model = PatternScannerLSTM(input_size=INPUT_SIZE).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    loss_fn = nn.MSELoss()

    print("Training Global Model...")
    for epoch in range(3):
        for batch_X, batch_y in loader:
            batch_X = batch_X.to(DEVICE)
            batch_y = batch_y.to(DEVICE)

            optimizer.zero_grad()
            loss = loss_fn(model(batch_X), batch_y)
            loss.backward()
            optimizer.step()
    torch.save(model.state_dict(), MODEL_PATH)
    return model


# ==============================
# MODEL
# ==============================
class PatternScannerLSTM(nn.Module):
    def __init__(self, input_size, hidden_size=64, num_layers=2, dropout=0.3):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size, hidden_size, num_layers,
            batch_first=True, dropout=dropout
        )
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])


# ==============================
# STATE HANDLING (RESUME SAFE)
# ==============================
def load_state():
    if os.path.exists(STATE_PATH):
        with open(STATE_PATH, "r") as f:
            return json.load(f)
    return {"last_ticker": None}

def save_state(ticker):
    with open(STATE_PATH, "w") as f:
        json.dump({"last_ticker": ticker}, f)


# ==============================
# DAILY RUN
# ==============================
def run_daily():
    # Use the global TICKERS list
    if os.path.exists(MODEL_PATH):
        model = PatternScannerLSTM(input_size=INPUT_SIZE).to(DEVICE)
        model.load_state_dict(torch.load(MODEL_PATH))
        print("Loaded global model.")
    else:
        # FIX: Pass TICKERS to the training function
        model = train_global_model(TICKERS)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.0005)
    loss_fn = nn.MSELoss()
    state = load_state()

    resume = state["last_ticker"]
    skipping = resume is not None

    print("Running daily scan...")
    buy_signals = []

    # Save clean global weights once
    base_state = {k: v.clone() for k, v in model.state_dict().items()}

    for t in TICKERS:
        if skipping:
            if t == resume:
                skipping = False
            continue

        print(f"Processing {t}")
        save_state(t)

        try:
            ticker_df = yf.download(t, period="2y", progress=False)
            X, y, X_pred = process_dataframe(ticker_df)
        except Exception as e:
            print(f"Error fetching {t}: {e}")
            continue

        if X is None:
            continue

        X = X.to(DEVICE)
        y = y.to(DEVICE)
        X_pred = X_pred.to(DEVICE)

        # Reset model + optimizer
        model.load_state_dict(base_state)
        optimizer = torch.optim.Adam(model.parameters(), lr=LR_FINE_TUNE)

        # Fine-tune
        model.train()
        for _ in range(2):
            optimizer.zero_grad()
            loss = loss_fn(model(X), y)
            loss.backward()
            optimizer.step()

        # Predict
        model.eval()
        with torch.no_grad():
            pred = model(X_pred.unsqueeze(0)).item()
            if pred > 0.02:
                print(f"--- BUY SIGNAL: {t} | Expected 5D Return: {pred*100:.2f}% ---")
                buy_signals.append({"ticker": t, "predicted_return": pred})

        time.sleep(0.1)

    torch.save(base_state, MODEL_PATH)
    write_buy_signals_csv(buy_signals)
    save_state(None)
    print("Daily run complete.")


# ==============================
# SCHEDULER
# ==============================
def wait_until_market_close():

    last_weekly_run = None

    while True:
        now = datetime.now()

        last_weekly_run = load_weekly_state()

        if now.weekday() == 5:
            week_id = now.isocalendar()[1]
            if last_weekly_run != week_id:
                weekly_global_update(TICKERS)
                save_weekly_state(week_id)
                last_weekly_run = week_id

        # Skip Saturday (5) and Sunday (6)
        if now.weekday() >= 5:
            print("It's the weekend. Sleeping until Monday...")
            time.sleep(3600) # Check every hour
            continue
            
        run_time = now.replace(hour=RUN_HOUR, minute=RUN_MINUTE, second=0, microsecond=0)
        if now >= run_time:
            run_time += timedelta(days=1)

        sleep_seconds = (run_time - now).total_seconds()
        print(f"Next run at {run_time}")
        time.sleep(sleep_seconds)
        run_daily()

def load_weekly_state():
    if os.path.exists(WEEKLY_STATE):
        with open(WEEKLY_STATE) as f:
            return json.load(f).get("week")
    return None

def save_weekly_state(week):
    with open(WEEKLY_STATE, "w") as f:
        json.dump({"week": week}, f)


def weekly_global_update(tickers, lookback_days=180):
    print("🔄 Weekly global model update (last 6 months)...")

    if not os.path.exists(MODEL_PATH):
        print("No existing model found — skipping weekly update.")
        return

    # Load existing global model
    model = PatternScannerLSTM(input_size=INPUT_SIZE).to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH))
    model.train()

    # 🔥 Small learning rate = gentle adaptation
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LR_GLOBAL * 0.1,   # e.g. 1e-4
        weight_decay=1e-5
    )
    loss_fn = nn.MSELoss()

    start_date = (datetime.now() - timedelta(days=lookback_days)).strftime("%Y-%m-%d")

    raw_data = yf.download(
        tickers,
        start=start_date,
        group_by="ticker",
        threads=True,
        progress=False
    )

    all_X, all_y = [], []

    for t in tickers:
        try:
            df = raw_data[t] if len(tickers) > 1 else raw_data
            X, y, _ = process_dataframe(df)
            if X is not None:
                all_X.append(X)
                all_y.append(y)
        except Exception:
            continue

    if not all_X:
        print("⚠️ No valid data for weekly update.")
        return

    X = torch.cat(all_X).to(DEVICE)
    y = torch.cat(all_y).to(DEVICE)

    # 🔥 ONE epoch only
    optimizer.zero_grad()
    loss = loss_fn(model(X), y)
    loss.backward()
    optimizer.step()

    torch.save(model.state_dict(), MODEL_PATH)
    print("✅ Weekly global update complete.")

# ==============================
# MAIN
# ==============================
if __name__ == "__main__":
    try:
        TICKERS = get_tickers()
        wait_until_market_close()
    except KeyboardInterrupt:
        print("\nInterrupted. State saved. Will resume next run.")
