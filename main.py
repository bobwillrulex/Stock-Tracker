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
WINDOW_SIZE = 20
MODEL_PATH = "global_model.pth"
STATE_PATH = "run_state.json"
RUN_HOUR = 16
RUN_MINUTE = 2   # 2 minutes after close
DEVICE = "cpu"

def get_tickers():
    headers = {'User-Agent': 'Mozilla/5.0'}
    all_symbols = []

    # 1. Russell 1000
    try:
        print("Fetching Russell 1000...")
        r_resp = requests.get("https://en.wikipedia.org/wiki/Russell_1000_Index", headers=headers)
        r_tables = pd.read_html(io.StringIO(r_resp.text))
        r_df = max(r_tables, key=len)
        col = [c for c in r_df.columns if any(k in str(c) for k in ['Symbol', 'Ticker'])][0]
        all_symbols.extend(r_df[col].tolist())
    except Exception as e: print(f"Russell Error: {e}")

    # 2. TSX 60 (FIXED LOGIC)
    try:
        print("Fetching TSX 60...")
        t_resp = requests.get("https://en.wikipedia.org/wiki/S%26P/TSX_60", headers=headers)
        t_tables = pd.read_html(io.StringIO(t_resp.text))
        # Usually the first table, but we search for the 'Symbol' column specifically
        t_df = t_tables[0]
        col = [c for c in t_df.columns if any(k in str(c) for k in ['Symbol', 'Ticker'])][0]
        tsx_list = [str(s).strip().replace('.', '-') + ".TO" for s in t_df[col].tolist()]
        all_symbols.extend(tsx_list)
    except Exception as e: print(f"TSX Error: {e}")

    return list(set(all_symbols))

# --- MAIN SCAN LOOP ---

TICKERS = get_tickers()


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
# DATA
# ==============================
def get_stock_data(ticker):
    stock = yf.Ticker(ticker)
    df = stock.history(period="2y")
    if len(df) < 150:
        return None, None, None

    df['RSI'] = ta.rsi(df['Close'], length=14)
    df['ATR'] = ta.atr(df['High'], df['Low'], df['Close'])
    df['EMA20'] = ta.ema(df['Close'], length=20)
    df['EMA50'] = ta.ema(df['Close'], length=50)
    df['Volume_Change'] = df['Volume'].pct_change()
    df = df.dropna()

    features = [
        'Open','High','Low','Close',
        'RSI','ATR','EMA20','EMA50','Volume_Change'
    ]

    train_df = df.iloc[:-5]
    predict_df = df.iloc[-WINDOW_SIZE:]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(train_df[features])
    X_pred = scaler.transform(predict_df[features])

    X, y = [], []
    for i in range(WINDOW_SIZE, len(X_scaled)):
        X.append(X_scaled[i-WINDOW_SIZE:i])
        y.append(train_df['Close'].iloc[i+5] / train_df['Close'].iloc[i] - 1)

    return (
        torch.tensor(X, dtype=torch.float32),
        torch.tensor(y, dtype=torch.float32).unsqueeze(1),
        torch.tensor(X_pred, dtype=torch.float32)
    )

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
# TRAIN GLOBAL MODEL
# ==============================
def train_global_model():
    print("Training global model...")
    model = PatternScannerLSTM(input_size=9).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    loss_fn = nn.MSELoss()

    for t in TICKERS:
        data = get_stock_data(t)
        if data[0] is None:
            continue
        X, y, _ = data
        for _ in range(2):  # light global pass
            optimizer.zero_grad()
            loss = loss_fn(model(X), y)
            loss.backward()
            optimizer.step()

    torch.save(model.state_dict(), MODEL_PATH)
    print("Global model saved.")
    return model

# ==============================
# DAILY RUN
# ==============================
def run_daily():
    if os.path.exists(MODEL_PATH):
        model = PatternScannerLSTM(input_size=9)
        model.load_state_dict(torch.load(MODEL_PATH))
        print("Loaded global model.")
    else:
        model = train_global_model()

    optimizer = torch.optim.Adam(model.parameters(), lr=0.0005)
    loss_fn = nn.MSELoss()
    state = load_state()

    resume = state["last_ticker"]
    skipping = resume is not None

    print("Running daily scan...")
    for t in TICKERS:
        if skipping:
            if t == resume:
                skipping = False
            continue

        print(f"Processing {t}")
        save_state(t)

        data = get_stock_data(t)
        if data[0] is None:
            continue

        X, y, X_pred = data

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
                print(f"BUY SIGNAL: {t} | Expected 5D Return: {pred*100:.2f}%")

    torch.save(model.state_dict(), MODEL_PATH)
    save_state(None)
    print("Daily run complete.")

# ==============================
# SCHEDULER
# ==============================
def wait_until_market_close():
    while True:
        now = datetime.now()
        run_time = now.replace(
            hour=RUN_HOUR, minute=RUN_MINUTE, second=0, microsecond=0
        )

        if now >= run_time:
            run_time += timedelta(days=1)

        sleep_seconds = (run_time - now).total_seconds()
        print(f"Next run at {run_time}")
        time.sleep(sleep_seconds)
        run_daily()

# ==============================
# MAIN
# ==============================
if __name__ == "__main__":
    try:
        wait_until_market_close()
    except KeyboardInterrupt:
        print("\nInterrupted. State saved. Will resume next run.")