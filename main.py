import torch
import torch.nn as nn
import yfinance as yf
import numpy as np
import os
import json
from json import JSONDecodeError
import time
from datetime import datetime, timedelta
from sklearn.preprocessing import StandardScaler

# ==============================
# CONFIG
# ==============================
import pandas as pd
import requests
import io
from bs4 import BeautifulSoup
from torch.utils.data import DataLoader, TensorDataset

# ==============================
# CONFIG & CONSTANTS
# ==============================
WINDOW_SIZE = 60          # Model looks at the last 60 trading days
MODEL_PATH = "stock_model.pth" 
BEST_MODEL_PATH = "stock_model_best.pth"
STATE_PATH = "run_state.json"
RUN_HOUR = 16             # 4 PM
RUN_MINUTE = 2            # 4:02 PM
WEEKLY_STATE = "weekly_state.json"
SIGNALS_CSV_PATH = "buy_signals.csv"
MACD_SIGNALS_CSV_PATH = "macd_signals.csv"
RSI_SIGNALS_CSV_PATH = "rsi_signals.csv"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"Using device: {DEVICE}")

# --- NEW CONFIG ---
BATCH_SIZE = 64
LR_GLOBAL = 0.001
LR_FINE_TUNE = 0.0001
GLOBAL_EPOCHS = 30
WEEKLY_EPOCHS = 8
PATIENCE = 5
WEIGHT_DECAY = 1e-5
GRAD_CLIP = 1.0
VAL_SPLIT = 0.2

FEATURE_COLUMNS = [
    'Log_Ret', 'RSI', 'ATR', 'EMA20', 'EMA50', 'Vol_Change',
    'MACD', 'MACD_SIGNAL', 'MACD_HIST',
    'Support_20', 'Resistance_20',
    'Fib_236', 'Fib_382', 'Fib_500', 'Fib_618',
]
INPUT_SIZE = len(FEATURE_COLUMNS)


def _normalize_ohlcv_dataframe(df):
    """Return a copy of OHLCV data with flat, 1D columns.

    yfinance can return multi-index columns (or single-column DataFrames when selecting
    e.g. `df['Close']`). Downstream feature engineering expects each OHLCV field to be a
    pandas Series, so we squeeze those to 1D here.
    """
    if df is None or len(df) == 0:
        return df

    frame = df.copy()

    if isinstance(frame.columns, pd.MultiIndex):
        if frame.columns.nlevels >= 2:
            # Keep the OHLCV field level when available.
            frame.columns = frame.columns.get_level_values(0)
        else:
            frame.columns = frame.columns.get_level_values(-1)

    required_cols = ["Open", "High", "Low", "Close", "Volume"]
    for col in required_cols:
        if col in frame.columns and isinstance(frame[col], pd.DataFrame):
            frame[col] = frame[col].iloc[:, 0]

    return frame


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
        pd.DataFrame(columns=["percentage", "confidence", "stock_name", "ticker", "marketcap"]).to_csv(path, index=False)
        print(f"No buy signals found. Wrote empty CSV to {path}.")
        return

    rows = []
    for signal in signals:
        metadata = get_ticker_metadata(signal["ticker"])
        rows.append({
            "percentage": round(signal["predicted_return"] * 100, 2),
            "confidence": round(signal.get("confidence", 0.0) * 100, 2),
            "stock_name": metadata["stock_name"],
            "ticker": metadata["ticker"],
            "marketcap": metadata["marketcap"],
        })

    report_df = pd.DataFrame(rows).sort_values(by="marketcap", ascending=False)
    report_df.to_csv(path, index=False)
    print(f"Wrote {len(report_df)} buy signals to {path} (sorted by market cap).")


def classify_macd_signal(close_series):
    """Return MACD signal category for the latest bar.

    1 = just crossed above 0
    2 = two bars above 0 after crossing
    3 = below 0 but rounding upward
    0 = no signal
    """
    if close_series is None or len(close_series) < 30:
        return 0

    ema_fast = close_series.ewm(span=12, adjust=False).mean()
    ema_slow = close_series.ewm(span=26, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    macd_signal = macd_line.ewm(span=9, adjust=False).mean()
    hist = macd_line - macd_signal

    if len(hist) < 4:
        return 0

    t0 = float(hist.iloc[-1])
    t1 = float(hist.iloc[-2])
    t2 = float(hist.iloc[-3])

    if t1 < 0 < t0:
        return 1
    if t2 < 0 and t1 > 0 and t0 > 0:
        return 2
    if -0.10 < t0 < 0 and t0 > t1:
        return 3
    return 0


def classify_rsi_signal(close_series, period=14):
    """Return RSI signal category for the latest bar.

    1 = oversold (<30)
    2 = almost oversold (30-35 and falling)
    3 = recovering (was <30, now >=30)
    0 = no signal
    """
    if close_series is None or len(close_series) < period + 2:
        return 0, None

    delta = close_series.diff()
    gain = delta.clip(lower=0).rolling(window=period).mean()
    loss = (-delta.clip(upper=0)).rolling(window=period).mean()
    rs = gain / loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))

    if len(rsi.dropna()) < 2:
        return 0, None

    current_rsi = float(rsi.iloc[-1])
    prev_rsi = float(rsi.iloc[-2])

    if current_rsi < 30:
        return 1, current_rsi
    if 30 <= current_rsi <= 35 and current_rsi < prev_rsi:
        return 2, current_rsi
    if prev_rsi < 30 <= current_rsi:
        return 3, current_rsi
    return 0, current_rsi


def write_technical_signals_csv(macd_signals, rsi_signals):
    """Persist flat MACD/RSI technical signal lists for UI tabs."""
    macd_cols = ["signal_type", "stock_name", "ticker", "price", "change_pct", "marketcap"]
    rsi_cols = ["signal_type", "stock_name", "ticker", "rsi", "marketcap"]

    macd_df = pd.DataFrame(macd_signals, columns=macd_cols)
    rsi_df = pd.DataFrame(rsi_signals, columns=rsi_cols)

    if not macd_df.empty:
        macd_df = macd_df.sort_values(by="marketcap", ascending=False)
    if not rsi_df.empty:
        rsi_df = rsi_df.sort_values(by="marketcap", ascending=False)

    macd_df.to_csv(MACD_SIGNALS_CSV_PATH, index=False)
    rsi_df.to_csv(RSI_SIGNALS_CSV_PATH, index=False)

    print(f"Wrote {len(macd_df)} MACD signals to {MACD_SIGNALS_CSV_PATH}.")
    print(f"Wrote {len(rsi_df)} RSI signals to {RSI_SIGNALS_CSV_PATH}.")


def compute_confidence_score(model, X_val, y_val, pred):
    """Estimate model confidence (0-1) using recent validation MAE + signal strength.

    Lower validation error and stronger positive predicted return both increase confidence.
    """
    if X_val is None or y_val is None or len(X_val) == 0:
        return 0.0

    model.eval()
    with torch.no_grad():
        val_pred = model(X_val.to(DEVICE))
        mae = torch.mean(torch.abs(val_pred - y_val.to(DEVICE))).item()

    # MAE in return-space is usually small; map to 0-1 with a conservative decay.
    base_confidence = float(np.exp(-mae / 0.03))
    # Reward stronger expected upside, capped at +10% return.
    strength_bonus = min(max(pred, 0.0), 0.10) / 0.10
    confidence = 0.75 * base_confidence + 0.25 * strength_bonus
    return float(np.clip(confidence, 0.0, 1.0))

def get_tickers():
    cache_file = "tickers_cache.json"
    min_expected_symbols = 100

    def load_cached_tickers():
        if not os.path.exists(cache_file):
            return None

        mtime = os.path.getmtime(cache_file)
        is_fresh = (time.time() - mtime) < 86400
        if not is_fresh:
            return None

        try:
            with open(cache_file, "r") as f:
                cached = json.load(f)
        except (OSError, JSONDecodeError):
            print("Ticker cache is unreadable. Rebuilding cache from online sources...")
            return None

        if not isinstance(cached, list):
            return None

        cached = [str(s).strip().upper() for s in cached if str(s).strip()]
        if len(cached) < min_expected_symbols:
            print(
                f"Ticker cache only has {len(cached)} symbols; refreshing from sources..."
            )
            return None

        print("Loading tickers from cache...")
        return sorted(set(cached))

    def get_symbol_column(df):
        for col in df.columns:
            normalized = str(col).strip().lower()
            if "symbol" in normalized or "ticker" in normalized:
                return col
        return None
    
    cached_tickers = load_cached_tickers()
    if cached_tickers is not None:
        return cached_tickers

    headers = {'User-Agent': 'Mozilla/5.0'}
    all_symbols = []

    # 1. Russell 1000
    try:
        print("Fetching Russell 1000...")
        r_resp = requests.get("https://en.wikipedia.org/wiki/Russell_1000_Index", headers=headers)
        r_resp.raise_for_status()
        r_tables = pd.read_html(io.StringIO(r_resp.text))
        for table in r_tables:
            col = get_symbol_column(table)
            if col:
                all_symbols.extend(table[col].tolist())
                break
    except Exception as e: print(f"Russell Error: {e}")

    # 2. TSX 60
    try:
        print("Fetching TSX 60...")
        t_resp = requests.get("https://en.wikipedia.org/wiki/S%26P/TSX_60", headers=headers)
        t_resp.raise_for_status()
        t_tables = pd.read_html(io.StringIO(t_resp.text))
        for table in t_tables:
            col = get_symbol_column(table)
            if col:
                tsx_list = [str(s).strip().replace('.', '-') + ".TO" for s in table[col].tolist()]
                all_symbols.extend(tsx_list)
                break
    except Exception as e: print(f"TSX Error: {e}")

    # 3. Yahoo Trending
    try:
        print("Fetching Yahoo Trending...")
        y_resp = requests.get("https://finance.yahoo.com/markets/stocks/trending/", headers=headers)
        y_resp.raise_for_status()
        soup = BeautifulSoup(y_resp.text, 'html.parser')
        trending = [
            el['data-symbol']
            for el in soup.find_all(attrs={"data-symbol": True})
            if not el['data-symbol'].isdigit()
        ]
        all_symbols.extend(trending)
    except Exception as e:
        print(f"Trending Error: {e}")

    all_symbols = sorted(
        set(
            s.strip().upper()
            for s in all_symbols
            if isinstance(s, str) and s.strip() and " " not in s
        )
    )

    if len(all_symbols) < min_expected_symbols:
        raise RuntimeError(
            f"Ticker fetch returned only {len(all_symbols)} symbols. Aborting to avoid training on an invalid universe."
        )

    with open(cache_file, "w") as f:
        json.dump(all_symbols, f)

    return all_symbols

# ==============================
# 1. IMPROVED DATA PROCESSING (No Leakage)
# ==============================
def process_dataframe(df, val_split=VAL_SPLIT):
    """Transforms raw OHLCV data into train/val tensors and prediction window."""
    df = _normalize_ohlcv_dataframe(df)

    required_cols = {"High", "Low", "Close", "Volume"}
    if df is None or not required_cols.issubset(set(df.columns)):
        return None, None, None, None, None

    if len(df) < 220:
        return None, None, None, None, None

    # Technical Indicators
    close_delta = df['Close'].diff()
    gains = close_delta.where(close_delta > 0, 0.0)
    losses = -close_delta.where(close_delta < 0, 0.0)
    avg_gain = gains.ewm(alpha=1 / 14, min_periods=14, adjust=False).mean()
    avg_loss = losses.ewm(alpha=1 / 14, min_periods=14, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    df['RSI'] = 100 - (100 / (1 + rs))

    high_low = df['High'] - df['Low']
    high_close_prev = (df['High'] - df['Close'].shift(1)).abs()
    low_close_prev = (df['Low'] - df['Close'].shift(1)).abs()
    true_range = np.maximum.reduce([high_low, high_close_prev, low_close_prev])
    df['ATR'] = pd.Series(true_range, index=df.index).ewm(alpha=1 / 14, min_periods=14, adjust=False).mean()

    df['EMA20'] = df['Close'].ewm(span=20, adjust=False).mean()
    df['EMA50'] = df['Close'].ewm(span=50, adjust=False).mean()
    df['Log_Ret'] = np.log(df['Close'] / df['Close'].shift(1))
    df['Vol_Change'] = df['Volume'].pct_change()

    # MACD family (12, 26, 9)
    ema_fast = df['Close'].ewm(span=12, adjust=False).mean()
    ema_slow = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = ema_fast - ema_slow
    df['MACD_SIGNAL'] = df['MACD'].ewm(span=9, adjust=False).mean()
    df['MACD_HIST'] = df['MACD'] - df['MACD_SIGNAL']

    # Dynamic support/resistance using rolling lows/highs
    rolling_low = df['Low'].rolling(window=20, min_periods=20).min()
    rolling_high = df['High'].rolling(window=20, min_periods=20).max()
    df['Support_20'] = (df['Close'] - rolling_low) / df['Close']
    df['Resistance_20'] = (rolling_high - df['Close']) / df['Close']

    # Fibonacci retracement ratios from rolling swing range
    swing_range = (rolling_high - rolling_low).replace(0, np.nan)
    df['Fib_236'] = (df['Close'] - (rolling_high - swing_range * 0.236)) / df['Close']
    df['Fib_382'] = (df['Close'] - (rolling_high - swing_range * 0.382)) / df['Close']
    df['Fib_500'] = (df['Close'] - (rolling_high - swing_range * 0.5)) / df['Close']
    df['Fib_618'] = (df['Close'] - (rolling_high - swing_range * 0.618)) / df['Close']

    df['Target'] = df['Close'].shift(-5) / df['Close'] - 1
    
    df = df.replace([np.inf, -np.inf], np.nan).dropna()
    features = FEATURE_COLUMNS

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

    if len(X) < 30:
        return None, None, None, None, None

    split_idx = int(len(X) * (1 - val_split))
    split_idx = max(split_idx, 1)
    split_idx = min(split_idx, len(X) - 1)

    X_train = torch.tensor(np.array(X[:split_idx]), dtype=torch.float32)
    y_train = torch.tensor(np.array(y[:split_idx]), dtype=torch.float32).unsqueeze(1)
    X_val = torch.tensor(np.array(X[split_idx:]), dtype=torch.float32)
    y_val = torch.tensor(np.array(y[split_idx:]), dtype=torch.float32).unsqueeze(1)

    return (
        X_train,
        y_train,
        X_val,
        y_val,
        torch.tensor(np.array(X_pred_raw), dtype=torch.float32)
    )


def run_epoch(model, loader, optimizer, loss_fn, scaler=None):
    model.train()
    total_loss = 0.0
    total_count = 0

    for batch_X, batch_y in loader:
        batch_X = batch_X.to(DEVICE)
        batch_y = batch_y.to(DEVICE)
        optimizer.zero_grad()

        with torch.autocast(device_type=DEVICE.type, dtype=torch.float16, enabled=(DEVICE.type == "cuda")):
            loss = loss_fn(model(batch_X), batch_y)

        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            optimizer.step()

        total_loss += loss.item() * len(batch_X)
        total_count += len(batch_X)

    return total_loss / max(total_count, 1)


def evaluate(model, loader, loss_fn):
    model.eval()
    total_loss = 0.0
    total_count = 0
    with torch.no_grad():
        for batch_X, batch_y in loader:
            batch_X = batch_X.to(DEVICE)
            batch_y = batch_y.to(DEVICE)
            loss = loss_fn(model(batch_X), batch_y)
            total_loss += loss.item() * len(batch_X)
            total_count += len(batch_X)
    return total_loss / max(total_count, 1)

# ==============================
# 2. BATCHED GLOBAL TRAINING
# ==============================
def train_global_model(tickers, progress_callback=None):
    def emit_progress(stage, current, total, message):
        if progress_callback:
            progress_callback({
                "stage": stage,
                "current": current,
                "total": total,
                "message": message,
            })

    emit_progress("download", 0, max(len(tickers), 1), "Downloading historical data...")
    print(f"Downloading data for {len(tickers)} symbols...")
    raw_data = yf.download(tickers, period="2y", group_by='ticker', threads=True)

    train_X, train_y, val_X, val_y = [], [], [], []
    total_tickers = len(tickers)
    for idx, t in enumerate(tickers, start=1):
        try:
            ticker_df = raw_data[t] if len(tickers) > 1 else raw_data
            X_train, y_train, X_val, y_val, _ = process_dataframe(ticker_df)
            if X_train is not None:
                train_X.append(X_train)
                train_y.append(y_train)
                val_X.append(X_val)
                val_y.append(y_val)
        except Exception:
            continue
        emit_progress("prepare", idx, max(total_tickers, 1), f"Preparing training set ({idx}/{total_tickers}): {t}")

    if len(train_X) == 0:
        raise RuntimeError("No valid training data found.")

    X_train_total = torch.cat(train_X)
    y_train_total = torch.cat(train_y)
    X_val_total = torch.cat(val_X)
    y_val_total = torch.cat(val_y)

    train_loader = DataLoader(TensorDataset(X_train_total, y_train_total), batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(TensorDataset(X_val_total, y_val_total), batch_size=BATCH_SIZE, shuffle=False)

    model = PatternScannerLSTM(input_size=INPUT_SIZE).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR_GLOBAL, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2)
    loss_fn = nn.MSELoss()
    scaler = torch.amp.GradScaler(enabled=(DEVICE.type == "cuda"))

    print("Training Global Model...")
    best_val = float('inf')
    best_state = None
    patience_counter = 0

    for epoch in range(GLOBAL_EPOCHS):
        train_loss = run_epoch(model, train_loader, optimizer, loss_fn, scaler=scaler)
        val_loss = evaluate(model, val_loader, loss_fn)
        scheduler.step(val_loss)

        print(f"Epoch {epoch + 1:02d}/{GLOBAL_EPOCHS} - train_loss={train_loss:.6f} val_loss={val_loss:.6f}")
        emit_progress("train", epoch + 1, GLOBAL_EPOCHS, f"Epoch {epoch + 1}/{GLOBAL_EPOCHS} train={train_loss:.6f} val={val_loss:.6f}")

        if val_loss < best_val:
            best_val = val_loss
            patience_counter = 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            torch.save(best_state, BEST_MODEL_PATH)
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                print(f"Early stopping triggered at epoch {epoch + 1}.")
                break

    if best_state is None:
        best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    model.load_state_dict(best_state)
    torch.save(best_state, MODEL_PATH)
    emit_progress("done", 1, 1, "Global model training complete.")
    return model


# ==============================
# MODEL
# ==============================
class PatternScannerLSTM(nn.Module):
    def __init__(self, input_size, hidden_size=128, num_layers=2, dropout=0.25):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size, hidden_size, num_layers,
            batch_first=True, dropout=dropout, bidirectional=True
        )
        self.attention = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1)
        )
        self.norm = nn.LayerNorm(hidden_size * 2)
        self.fc = nn.Linear(hidden_size * 2, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        attn_scores = torch.softmax(self.attention(out), dim=1)
        context = torch.sum(attn_scores * out, dim=1)
        context = self.norm(context)
        return self.fc(context)


# ==============================
# STATE HANDLING (RESUME SAFE)
# ==============================
def load_state():
    if os.path.exists(STATE_PATH):
        try:
            with open(STATE_PATH, "r") as f:
                state = json.load(f)
            if isinstance(state, dict):
                return state
        except (OSError, JSONDecodeError):
            print("run_state.json is empty/corrupt. Starting scan from first ticker.")
    return {"last_ticker": None}

def save_state(ticker):
    with open(STATE_PATH, "w") as f:
        json.dump({"last_ticker": ticker}, f)


# ==============================
# DAILY RUN
# ==============================
def run_daily(tickers=None, progress_callback=None):
    def emit_progress(stage, current, total, message):
        if progress_callback is not None:
            progress_callback({
                "stage": stage,
                "current": current,
                "total": total,
                "message": message,
            })

    if tickers is None:
        tickers = globals().get("TICKERS") or get_tickers()

    total_tickers = max(len(tickers), 1)

    checkpoint = BEST_MODEL_PATH if os.path.exists(BEST_MODEL_PATH) else MODEL_PATH
    if os.path.exists(checkpoint):
        model = PatternScannerLSTM(input_size=INPUT_SIZE).to(DEVICE)
        model.load_state_dict(torch.load(checkpoint, map_location=DEVICE))
        print(f"Loaded global model from {checkpoint}.")
    else:
        model = train_global_model(tickers)

    loss_fn = nn.MSELoss()
    state = load_state()

    resume = state["last_ticker"]
    skipping = resume is not None

    print("Running daily scan...")
    emit_progress("scan", 0, total_tickers, "Starting daily scan...")
    buy_signals = []
    macd_signals = []
    rsi_signals = []

    # Save clean global weights once
    base_state = {k: v.clone() for k, v in model.state_dict().items()}

    processed = 0
    for t in tickers:
        if skipping:
            if t == resume:
                skipping = False
                emit_progress("scan", processed, total_tickers, f"Resuming scan from {t}...")
            continue

        print(f"Processing {t}")
        save_state(t)

        try:
            ticker_df = yf.download(t, period="2y", progress=False)
            X_train, y_train, X_val, y_val, X_pred = process_dataframe(ticker_df, val_split=0.1)
        except Exception as e:
            print(f"Error fetching {t}: {e}")
            processed += 1
            emit_progress("scan", processed, total_tickers, f"Daily scan: {processed}/{total_tickers} ({t})")
            continue

        if X_train is None:
            processed += 1
            emit_progress("scan", processed, total_tickers, f"Daily scan: {processed}/{total_tickers} ({t})")
            continue

        X_pred = X_pred.to(DEVICE)

        # Reset model + optimizer
        model.load_state_dict(base_state)
        optimizer = torch.optim.AdamW(model.parameters(), lr=LR_FINE_TUNE, weight_decay=WEIGHT_DECAY)

        # Fine-tune with mini-batches
        train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=32, shuffle=True)
        for _ in range(3):
            run_epoch(model, train_loader, optimizer, loss_fn, scaler=None)

        # Predict
        model.eval()
        with torch.no_grad():
            pred = model(X_pred.unsqueeze(0)).item()
            if pred > 0.02:
                confidence = compute_confidence_score(model, X_val, y_val, pred)
                print(f"--- BUY SIGNAL: {t} | Expected 5D Return: {pred*100:.2f}% ---")
                buy_signals.append(
                    {
                        "ticker": t,
                        "predicted_return": pred,
                        "confidence": confidence,
                    }
                )

        close_series = ticker_df.get("Close")
        if close_series is not None and len(close_series) > 30:
            macd_type = classify_macd_signal(close_series)
            rsi_type, rsi_value = classify_rsi_signal(close_series)

            if macd_type in (1, 2, 3):
                metadata = get_ticker_metadata(t)
                close_today = float(close_series.iloc[-1])
                close_prev = float(close_series.iloc[-2])
                change_pct = ((close_today - close_prev) / close_prev) * 100 if close_prev else 0.0
                macd_signals.append(
                    {
                        "signal_type": macd_type,
                        "stock_name": metadata["stock_name"],
                        "ticker": t,
                        "price": round(close_today, 2),
                        "change_pct": round(change_pct, 2),
                        "marketcap": metadata["marketcap"],
                    }
                )

            if rsi_type in (1, 2, 3):
                metadata = get_ticker_metadata(t)
                rsi_signals.append(
                    {
                        "signal_type": rsi_type,
                        "stock_name": metadata["stock_name"],
                        "ticker": t,
                        "rsi": round(float(rsi_value), 2) if rsi_value is not None else 0.0,
                        "marketcap": metadata["marketcap"],
                    }
                )

        time.sleep(0.1)
        processed += 1
        emit_progress("scan", processed, total_tickers, f"Daily scan: {processed}/{total_tickers} ({t})")

    torch.save(base_state, MODEL_PATH)
    write_buy_signals_csv(buy_signals)
    write_technical_signals_csv(macd_signals, rsi_signals)
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
        try:
            with open(WEEKLY_STATE) as f:
                payload = json.load(f)
            if isinstance(payload, dict):
                return payload.get("week")
        except (OSError, JSONDecodeError):
            print("weekly_state.json is empty/corrupt. Weekly scheduler state reset.")
    return None

def save_weekly_state(week):
    with open(WEEKLY_STATE, "w") as f:
        json.dump({"week": week}, f)


def weekly_global_update(tickers, lookback_days=180):
    print("🔄 Weekly global model update (last 6 months)...")

    checkpoint = BEST_MODEL_PATH if os.path.exists(BEST_MODEL_PATH) else MODEL_PATH
    if not os.path.exists(checkpoint):
        print("No existing model found — skipping weekly update.")
        return

    model = PatternScannerLSTM(input_size=INPUT_SIZE).to(DEVICE)
    model.load_state_dict(torch.load(checkpoint, map_location=DEVICE))

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LR_FINE_TUNE,
        weight_decay=WEIGHT_DECAY,
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2)
    loss_fn = nn.MSELoss()

    start_date = (datetime.now() - timedelta(days=lookback_days)).strftime("%Y-%m-%d")

    raw_data = yf.download(
        tickers,
        start=start_date,
        group_by="ticker",
        threads=True,
        progress=False
    )

    train_X, train_y, val_X, val_y = [], [], [], []

    for t in tickers:
        try:
            df = raw_data[t] if len(tickers) > 1 else raw_data
            X_train, y_train, X_val, y_val, _ = process_dataframe(df, val_split=0.15)
            if X_train is not None:
                train_X.append(X_train)
                train_y.append(y_train)
                val_X.append(X_val)
                val_y.append(y_val)
        except Exception:
            continue

    if not train_X:
        print("⚠️ No valid data for weekly update.")
        return

    train_loader = DataLoader(TensorDataset(torch.cat(train_X), torch.cat(train_y)), batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(TensorDataset(torch.cat(val_X), torch.cat(val_y)), batch_size=BATCH_SIZE, shuffle=False)

    best_val = float('inf')
    best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    patience_counter = 0

    for epoch in range(WEEKLY_EPOCHS):
        train_loss = run_epoch(model, train_loader, optimizer, loss_fn, scaler=None)
        val_loss = evaluate(model, val_loader, loss_fn)
        scheduler.step(val_loss)
        print(f"Weekly epoch {epoch + 1}/{WEEKLY_EPOCHS} - train_loss={train_loss:.6f} val_loss={val_loss:.6f}")

        if val_loss < best_val:
            best_val = val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= 3:
                print("Weekly update early stopped.")
                break

    torch.save(best_state, BEST_MODEL_PATH)
    torch.save(best_state, MODEL_PATH)
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
