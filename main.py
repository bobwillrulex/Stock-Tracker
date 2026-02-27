import torch
import torch.nn as nn
import yfinance as yf
import numpy as np
import os
import json
from json import JSONDecodeError
import time
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo
import sqlite3
import random
import hashlib
import html
import subprocess
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
MARKET_FORECAST_CSV_PATH = "sp500_forecast.csv"
MARKET_FORECAST_DB_PATH = "sp500_forecast.db"
LIVE_SIGNAL_DB_PATH = "live_trading_signals.db"
STOCK_DETAIL_CACHE_DB_PATH = "stock_details_cache.db"
WEB_REPORT_CACHE_DB_PATH = "web_report_cache.db"
PAGES_DIR = "docs"
PAGES_INDEX_PATH = os.path.join(PAGES_DIR, "index.html")
PAGES_PORTFOLIO_PATH = os.path.join(PAGES_DIR, "portfolio.html")
PAGES_NOJEKYLL_PATH = os.path.join(PAGES_DIR, ".nojekyll")
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
    'Weekly_Support_Dist', 'Weekly_Resistance_Dist',
    'Weekly_Support_Strength', 'Weekly_Resistance_Strength',
    'SR_Confluence',
]
INPUT_SIZE = len(FEATURE_COLUMNS)


def _stable_seed(*parts):
    """Create a deterministic 32-bit seed from arbitrary values."""
    token = "::".join(str(p) for p in parts)
    digest = hashlib.sha256(token.encode("utf-8")).hexdigest()
    return int(digest[:8], 16)


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


def _coerce_to_series(values):
    """Return a 1D numeric Series for price-like inputs.

    Handles DataFrames produced by yfinance multi-index outputs by selecting the
    first column. Returns ``None`` when coercion isn't possible.
    """
    if values is None:
        return None

    if isinstance(values, pd.DataFrame):
        if values.shape[1] == 0:
            return None
        values = values.iloc[:, 0]

    if not isinstance(values, pd.Series):
        try:
            values = pd.Series(values)
        except Exception:
            return None

    return pd.to_numeric(values, errors="coerce")

def _touch_density(close_series, level_series, atr_series, lookback=20, tolerance=0.75):
    """Return rolling touch density for a level, normalized to [0,1]."""
    valid = (
        close_series.notna()
        & level_series.notna()
        & atr_series.notna()
        & (atr_series > 0)
    )
    touch = pd.Series(0.0, index=close_series.index)
    touch.loc[valid] = (
        (close_series.loc[valid] - level_series.loc[valid]).abs()
        <= (atr_series.loc[valid] * tolerance)
    ).astype(float)
    return touch.rolling(window=lookback, min_periods=lookback).mean()


def add_multitimeframe_sr_features(df):
    """Add AI-friendly support/resistance strength features using weekly context."""
    if df is None or len(df) == 0:
        return df

    frame = df.copy()
    if not isinstance(frame.index, pd.DatetimeIndex):
        frame.index = pd.to_datetime(frame.index, errors='coerce')
    frame = frame[frame.index.notna()]

    close = pd.to_numeric(frame.get('Close'), errors='coerce')
    atr = pd.to_numeric(frame.get('ATR'), errors='coerce')

    weekly = frame[['High', 'Low']].resample('W-FRI').agg({'High': 'max', 'Low': 'min'})
    weekly['W_Support'] = weekly['Low'].rolling(window=12, min_periods=12).min()
    weekly['W_Resistance'] = weekly['High'].rolling(window=12, min_periods=12).max()

    w_support = weekly['W_Support'].reindex(frame.index, method='ffill')
    w_resistance = weekly['W_Resistance'].reindex(frame.index, method='ffill')

    safe_close = close.replace(0, np.nan)
    frame['Weekly_Support_Dist'] = (close - w_support) / safe_close
    frame['Weekly_Resistance_Dist'] = (w_resistance - close) / safe_close

    frame['Weekly_Support_Strength'] = _touch_density(close, w_support, atr, lookback=20, tolerance=0.75)
    frame['Weekly_Resistance_Strength'] = _touch_density(close, w_resistance, atr, lookback=20, tolerance=0.75)

    support_dist_abs = frame['Weekly_Support_Dist'].abs()
    resistance_dist_abs = frame['Weekly_Resistance_Dist'].abs()
    closer_dist = pd.concat([support_dist_abs, resistance_dist_abs], axis=1).min(axis=1)
    frame['SR_Confluence'] = (
        (frame['Weekly_Support_Strength'] + frame['Weekly_Resistance_Strength'])
        / (1 + (closer_dist * 100.0))
    )

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
    close_series = _coerce_to_series(close_series)
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
    close_series = _coerce_to_series(close_series)
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


def _load_watchlist_rows(path="watchlist.db"):
    if not os.path.exists(path):
        return []

    try:
        with sqlite3.connect(path) as conn:
            rows = conn.execute(
                "SELECT ticker, stock_name, created_at FROM watchlist ORDER BY created_at ASC"
            ).fetchall()
    except Exception:
        return []

    return [
        {
            "ticker": str(ticker or "").upper(),
            "stock_name": str(stock_name or ""),
            "created_at": str(created_at or ""),
        }
        for ticker, stock_name, created_at in rows
        if str(ticker or "").strip()
    ]


def _load_portfolio_rows(path="portfolio.db"):
    if not os.path.exists(path):
        return []

    try:
        with sqlite3.connect(path) as conn:
            rows = conn.execute(
                """
                SELECT ticker, shares, cost_basis, created_at
                FROM portfolio_positions
                ORDER BY created_at DESC
                """
            ).fetchall()
    except Exception:
        return []

    payload = []
    for ticker, shares, cost_basis, created_at in rows:
        symbol = str(ticker or "").upper().strip()
        if not symbol:
            continue
        qty = float(shares or 0)
        basis = float(cost_basis or 0)
        total_cost = qty * basis
        payload.append(
            {
                "ticker": symbol,
                "shares": round(qty, 4),
                "cost_basis": round(basis, 4),
                "position_cost": round(total_cost, 2),
                "created_at": str(created_at or ""),
            }
        )
    return payload


def _to_tv_symbol(ticker):
    symbol = str(ticker or "").upper().strip()
    if not symbol:
        return ""
    if symbol.endswith(".TO"):
        return f"TSX:{symbol[:-3]}"
    return symbol


def _format_market_cap(value):
    try:
        n = float(value)
    except (TypeError, ValueError):
        return "-"

    if n >= 1e12:
        return f"${n / 1e12:.2f}T"
    if n >= 1e9:
        return f"${n / 1e9:.2f}B"
    if n >= 1e6:
        return f"${n / 1e6:.2f}M"
    if n >= 1e3:
        return f"${n / 1e3:.2f}K"
    return f"${n:.0f}"


def _build_stock_name_map(report_df, watchlist_rows, portfolio_rows):
    """Resolve ticker->stock name for tables beyond the recommendation list."""
    name_map = {}

    if report_df is not None and not report_df.empty:
        for _, row in report_df.iterrows():
            symbol = str(row.get("ticker", "")).upper().strip()
            stock_name = str(row.get("stock_name", "")).strip()
            if symbol and stock_name:
                name_map[symbol] = stock_name

    pending_symbols = set()
    for row in watchlist_rows + portfolio_rows:
        symbol = str(row.get("ticker", "")).upper().strip()
        stock_name = str(row.get("stock_name", "")).strip()
        if not symbol:
            continue
        if stock_name:
            name_map[symbol] = stock_name
        elif symbol not in name_map:
            pending_symbols.add(symbol)

    for symbol in sorted(pending_symbols):
        metadata = get_ticker_metadata(symbol)
        stock_name = str(metadata.get("stock_name") or "").strip()
        if stock_name:
            name_map[symbol] = stock_name

    return name_map


def _ensure_web_report_cache_table(conn):
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS web_report_cache (
            ticker TEXT PRIMARY KEY,
            day5_prediction_pct REAL,
            day5_confidence_pct REAL,
            updated_at_utc TEXT NOT NULL
        )
        """
    )
    conn.commit()


def upsert_web_report_cache(ticker, day5_prediction_pct=None, day5_confidence_pct=None, db_path=WEB_REPORT_CACHE_DB_PATH):
    symbol = str(ticker or "").strip().upper()
    if not symbol:
        return False

    now_utc = datetime.now(timezone.utc).replace(microsecond=0).isoformat()
    with sqlite3.connect(db_path) as conn:
        _ensure_web_report_cache_table(conn)
        conn.execute(
            """
            INSERT INTO web_report_cache (ticker, day5_prediction_pct, day5_confidence_pct, updated_at_utc)
            VALUES (?, ?, ?, ?)
            ON CONFLICT(ticker) DO UPDATE SET
                day5_prediction_pct = excluded.day5_prediction_pct,
                day5_confidence_pct = excluded.day5_confidence_pct,
                updated_at_utc = excluded.updated_at_utc
            """,
            (symbol, day5_prediction_pct, day5_confidence_pct, now_utc),
        )
        conn.commit()
    return True


def load_web_report_cache_map(db_path=WEB_REPORT_CACHE_DB_PATH):
    if not os.path.exists(db_path):
        return {}

    try:
        with sqlite3.connect(db_path) as conn:
            _ensure_web_report_cache_table(conn)
            rows = conn.execute(
                "SELECT ticker, day5_prediction_pct, day5_confidence_pct FROM web_report_cache"
            ).fetchall()
    except Exception:
        return {}

    payload = {}
    for ticker, pred_pct, conf_pct in rows:
        symbol = str(ticker or "").strip().upper()
        if not symbol:
            continue
        payload[symbol] = {
            "day5_prediction_pct": float(pred_pct) if pred_pct is not None else None,
            "day5_confidence_pct": float(conf_pct) if conf_pct is not None else None,
        }
    return payload


def clear_all_web_report_cache(db_path=WEB_REPORT_CACHE_DB_PATH):
    if not os.path.exists(db_path):
        return 0

    with sqlite3.connect(db_path) as conn:
        _ensure_web_report_cache_table(conn)
        cursor = conn.execute("DELETE FROM web_report_cache")
        conn.commit()
        return int(cursor.rowcount or 0)


def _fetch_latest_price_and_change(ticker):
    symbol = str(ticker or "").strip().upper()
    if not symbol:
        return None, None

    try:
        quote_df = yf.download(symbol, period="5d", interval="1d", progress=False)
        if quote_df is None or quote_df.empty:
            quote_df = yf.download(symbol, period="1mo", interval="1d", progress=False)
        if quote_df is None or quote_df.empty:
            return None, None

        close_values = quote_df.get("Close")
        if isinstance(close_values, pd.DataFrame):
            if close_values.shape[1] == 0:
                return None, None
            close_values = close_values.iloc[:, 0]

        closes = pd.to_numeric(close_values, errors="coerce").dropna()
        if closes.empty:
            return None, None

        latest = float(closes.iloc[-1])
        if len(closes) >= 2:
            prev = float(closes.iloc[-2])
            change_pct = ((latest - prev) / prev) * 100 if prev else 0.0
        else:
            change_pct = 0.0
        return latest, change_pct
    except Exception:
        return None, None


def _score_to_action(score):
    if score >= 67:
        return "BUY"
    if score <= 33:
        return "SELL"
    return "HOLD"


def _compute_recommendation_score(pnl_percent, ai_prediction_percent=None):
    momentum_component = max(min((pnl_percent + 15.0) / 30.0, 1.0), 0.0) * 60.0
    if ai_prediction_percent is None:
        ai_component = 20.0
    else:
        ai_component = max(min((ai_prediction_percent + 10.0) / 20.0, 1.0), 0.0) * 40.0
    return max(min(momentum_component + ai_component, 100.0), 0.0)


def _resolve_day5_prediction_pct(ticker, ai_prediction_map, web_cache_map):
    symbol = str(ticker or "").strip().upper()
    if not symbol:
        return None, None

    ai_pct = ai_prediction_map.get(symbol)
    if ai_pct is not None and ai_pct > 2.0:
        return float(ai_pct), None

    cached = web_cache_map.get(symbol, {})
    cached_pred = cached.get("day5_prediction_pct")
    cached_conf = cached.get("day5_confidence_pct")
    if cached_pred is not None:
        return float(cached_pred), float(cached_conf) if cached_conf is not None else None

    detail_payload = load_stock_trade_plan_cache(symbol)
    if detail_payload:
        trade_plan = detail_payload.get("trade_plan", {})
        pred = trade_plan.get("day5_predicted_return")
        conf = trade_plan.get("day5_confidence")
        if pred is not None:
            pred_pct = float(pred) * 100.0
            conf_pct = float(conf) * 100.0 if conf is not None else None
            upsert_web_report_cache(symbol, pred_pct, conf_pct)
            web_cache_map[symbol] = {"day5_prediction_pct": pred_pct, "day5_confidence_pct": conf_pct}
            return pred_pct, conf_pct

    try:
        payload = generate_stock_trade_plan(symbol)
        save_stock_trade_plan_cache(payload)
        trade_plan = payload.get("trade_plan", {})
        pred = trade_plan.get("day5_predicted_return")
        conf = trade_plan.get("day5_confidence")
        if pred is not None:
            pred_pct = float(pred) * 100.0
            conf_pct = float(conf) * 100.0 if conf is not None else None
            upsert_web_report_cache(symbol, pred_pct, conf_pct)
            web_cache_map[symbol] = {"day5_prediction_pct": pred_pct, "day5_confidence_pct": conf_pct}
            return pred_pct, conf_pct
    except Exception:
        return None, None

    return None, None


def generate_github_pages_report(source_csv=SIGNALS_CSV_PATH, output_html=PAGES_INDEX_PATH):
    """Render sortable GitHub Pages dashboard pages for recommendations and portfolio."""
    os.makedirs(os.path.dirname(output_html), exist_ok=True)

    report_df = pd.read_csv(source_csv) if os.path.exists(source_csv) else pd.DataFrame(columns=["stock_name", "ticker", "percentage", "confidence", "marketcap"])
    if not report_df.empty:
        report_df = report_df.fillna("").sort_values(by="marketcap", ascending=False)

    watchlist_rows = _load_watchlist_rows()
    portfolio_rows = _load_portfolio_rows()
    stock_name_map = _build_stock_name_map(report_df, watchlist_rows, portfolio_rows)
    market_history_rows = load_recent_sp500_forecast_history(days_to_keep=5)

    ai_prediction_map = {}
    for _, row in report_df.iterrows():
        symbol = str(row.get("ticker", "")).upper().strip()
        if symbol:
            ai_prediction_map[symbol] = float(pd.to_numeric(row.get("percentage", 0), errors="coerce") or 0)

    web_cache_map = load_web_report_cache_map()
    generated_at = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    rec_rows = []
    for _, row in report_df.iterrows():
        stock_name = html.escape(str(row.get("stock_name", "")) or "-")
        ticker = html.escape(str(row.get("ticker", "")) or "-")
        percentage = float(pd.to_numeric(row.get("percentage", 0), errors="coerce") or 0)
        confidence = float(pd.to_numeric(row.get("confidence", 0), errors="coerce") or 0)
        marketcap_num = float(pd.to_numeric(row.get("marketcap", 0), errors="coerce") or 0)
        tv_symbol = _to_tv_symbol(ticker)
        tv_link = f"<a class='tv-link' href='https://www.tradingview.com/symbols/{html.escape(tv_symbol)}/' target='_blank' rel='noopener'>View</a>" if tv_symbol else "-"
        rec_rows.append(
            f"<tr><td>{stock_name}</td><td>{ticker}</td><td data-sort-value='{percentage:.4f}'>{percentage:+.2f}%</td><td data-sort-value='{confidence:.4f}'>{confidence:.2f}%</td><td data-sort-value='{marketcap_num:.0f}'>{_format_market_cap(marketcap_num)}</td><td>{tv_link}</td></tr>"
        )
    rec_body = "\n".join(rec_rows) if rec_rows else "<tr><td colspan='6'>No buy signals yet. Run a scan first.</td></tr>"

    watch_rows = []
    for row in watchlist_rows:
        ticker = str(row.get("ticker", "")).upper().strip()
        if not ticker:
            continue
        resolved_name = stock_name_map.get(ticker) or row.get("stock_name") or ticker
        tv_symbol = _to_tv_symbol(ticker)
        tv_link = f"<a class='tv-link' href='https://www.tradingview.com/symbols/{html.escape(tv_symbol)}/' target='_blank' rel='noopener'>View</a>" if tv_symbol else "-"
        pred_pct, conf_pct = _resolve_day5_prediction_pct(ticker, ai_prediction_map, web_cache_map)
        pred_text = f"{pred_pct:+.2f}%" if pred_pct is not None else "--"
        conf_text = f" ({conf_pct:.1f}%)" if conf_pct is not None else ""
        watch_rows.append(
            f"<tr><td>{html.escape(ticker)}</td><td>{html.escape(str(resolved_name))}</td><td data-sort-value='{pred_pct if pred_pct is not None else ''}'>{pred_text}{conf_text}</td><td>{tv_link}</td></tr>"
        )
    watch_body = "\n".join(watch_rows) if watch_rows else "<tr><td colspan='4'>No watchlist entries yet.</td></tr>"

    portfolio_html_rows = []
    for row in portfolio_rows:
        ticker = str(row.get("ticker", "")).upper().strip()
        if not ticker:
            continue
        resolved_name = stock_name_map.get(ticker) or ticker
        shares = float(row.get("shares", 0) or 0)
        cost_basis = float(row.get("cost_basis", 0) or 0)
        total_cost = shares * cost_basis
        current_price, _ = _fetch_latest_price_and_change(ticker)
        ai_pred_pct, _ = _resolve_day5_prediction_pct(ticker, ai_prediction_map, web_cache_map)

        if current_price is None:
            current_price_text = "--"
            pnl_text = "--"
            pnl_pct_text = "--"
            signal_text = "HOLD 50/100"
            sort_pnl = ""
            sort_pnl_pct = ""
        else:
            current_value = shares * current_price
            pnl = current_value - total_cost
            pnl_pct = (pnl / total_cost * 100.0) if total_cost else 0.0
            score = _compute_recommendation_score(pnl_pct, ai_pred_pct)
            signal_text = f"{_score_to_action(score)} {score:.0f}/100"
            current_price_text = f"${current_price:.2f}"
            pnl_text = f"${pnl:+.2f}"
            pnl_pct_text = f"{pnl_pct:+.2f}%"
            sort_pnl = f"{pnl:.4f}"
            sort_pnl_pct = f"{pnl_pct:.4f}"

        ai_pred_text = f"{ai_pred_pct:+.2f}%" if ai_pred_pct is not None else "--"
        tv_symbol = _to_tv_symbol(ticker)
        tv_link = f"<a class='tv-link' href='https://www.tradingview.com/symbols/{html.escape(tv_symbol)}/' target='_blank' rel='noopener'>View</a>" if tv_symbol else "-"
        portfolio_html_rows.append(
            f"<tr><td>{html.escape(ticker)}</td><td>{html.escape(str(resolved_name))}</td><td data-sort-value='{shares:.4f}'>{shares:.4f}</td><td data-sort-value='{cost_basis:.4f}'>${cost_basis:.2f}</td><td data-sort-value='{total_cost:.2f}'>${total_cost:.2f}</td><td>{current_price_text}</td><td data-sort-value='{sort_pnl}'>{pnl_text}</td><td data-sort-value='{sort_pnl_pct}'>{pnl_pct_text}</td><td data-sort-value='{ai_pred_pct if ai_pred_pct is not None else ''}'>{ai_pred_text}</td><td>{signal_text}</td><td>{tv_link}</td></tr>"
        )
    portfolio_body = "\n".join(portfolio_html_rows) if portfolio_html_rows else "<tr><td colspan='11'>No portfolio positions yet.</td></tr>"

    day_labels = {1: "Tomorrow", 2: "2 days", 3: "3 days", 4: "4 days", 5: "5 days"}
    market_history_payload = []
    for entry in market_history_rows:
        run_date = str(entry.get("run_date", "") or "")
        if not run_date:
            continue
        forecasts = []
        for r in sorted(entry.get("forecasts", []), key=lambda item: int(item.get("day", 0))):
            day = int(r.get("day", 0))
            forecasts.append({"day": day, "percentage": float(r.get("percentage", 0) or 0), "confidence": float(r.get("confidence", 0) or 0), "label": day_labels.get(day, f"{day} days")})
        market_history_payload.append({"run_date": run_date, "forecasts": forecasts})
    market_history_json = json.dumps(market_history_payload)

    def _build_page(main_sections_html, page_title):
        rec_active = "active" if "Recommendation List" in page_title else ""
        portfolio_active = "active" if "Portfolio" in page_title else ""
        return """<!doctype html>
<html lang="en"><head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>__PAGE_TITLE__</title>
  <style>
    :root {--bg:#0b1220;--panel:#121b2d;--panel-soft:#1b2740;--line:#2a3c60;--text:#eef3ff;--muted:#9caece;--accent:#4da3ff;--shadow:rgba(5,10,20,.45);} * { box-sizing:border-box; }
    body { margin:0;font-family:Inter,system-ui,-apple-system,Segoe UI,Roboto,Arial,sans-serif;color:var(--text);background:radial-gradient(circle at 15% -10%, #1c3360 0%, var(--bg) 52%);min-height:100vh; }
    .page { max-width:1360px;margin:0 auto;padding:1.5rem; } .header { background:linear-gradient(135deg, rgba(77,163,255,.18), rgba(18,27,45,.85));border:1px solid var(--line);border-radius:16px;padding:1.2rem 1.4rem;margin-bottom:1rem;box-shadow:0 10px 30px var(--shadow); }
    .meta { margin-top:.45rem;color:var(--muted);font-size:.92rem; } .nav { margin-top:.75rem;display:flex;gap:.5rem;flex-wrap:wrap; }
    .nav a { color:var(--text);text-decoration:none;border:1px solid var(--line);border-radius:999px;padding:.22rem .75rem;background:var(--panel-soft);font-size:.85rem; } .nav a.active { border-color:var(--accent);color:var(--accent); }
    .card { background:linear-gradient(180deg, rgba(27,39,64,.98), rgba(18,27,45,.98));border:1px solid var(--line);border-radius:14px;padding:1rem;box-shadow:0 8px 24px var(--shadow);margin-bottom:1rem; }
    table { width:100%; border-collapse:collapse; font-size:.92rem; } th,td { border-bottom:1px solid rgba(156,174,206,.2); padding:.62rem .55rem; text-align:left; } th.sortable { cursor:pointer; } th.sortable::after { content:' ↕'; color:#84a6df;font-size:.85em; }
    .tv-link { color:var(--accent); text-decoration:none; font-weight:600; } .footer { color:var(--muted); font-size:.86rem; text-align:right; margin-top:1rem; }
    .forecast-text { color: var(--muted); font-size: .94rem; line-height: 1.5; margin: .3rem 0 0 0; } .forecast-grid { display:flex;flex-wrap:wrap;gap:.55rem;margin-top:.45rem; }
    .forecast-card { background:var(--panel-soft);border:1px solid var(--line);border-radius:10px;padding:.45rem .55rem;min-width:132px; } .forecast-label { font-size:.8rem;color:#dbe7ff;margin:0 0 .35rem 0;font-weight:600; }
    .forecast-value { display:inline-block;font-size:.82rem;font-weight:700;border-radius:6px;padding:.15rem .45rem; } .forecast-value.positive { background:rgba(56,217,150,.25);color:#b4ffd9;border:1px solid rgba(56,217,150,.55); } .forecast-value.negative { background:rgba(255,107,107,.22);color:#ffd2d2;border:1px solid rgba(255,107,107,.5); }
    .forecast-conf { margin-top:.35rem;color:var(--muted);font-size:.75rem; } .forecast-tabs { display:flex;flex-wrap:wrap;gap:.45rem;margin-top:.75rem; } .forecast-tab { border:1px solid var(--line);border-radius:999px;background:var(--panel-soft);color:var(--text);padding:.2rem .65rem;font-size:.78rem;cursor:pointer; } .forecast-tab.active { border-color:var(--accent);color:var(--accent); }
  </style>
</head><body><main class="page"><section class="header"><h1>Stock Tracker Dashboard</h1><p class="meta">Auto-generated from <code>__SOURCE_CSV__</code> at __GENERATED_AT__. Click table headers to sort.</p><nav class="nav"><a href="index.html" class="__REC_ACTIVE__">Recommendations</a><a href="portfolio.html" class="__PORT_ACTIVE__">Portfolio & Watchlist</a></nav></section>__MAIN_SECTIONS__<footer class="footer">Last updated at: __GENERATED_AT__</footer></main>
<script>
(function () {
const marketHistory = __MARKET_HISTORY_JSON__;
const forecastText = document.getElementById('sp500-forecast-text');
const forecastTabs = document.getElementById('sp500-forecast-tabs');
const formatForecast = (row) => { const pct=Number(row?.percentage ?? 0); const conf=Number(row?.confidence ?? 0); const pctLabel=`${pct >= 0 ? '+' : ''}${pct.toFixed(2)}%`; const signClass=pct >= 0 ? 'positive' : 'negative'; return `<div class="forecast-card"><p class="forecast-label">${row?.label || 'N/A'}</p><span class="forecast-value ${signClass}">${pctLabel}</span><div class="forecast-conf">conf ${conf.toFixed(1)}%</div></div>`; };
const renderForecast = (rows) => { if(!forecastText) return; if(!rows || rows.length === 0){ forecastText.textContent='No S&P 500 forecast data yet. Run a daily scan to generate it.'; return; } forecastText.innerHTML=`<div class="forecast-grid">${rows.map(formatForecast).join('')}</div>`; };
const renderForecastTabs = () => { if(!forecastTabs) return; forecastTabs.innerHTML=''; if(!marketHistory || marketHistory.length === 0){ renderForecast([]); return; } let selected = marketHistory[0].run_date; const selectDate = (runDate) => { selected = runDate; Array.from(forecastTabs.querySelectorAll('button')).forEach((btn) => btn.classList.toggle('active', btn.dataset.runDate === selected)); const row = marketHistory.find((item) => item.run_date === selected); renderForecast(row?.forecasts || []); }; marketHistory.forEach((item) => { const button=document.createElement('button'); button.type='button'; button.className='forecast-tab'; button.dataset.runDate=item.run_date; const parsed=new Date(`${item.run_date}T00:00:00`); button.textContent=Number.isNaN(parsed.valueOf()) ? item.run_date : parsed.toLocaleDateString(undefined, { month:'short', day:'2-digit' }); button.addEventListener('click', () => selectDate(item.run_date)); forecastTabs.appendChild(button); }); selectDate(selected); };
renderForecastTabs();
const collator = new Intl.Collator(undefined, { numeric:true, sensitivity:'base' });
document.querySelectorAll('.sortable-table').forEach((table) => { const headers = table.querySelectorAll('th.sortable'); headers.forEach((header, index) => { let asc=false; header.addEventListener('click', () => { const tbody = table.querySelector('tbody'); const rows = Array.from(tbody.querySelectorAll('tr')); asc = !asc; rows.sort((a,b) => { const aCell=a.children[index]; const bCell=b.children[index]; const aVal=aCell?.getAttribute('data-sort-value') ?? aCell?.innerText ?? ''; const bVal=bCell?.getAttribute('data-sort-value') ?? bCell?.innerText ?? ''; const aNum=Number(aVal.replace(/[^0-9+-.]/g,'')); const bNum=Number(bVal.replace(/[^0-9+-.]/g,'')); if(!Number.isNaN(aNum) && !Number.isNaN(bNum) && aVal.trim() !== '' && bVal.trim() !== '') return asc ? aNum - bNum : bNum - aNum; return asc ? collator.compare(aVal,bVal) : collator.compare(bVal,aVal); }); rows.forEach((row) => tbody.appendChild(row)); }); }); });
})();
</script></body></html>""".replace("__PAGE_TITLE__", page_title).replace("__SOURCE_CSV__", html.escape(source_csv)).replace("__GENERATED_AT__", generated_at).replace("__REC_ACTIVE__", rec_active).replace("__PORT_ACTIVE__", portfolio_active).replace("__MAIN_SECTIONS__", main_sections_html).replace("__MARKET_HISTORY_JSON__", market_history_json)

    rec_sections = f"""
    <section class="card"><h2>S&amp;P 500 Forecast (next 5 trading days)</h2><p id="sp500-forecast-text" class="forecast-text">No S&amp;P 500 forecast data yet. Run a daily scan to generate it.</p><div id="sp500-forecast-tabs" class="forecast-tabs"></div></section>
    <section class="card"><h2>Recommendation List</h2><table class="sortable-table"><thead><tr><th class="sortable">Stock Name</th><th class="sortable">Ticker</th><th class="sortable">Expected 5D Return</th><th class="sortable">Confidence</th><th class="sortable">Market Cap</th><th>TradingView</th></tr></thead><tbody>{rec_body}</tbody></table></section>
    """

    portfolio_sections = f"""
    <section class="card"><h2>Portfolio</h2><table class="sortable-table"><thead><tr><th class="sortable">Ticker</th><th class="sortable">Stock Name</th><th class="sortable">Shares</th><th class="sortable">Cost Basis</th><th class="sortable">Total Cost</th><th class="sortable">Current Price</th><th class="sortable">P&amp;L</th><th class="sortable">P&amp;L %</th><th class="sortable">AI 5D</th><th class="sortable">Signal</th><th>TradingView</th></tr></thead><tbody>{portfolio_body}</tbody></table></section>
    <section class="card"><h2>Watchlist (AI 5-day prediction)</h2><table class="sortable-table"><thead><tr><th class="sortable">Ticker</th><th class="sortable">Name</th><th class="sortable">AI 5D Return</th><th>TradingView</th></tr></thead><tbody>{watch_body}</tbody></table></section>
    """

    with open(output_html, "w", encoding="utf-8") as f:
        f.write(_build_page(rec_sections, "Stock Tracker - Recommendation List"))

    portfolio_path = PAGES_PORTFOLIO_PATH
    with open(portfolio_path, "w", encoding="utf-8") as f:
        f.write(_build_page(portfolio_sections, "Stock Tracker - Portfolio"))

    with open(PAGES_NOJEKYLL_PATH, "w", encoding="utf-8") as f:
        f.write("")

    print(f"Generated GitHub Pages dashboard: {output_html}")
    print(f"Generated GitHub Pages portfolio page: {portfolio_path}")

def auto_commit_and_push_pages(paths_to_commit):
    """Commit and push generated report files so GitHub Pages updates automatically."""
    if not os.path.isdir(".git"):
        return

    auto_push = os.environ.get("STOCK_TRACKER_AUTO_PUSH", "1").lower() in {"1", "true", "yes", "on"}
    if not auto_push:
        print("STOCK_TRACKER_AUTO_PUSH disabled; skipping git push.")
        return

    try:
        for path in paths_to_commit:
            subprocess.run(["git", "add", path], check=True)

        status = subprocess.run(
            ["git", "diff", "--cached", "--quiet"],
            check=False,
        )
        if status.returncode == 0:
            print("No report changes detected; skipping git commit/push.")
            return

        timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
        commit_message = f"chore: update stock recommendations ({timestamp})"
        subprocess.run(["git", "commit", "-m", commit_message], check=True)
        subprocess.run(["git", "push"], check=True)
        print("Pushed latest recommendations to GitHub.")
    except Exception as exc:
        print(f"Auto push failed: {exc}")


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
def process_dataframe(df, val_split=VAL_SPLIT, target_horizon=5):
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

    # Multi-timeframe AI features for level significance.
    df = add_multitimeframe_sr_features(df)

    target_horizon = int(max(target_horizon, 1))
    df['Target'] = df['Close'].shift(-target_horizon) / df['Close'] - 1
    
    df = df.replace([np.inf, -np.inf], np.nan).dropna()

    # Some symbols (indexes/futures/thin tickers) can collapse to an empty frame
    # once indicators are built. Return a graceful skip instead of raising inside
    # StandardScaler with "0 sample(s)".
    if len(df) <= WINDOW_SIZE:
        return None, None, None, None, None

    features = FEATURE_COLUMNS

    # Split before scaling to prevent leakage
    train_df = df.iloc[:-WINDOW_SIZE].copy()
    predict_df = df.iloc[-WINDOW_SIZE:].copy()

    if train_df.empty or predict_df.empty:
        return None, None, None, None, None

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


def write_sp500_forecast_csv(forecasts, path=MARKET_FORECAST_CSV_PATH):
    """Persist 1-5 day S&P 500 return + confidence forecasts for the UI header row."""
    rows = []
    for item in forecasts:
        rows.append(
            {
                "day": int(item.get("day", 0)),
                "percentage": round(float(item.get("predicted_return", 0.0)) * 100, 2),
                "confidence": round(float(item.get("confidence", 0.0)) * 100, 2),
            }
        )

    pd.DataFrame(rows, columns=["day", "percentage", "confidence"]).to_csv(path, index=False)
    print(f"Wrote {len(rows)} S&P 500 forecast rows to {path}.")


def _ensure_sp500_forecast_table(conn):
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS sp500_forecast_history (
            run_date TEXT NOT NULL,
            day INTEGER NOT NULL,
            percentage REAL NOT NULL,
            confidence REAL NOT NULL,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            PRIMARY KEY (run_date, day)
        )
        """
    )


def save_sp500_forecast_history(forecasts, db_path=MARKET_FORECAST_DB_PATH, run_date=None, days_to_keep=5):
    """Upsert one run-day and keep only the latest `days_to_keep` run-days."""
    if run_date is None:
        run_date = datetime.now().date()

    if isinstance(run_date, datetime):
        run_date = run_date.date()

    run_date_str = run_date.isoformat()
    now_str = datetime.now().isoformat(timespec="seconds")

    with sqlite3.connect(db_path) as conn:
        _ensure_sp500_forecast_table(conn)

        for item in forecasts:
            day = int(item.get("day", 0))
            if day <= 0:
                continue

            percentage = round(float(item.get("predicted_return", 0.0)) * 100, 2)
            confidence = round(float(item.get("confidence", 0.0)) * 100, 2)

            conn.execute(
                """
                INSERT INTO sp500_forecast_history (run_date, day, percentage, confidence, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(run_date, day) DO UPDATE SET
                    percentage=excluded.percentage,
                    confidence=excluded.confidence,
                    updated_at=excluded.updated_at
                """,
                (run_date_str, day, percentage, confidence, now_str, now_str),
            )

        conn.execute(
            """
            DELETE FROM sp500_forecast_history
            WHERE run_date NOT IN (
                SELECT run_date
                FROM sp500_forecast_history
                GROUP BY run_date
                ORDER BY run_date DESC
                LIMIT ?
            )
            """,
            (max(int(days_to_keep), 0),),
        )
        conn.commit()


def load_recent_sp500_forecast_history(db_path=MARKET_FORECAST_DB_PATH, days_to_keep=5):
    """Load latest `days_to_keep` run-days of S&P forecast history (newest first)."""
    if not os.path.exists(db_path):
        return []

    with sqlite3.connect(db_path) as conn:
        _ensure_sp500_forecast_table(conn)
        rows = conn.execute(
            """
            WITH recent_dates AS (
                SELECT run_date
                FROM sp500_forecast_history
                GROUP BY run_date
                ORDER BY run_date DESC
                LIMIT ?
            )
            SELECT h.run_date, h.day, h.percentage, h.confidence
            FROM sp500_forecast_history h
            INNER JOIN recent_dates r ON r.run_date = h.run_date
            ORDER BY h.run_date DESC, h.day ASC
            """,
            (max(int(days_to_keep), 0),),
        ).fetchall()

    grouped = {}
    for run_date, day, percentage, confidence in rows:
        grouped.setdefault(run_date, []).append(
            {
                "day": int(day),
                "percentage": float(percentage),
                "confidence": float(confidence),
            }
        )

    return [{"run_date": run_date, "forecasts": forecasts} for run_date, forecasts in grouped.items()]


def generate_sp500_forecast(base_model_state=None, checkpoint=None, ticker="^GSPC"):
    """Generate next 1-5 day S&P 500 forecasts from the current global model."""
    try:
        if base_model_state is None:
            checkpoint = checkpoint or (BEST_MODEL_PATH if os.path.exists(BEST_MODEL_PATH) else MODEL_PATH)
            if not os.path.exists(checkpoint):
                write_sp500_forecast_csv([])
                return []
            base_model_state = torch.load(checkpoint, map_location=DEVICE)

        ticker_df = yf.download(ticker, period="5y", progress=False)
        forecasts = []

        for horizon in range(1, 6):
            X_train, y_train, X_val, y_val, X_pred = process_dataframe(
                ticker_df.copy(), val_split=0.1, target_horizon=horizon
            )
            if X_train is None:
                continue

            model = PatternScannerLSTM(input_size=INPUT_SIZE).to(DEVICE)
            model.load_state_dict(base_model_state)
            optimizer = torch.optim.AdamW(model.parameters(), lr=LR_FINE_TUNE, weight_decay=WEIGHT_DECAY)
            loss_fn = nn.MSELoss()

            train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=32, shuffle=True)
            for _ in range(2):
                run_epoch(model, train_loader, optimizer, loss_fn, scaler=None)

            model.eval()
            with torch.no_grad():
                pred = model(X_pred.unsqueeze(0).to(DEVICE)).item()
                confidence = compute_confidence_score(model, X_val, y_val, pred)

            forecasts.append(
                {
                    "day": horizon,
                    "predicted_return": float(pred),
                    "confidence": float(confidence),
                }
            )

        write_sp500_forecast_csv(forecasts)
        save_sp500_forecast_history(forecasts)
        return forecasts
    except Exception as exc:
        print(f"S&P 500 forecast generation failed: {exc}")
        write_sp500_forecast_csv([])
        return []


def _build_ai_sr_feature_frame(df):
    """Build the same SR-related feature set used by model training."""
    frame = _normalize_ohlcv_dataframe(df).copy()
    required_cols = {"High", "Low", "Close", "Volume"}
    if frame is None or frame.empty or not required_cols.issubset(frame.columns):
        return None

    close_delta = frame['Close'].diff()
    gains = close_delta.where(close_delta > 0, 0.0)
    losses = -close_delta.where(close_delta < 0, 0.0)
    avg_gain = gains.ewm(alpha=1 / 14, min_periods=14, adjust=False).mean()
    avg_loss = losses.ewm(alpha=1 / 14, min_periods=14, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    frame['RSI'] = 100 - (100 / (1 + rs))

    high_low = frame['High'] - frame['Low']
    high_close_prev = (frame['High'] - frame['Close'].shift(1)).abs()
    low_close_prev = (frame['Low'] - frame['Close'].shift(1)).abs()
    true_range = np.maximum.reduce([high_low, high_close_prev, low_close_prev])
    frame['ATR'] = pd.Series(true_range, index=frame.index).ewm(alpha=1 / 14, min_periods=14, adjust=False).mean()

    rolling_low = frame['Low'].rolling(window=20, min_periods=20).min()
    rolling_high = frame['High'].rolling(window=20, min_periods=20).max()
    frame['Support_20'] = (frame['Close'] - rolling_low) / frame['Close']
    frame['Resistance_20'] = (rolling_high - frame['Close']) / frame['Close']

    swing_range = (rolling_high - rolling_low).replace(0, np.nan)
    frame['Fib_236'] = (frame['Close'] - (rolling_high - swing_range * 0.236)) / frame['Close']
    frame['Fib_382'] = (frame['Close'] - (rolling_high - swing_range * 0.382)) / frame['Close']
    frame['Fib_500'] = (frame['Close'] - (rolling_high - swing_range * 0.5)) / frame['Close']
    frame['Fib_618'] = (frame['Close'] - (rolling_high - swing_range * 0.618)) / frame['Close']

    frame = add_multitimeframe_sr_features(frame)
    frame = frame.replace([np.inf, -np.inf], np.nan)
    return frame


def _select_ai_optimized_levels(feature_df, last_close, atr):
    """Choose support/resistance using AI training features and SR strength metrics."""
    if feature_df is None or feature_df.empty:
        return np.nan, np.nan, np.nan, np.nan

    latest = feature_df.iloc[-1]
    atr_safe = max(float(atr), 0.01)

    def _to_float(v):
        try:
            return float(v)
        except (TypeError, ValueError):
            return np.nan

    support_strength = _to_float(latest.get('Weekly_Support_Strength'))
    resistance_strength = _to_float(latest.get('Weekly_Resistance_Strength'))
    sr_confluence = _to_float(latest.get('SR_Confluence'))

    support_candidates = []
    resistance_candidates = []

    support20 = _to_float(latest.get('Support_20'))
    resistance20 = _to_float(latest.get('Resistance_20'))
    if np.isfinite(support20):
        support_candidates.append(last_close * (1.0 - support20))
    if np.isfinite(resistance20):
        resistance_candidates.append(last_close * (1.0 + resistance20))

    ws_dist = _to_float(latest.get('Weekly_Support_Dist'))
    wr_dist = _to_float(latest.get('Weekly_Resistance_Dist'))
    if np.isfinite(ws_dist):
        support_candidates.append(last_close * (1.0 - ws_dist))
    if np.isfinite(wr_dist):
        resistance_candidates.append(last_close * (1.0 + wr_dist))

    for fib_col in ('Fib_236', 'Fib_382', 'Fib_500', 'Fib_618'):
        fib_ratio = _to_float(latest.get(fib_col))
        if np.isfinite(fib_ratio):
            fib_level = last_close * (1.0 - fib_ratio)
            if fib_level < last_close:
                support_candidates.append(fib_level)
            elif fib_level > last_close:
                resistance_candidates.append(fib_level)

    support_candidates = [lvl for lvl in support_candidates if np.isfinite(lvl) and lvl < last_close and lvl > 0]
    resistance_candidates = [lvl for lvl in resistance_candidates if np.isfinite(lvl) and lvl > last_close]

    def _score(level, strength):
        dist_pct = abs(level - last_close) / max(last_close, 0.01)
        proximity = float(np.exp(-dist_pct / 0.03))
        norm_strength = np.clip(strength if np.isfinite(strength) else 0.5, 0.0, 1.0)
        norm_conf = np.clip(sr_confluence if np.isfinite(sr_confluence) else 0.5, 0.0, 1.0)
        # AI-oriented: prioritize learned SR strength/confluence over pure proximity.
        return (0.25 * proximity) + (0.5 * norm_strength) + (0.25 * norm_conf)

    primary_support = max(support_candidates, key=lambda lvl: _score(lvl, support_strength), default=(last_close - 1.5 * atr_safe))
    primary_resistance = max(resistance_candidates, key=lambda lvl: _score(lvl, resistance_strength), default=np.nan)

    return (
        float(primary_support),
        float(primary_resistance) if np.isfinite(primary_resistance) else np.nan,
        float(support_strength) if np.isfinite(support_strength) else np.nan,
        float(resistance_strength) if np.isfinite(resistance_strength) else np.nan,
    )


def generate_stock_trade_plan(ticker, total_capital=100000.0, base_model_state=None, checkpoint=None):
    """Generate on-demand 5-day forecast and trade plan for a single stock.

    This is intentionally computed lazily when the user opens a stock detail view,
    so daily scans do not store per-stock forecast payloads.
    """
    symbol = str(ticker).strip().upper()
    if not symbol:
        raise ValueError("Ticker is required")

    if base_model_state is None:
        checkpoint = checkpoint or (BEST_MODEL_PATH if os.path.exists(BEST_MODEL_PATH) else MODEL_PATH)
        if not os.path.exists(checkpoint):
            raise FileNotFoundError("No trained model checkpoint found. Run training first.")
        base_model_state = torch.load(checkpoint, map_location=DEVICE)

    ticker_df = yf.download(symbol, period="5y", progress=False)
    if ticker_df is None or len(ticker_df) < WINDOW_SIZE + 30:
        raise ValueError(f"Not enough price history for {symbol}.")

    forecasts = []
    last_close = float(pd.to_numeric(_normalize_ohlcv_dataframe(ticker_df)["Close"], errors="coerce").iloc[-1])

    for horizon in range(1, 6):
        # Keep per-ticker/per-horizon forecasts stable across repeated UI opens.
        seed = _stable_seed(symbol, horizon)
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

        X_train, y_train, X_val, y_val, X_pred = process_dataframe(
            ticker_df.copy(), val_split=0.1, target_horizon=horizon
        )
        if X_train is None:
            continue

        model = PatternScannerLSTM(input_size=INPUT_SIZE).to(DEVICE)
        model.load_state_dict(base_model_state)
        optimizer = torch.optim.AdamW(model.parameters(), lr=LR_FINE_TUNE, weight_decay=WEIGHT_DECAY)
        loss_fn = nn.MSELoss()

        train_gen = torch.Generator()
        train_gen.manual_seed(seed)
        train_loader = DataLoader(
            TensorDataset(X_train, y_train),
            batch_size=32,
            shuffle=True,
            generator=train_gen,
        )
        for _ in range(2):
            run_epoch(model, train_loader, optimizer, loss_fn, scaler=None)

        model.eval()
        with torch.no_grad():
            pred = model(X_pred.unsqueeze(0).to(DEVICE)).item()
            confidence = compute_confidence_score(model, X_val, y_val, pred)

        forecasts.append(
            {
                "day": horizon,
                "predicted_return": float(pred),
                "confidence": float(confidence),
                "projected_price": float(last_close * (1.0 + pred)),
            }
        )

    if not forecasts:
        raise ValueError(f"Unable to generate forecast for {symbol}.")

    normalized_df = _normalize_ohlcv_dataframe(ticker_df)
    atr_values = normalized_df["ATR"] if "ATR" in normalized_df.columns else pd.Series(index=normalized_df.index, dtype=float)
    atr_series = pd.to_numeric(atr_values, errors="coerce")
    if np.isscalar(atr_series):
        atr_series = pd.Series([atr_series], dtype=float)

    if atr_series.dropna().empty:
        atr_window = normalized_df.copy()
        atr_window["TR"] = np.maximum.reduce([
            (atr_window["High"] - atr_window["Low"]).abs(),
            (atr_window["High"] - atr_window["Close"].shift(1)).abs(),
            (atr_window["Low"] - atr_window["Close"].shift(1)).abs(),
        ])
        atr = float(pd.to_numeric(atr_window["TR"], errors="coerce").rolling(14).mean().iloc[-1])
    else:
        atr = float(atr_series.dropna().iloc[-1])

    avg_return = float(np.mean([f["predicted_return"] for f in forecasts]))
    avg_confidence = float(np.mean([f["confidence"] for f in forecasts]))

    day5_forecast = next((row for row in forecasts if int(row.get("day", 0)) == 5), None)
    day5_return = float(day5_forecast.get("predicted_return", avg_return)) if day5_forecast else float(avg_return)
    day5_confidence = float(day5_forecast.get("confidence", avg_confidence)) if day5_forecast else float(avg_confidence)

    # Support/Resistance-centric plan using AI-training SR features.
    feature_df = _build_ai_sr_feature_frame(ticker_df)
    atr_safe = max(atr, 0.01)
    primary_support, primary_resistance, support_strength, resistance_strength = _select_ai_optimized_levels(
        feature_df, last_close, atr_safe
    )

    # Stop is set just below support with a volatility buffer.
    stop_loss = max(primary_support - (0.25 * atr_safe), 0.01)
    risk_per_share = max(last_close - stop_loss, 0.01)

    # TP leans heavily on resistance, with model prediction as secondary anchor.
    model_target = last_close * (1.0 + max(avg_return, 0.0))
    if np.isfinite(primary_resistance):
        sr_target = primary_resistance * 0.995
        blended_target = (0.7 * sr_target) + (0.3 * model_target)
    else:
        sr_target = np.nan
        blended_target = model_target

    min_r_multiple_target = last_close + (1.5 * risk_per_share)
    take_profit = max(blended_target, min_r_multiple_target)

    # Position sizing from confidence/edge and volatility clamp.
    edge_score = max(avg_return, 0.0) / 0.05
    confidence_score = avg_confidence
    raw_allocation = 0.3 * edge_score + 0.7 * confidence_score
    volatility_penalty = min(max((atr / max(last_close, 0.01)), 0.0), 0.08) / 0.08
    allocation_pct = np.clip(raw_allocation * (1.0 - 0.35 * volatility_penalty), 0.05, 1.0)
    capital_to_use = float(total_capital) * float(allocation_pct)
    shares = int(capital_to_use // max(last_close, 0.01))

    generated_at_utc = datetime.now(timezone.utc).replace(microsecond=0).isoformat()

    return {
        "ticker": symbol,
        "generated_at_utc": generated_at_utc,
        "current_price": float(last_close),
        "daily_candles": [
            {
                "date": idx.strftime("%Y-%m-%d"),
                "open": float(row.Open),
                "high": float(row.High),
                "low": float(row.Low),
                "close": float(row.Close),
            }
            for idx, row in normalized_df[["Open", "High", "Low", "Close"]].tail(90).iterrows()
            if np.isfinite(row.Open) and np.isfinite(row.High) and np.isfinite(row.Low) and np.isfinite(row.Close)
        ],
        "forecast": forecasts,
        "trade_plan": {
            "take_profit": float(take_profit),
            "stop_loss": float(stop_loss),
            "position_size_pct": float(allocation_pct),
            "capital_used": float(capital_to_use),
            "shares": int(max(shares, 0)),
            "avg_predicted_return": float(avg_return),
            "avg_confidence": float(avg_confidence),
            "day5_predicted_return": float(day5_return),
            "day5_confidence": float(day5_confidence),
            "support_level": float(primary_support),
            "resistance_level": float(primary_resistance) if np.isfinite(primary_resistance) else None,
            "support_strength": float(support_strength) if np.isfinite(support_strength) else None,
            "resistance_strength": float(resistance_strength) if np.isfinite(resistance_strength) else None,
            "sr_target": float(sr_target) if np.isfinite(sr_target) else None,
            "sr_method": "ai_feature_optimized",
        },
    }


def _ensure_stock_detail_cache_table(conn):
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS stock_detail_cache (
            ticker TEXT PRIMARY KEY,
            payload_json TEXT NOT NULL,
            generated_at_utc TEXT,
            updated_at_utc TEXT NOT NULL
        )
        """
    )
    conn.commit()


def save_stock_trade_plan_cache(payload, db_path=STOCK_DETAIL_CACHE_DB_PATH):
    ticker = str((payload or {}).get("ticker", "")).strip().upper()
    if not ticker:
        return False

    now_utc = datetime.now(timezone.utc).replace(microsecond=0).isoformat()
    payload_copy = dict(payload)
    payload_copy["ticker"] = ticker
    payload_copy["updated_at_utc"] = now_utc
    payload_json = json.dumps(payload_copy)
    generated_at_utc = str(payload_copy.get("generated_at_utc", "") or "")

    with sqlite3.connect(db_path) as conn:
        _ensure_stock_detail_cache_table(conn)
        conn.execute(
            """
            INSERT INTO stock_detail_cache (ticker, payload_json, generated_at_utc, updated_at_utc)
            VALUES (?, ?, ?, ?)
            ON CONFLICT(ticker) DO UPDATE SET
                payload_json = excluded.payload_json,
                generated_at_utc = excluded.generated_at_utc,
                updated_at_utc = excluded.updated_at_utc
            """,
            (ticker, payload_json, generated_at_utc, now_utc),
        )
        conn.commit()
    return True


def load_stock_trade_plan_cache(ticker, db_path=STOCK_DETAIL_CACHE_DB_PATH):
    symbol = str(ticker).strip().upper()
    if not symbol or not os.path.exists(db_path):
        return None

    with sqlite3.connect(db_path) as conn:
        _ensure_stock_detail_cache_table(conn)
        row = conn.execute(
            """
            SELECT payload_json
            FROM stock_detail_cache
            WHERE ticker = ?
            LIMIT 1
            """,
            (symbol,),
        ).fetchone()

    if not row:
        return None

    try:
        payload = json.loads(row[0])
    except (TypeError, JSONDecodeError):
        return None

    if isinstance(payload, dict):
        payload["ticker"] = symbol
    return payload


def clear_all_stock_trade_plan_cache(db_path=STOCK_DETAIL_CACHE_DB_PATH):
    if not os.path.exists(db_path):
        return 0

    with sqlite3.connect(db_path) as conn:
        _ensure_stock_detail_cache_table(conn)
        cursor = conn.execute("DELETE FROM stock_detail_cache")
        conn.commit()
        return int(cursor.rowcount or 0)


def _ensure_live_signal_table(conn):
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS live_signal_snapshots (
            ticker TEXT NOT NULL,
            captured_at_utc TEXT NOT NULL,
            market_open INTEGER NOT NULL,
            action TEXT NOT NULL,
            buy_pct REAL NOT NULL,
            hold_pct REAL NOT NULL,
            sell_pct REAL NOT NULL,
            confidence REAL NOT NULL,
            current_price REAL,
            PRIMARY KEY (ticker, captured_at_utc)
        )
        """
    )
    conn.commit()


def is_us_market_open(now_utc=None):
    """Return True when NYSE/NASDAQ cash market is open (Mon-Fri, 9:30-16:00 ET)."""
    if now_utc is None:
        now_utc = datetime.now(timezone.utc)
    if now_utc.tzinfo is None:
        now_utc = now_utc.replace(tzinfo=timezone.utc)

    ny_time = now_utc.astimezone(ZoneInfo("America/New_York"))

    if ny_time.weekday() >= 5:
        return False

    start = ny_time.replace(hour=9, minute=30, second=0, microsecond=0)
    end = ny_time.replace(hour=16, minute=0, second=0, microsecond=0)
    return start <= ny_time <= end


def generate_intraday_signal_mix(ticker, period="5d", interval="5m"):
    """Build a live 5-minute candle signal mix: buy/hold/sell percentages totaling 100."""
    symbol = str(ticker).strip().upper()
    if not symbol:
        raise ValueError("Ticker is required")

    intraday = yf.download(symbol, period=period, interval=interval, progress=False)
    if intraday is None or intraday.empty:
        raise ValueError(f"No intraday data available for {symbol}.")

    df = _normalize_ohlcv_dataframe(intraday).copy()
    close = pd.to_numeric(df.get("Close"), errors="coerce").dropna()
    if len(close) < 40:
        raise ValueError(f"Not enough 5-minute candles for {symbol}.")

    ret = close.pct_change()
    momentum_6 = ret.tail(6).mean()
    momentum_18 = ret.tail(18).mean()
    volatility = ret.tail(30).std()

    ema_fast = close.ewm(span=9).mean()
    ema_slow = close.ewm(span=21).mean()
    ema_signal = (ema_fast.iloc[-1] - ema_slow.iloc[-1]) / max(close.iloc[-1], 0.01)

    raw_buy = (momentum_6 * 1200.0) + (momentum_18 * 900.0) + (ema_signal * 400.0)
    raw_sell = (-momentum_6 * 1200.0) + (-momentum_18 * 900.0) + (-ema_signal * 400.0)
    risk_penalty = float(np.clip((volatility or 0.0) * 1200.0, 0.0, 35.0))

    buy_score = max(raw_buy, 0.0)
    sell_score = max(raw_sell, 0.0)
    hold_score = 25.0 + risk_penalty + max(0.0, 10.0 - abs(raw_buy - raw_sell))

    total = max(buy_score + hold_score + sell_score, 1e-9)
    buy_pct = round((buy_score / total) * 100.0, 1)
    hold_pct = round((hold_score / total) * 100.0, 1)
    sell_pct = round((sell_score / total) * 100.0, 1)

    remainder = round(100.0 - (buy_pct + hold_pct + sell_pct), 1)
    hold_pct = round(hold_pct + remainder, 1)

    action = "HOLD"
    if buy_pct >= max(hold_pct, sell_pct):
        action = "BUY"
    elif sell_pct >= max(hold_pct, buy_pct):
        action = "SELL"

    confidence = round(max(buy_pct, hold_pct, sell_pct), 1)
    return {
        "ticker": symbol,
        "captured_at_utc": datetime.now(timezone.utc).isoformat(),
        "market_open": bool(is_us_market_open()),
        "action": action,
        "buy_pct": float(buy_pct),
        "hold_pct": float(hold_pct),
        "sell_pct": float(sell_pct),
        "confidence": float(confidence),
        "current_price": float(close.iloc[-1]),
        "interval": interval,
        "period": period,
    }


def save_live_signal_snapshot(snapshot, db_path=LIVE_SIGNAL_DB_PATH):
    symbol = str(snapshot.get("ticker", "")).strip().upper()
    captured_at = snapshot.get("captured_at_utc") or datetime.now(timezone.utc).isoformat()
    if not symbol:
        raise ValueError("snapshot must include a ticker")

    with sqlite3.connect(db_path) as conn:
        _ensure_live_signal_table(conn)
        conn.execute(
            """
            INSERT OR REPLACE INTO live_signal_snapshots
            (ticker, captured_at_utc, market_open, action, buy_pct, hold_pct, sell_pct, confidence, current_price)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                symbol,
                str(captured_at),
                1 if snapshot.get("market_open") else 0,
                str(snapshot.get("action", "HOLD")),
                float(snapshot.get("buy_pct", 0.0)),
                float(snapshot.get("hold_pct", 0.0)),
                float(snapshot.get("sell_pct", 0.0)),
                float(snapshot.get("confidence", 0.0)),
                float(snapshot.get("current_price")) if snapshot.get("current_price") is not None else None,
            ),
        )
        conn.commit()


def load_latest_live_signal_snapshot(ticker, db_path=LIVE_SIGNAL_DB_PATH):
    symbol = str(ticker).strip().upper()
    if not symbol:
        return None

    with sqlite3.connect(db_path) as conn:
        _ensure_live_signal_table(conn)
        row = conn.execute(
            """
            SELECT ticker, captured_at_utc, market_open, action, buy_pct, hold_pct, sell_pct, confidence, current_price
            FROM live_signal_snapshots
            WHERE ticker = ?
            ORDER BY captured_at_utc DESC
            LIMIT 1
            """,
            (symbol,),
        ).fetchone()

    if not row:
        return None

    return {
        "ticker": row[0],
        "captured_at_utc": row[1],
        "market_open": bool(row[2]),
        "action": row[3],
        "buy_pct": float(row[4]),
        "hold_pct": float(row[5]),
        "sell_pct": float(row[6]),
        "confidence": float(row[7]),
        "current_price": float(row[8]) if row[8] is not None else None,
    }


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

        close_series = _coerce_to_series(ticker_df.get("Close"))
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
    generate_sp500_forecast(base_model_state=base_state)
    generate_github_pages_report()
    auto_commit_and_push_pages([SIGNALS_CSV_PATH, PAGES_INDEX_PATH, PAGES_NOJEKYLL_PATH])
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
