import os
import queue
import sqlite3
import threading
import time
import traceback
import webbrowser
import tkinter as tk
from tkinter import ttk, simpledialog, messagebox
from datetime import datetime

import pandas as pd

try:
    import yfinance as yf
except ImportError:
    yf = None

try:
    import matplotlib.dates as mdates
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
    from matplotlib.figure import Figure
except ImportError:
    mdates = None
    Figure = None
    FigureCanvasTkAgg = None

import main

SIGNALS_CSV_PATH = "buy_signals.csv"
MACD_SIGNALS_CSV_PATH = "macd_signals.csv"
RSI_SIGNALS_CSV_PATH = "rsi_signals.csv"
MARKET_FORECAST_CSV_PATH = "sp500_forecast.csv"
WATCHLIST_DB_PATH = "watchlist.db"
TV_LAYOUT_ID = "ClEM8BLT"
FORECAST_TABS_DAYS_TO_KEEP = 5
LIVE_SIGNAL_REFRESH_MS = 5 * 60 * 1000
PORTFOLIO_DB_PATH = "portfolio.db"


def _ensure_watchlist_table(conn):
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS watchlist (
            ticker TEXT PRIMARY KEY,
            stock_name TEXT,
            created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
        )
        """
    )
    conn.commit()


def load_watchlist_items(path=WATCHLIST_DB_PATH):
    try:
        with sqlite3.connect(path) as conn:
            _ensure_watchlist_table(conn)
            rows = conn.execute(
                "SELECT ticker, stock_name FROM watchlist ORDER BY created_at ASC"
            ).fetchall()
        return [{"ticker": str(ticker).upper(), "name": stock_name or ""} for ticker, stock_name in rows]
    except Exception:
        return []


def upsert_watchlist_item(ticker, stock_name="", path=WATCHLIST_DB_PATH):
    symbol = str(ticker).strip().upper()
    if not symbol:
        return
    try:
        with sqlite3.connect(path) as conn:
            _ensure_watchlist_table(conn)
            conn.execute(
                """
                INSERT INTO watchlist (ticker, stock_name, created_at)
                VALUES (?, ?, CURRENT_TIMESTAMP)
                ON CONFLICT(ticker) DO UPDATE SET
                    stock_name = excluded.stock_name
                """,
                (symbol, stock_name or ""),
            )
            conn.commit()
    except Exception:
        pass


def delete_watchlist_item(ticker, path=WATCHLIST_DB_PATH):
    symbol = str(ticker).strip().upper()
    if not symbol:
        return
    try:
        with sqlite3.connect(path) as conn:
            _ensure_watchlist_table(conn)
            conn.execute("DELETE FROM watchlist WHERE ticker = ?", (symbol,))
            conn.commit()
    except Exception:
        pass


def open_tradingview(ticker):
    """Open TradingView chart for a ticker, handling TSX suffixes."""
    symbol = str(ticker).upper()
    if symbol.endswith('.TO'):
        tv_symbol = f"TSX:{symbol.replace('.TO', '')}"
    else:
        tv_symbol = symbol

    url = f"https://www.tradingview.com/chart/{TV_LAYOUT_ID}/?symbol={tv_symbol}"
    webbrowser.open(url)


def _ensure_portfolio_table(conn):
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS portfolio_positions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ticker TEXT NOT NULL,
            shares REAL NOT NULL,
            cost_basis REAL NOT NULL,
            created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
        )
        """
    )
    conn.commit()


def load_portfolio_positions(path=PORTFOLIO_DB_PATH):
    try:
        with sqlite3.connect(path) as conn:
            _ensure_portfolio_table(conn)
            rows = conn.execute(
                "SELECT id, ticker, shares, cost_basis FROM portfolio_positions ORDER BY created_at DESC"
            ).fetchall()
        return [
            {
                "id": row_id,
                "ticker": str(ticker or "").upper(),
                "shares": float(shares or 0),
                "cost_basis": float(cost_basis or 0),
            }
            for row_id, ticker, shares, cost_basis in rows
        ]
    except Exception:
        return []


def insert_portfolio_position(ticker, shares, cost_basis, path=PORTFOLIO_DB_PATH):
    symbol = normalize_ticker_input(ticker)
    if not symbol:
        raise ValueError("Ticker is required.")

    qty = float(shares)
    if qty <= 0:
        raise ValueError("Shares must be greater than 0.")

    basis = float(cost_basis)
    if basis <= 0:
        raise ValueError("Cost basis must be greater than 0.")

    with sqlite3.connect(path) as conn:
        _ensure_portfolio_table(conn)
        conn.execute(
            "INSERT INTO portfolio_positions (ticker, shares, cost_basis) VALUES (?, ?, ?)",
            (symbol, qty, basis),
        )
        conn.commit()


def delete_portfolio_position(position_id, path=PORTFOLIO_DB_PATH):
    with sqlite3.connect(path) as conn:
        _ensure_portfolio_table(conn)
        conn.execute("DELETE FROM portfolio_positions WHERE id = ?", (position_id,))
        conn.commit()


def score_to_action(score):
    if score >= 67:
        return "BUY"
    if score <= 33:
        return "SELL"
    return "HOLD"


def compute_recommendation_score(pnl_percent, ai_prediction_percent=None):
    momentum_component = max(min((pnl_percent + 15.0) / 30.0, 1.0), 0.0) * 60.0
    if ai_prediction_percent is None:
        ai_component = 20.0
    else:
        ai_component = max(min((ai_prediction_percent + 10.0) / 20.0, 1.0), 0.0) * 40.0
    return max(min(momentum_component + ai_component, 100.0), 0.0)


def normalize_ticker_input(raw_ticker):
    """Normalize user-entered ticker symbols for yfinance/trading flows."""
    symbol = str(raw_ticker or "").strip().upper()
    if not symbol:
        return ""

    symbol = symbol.replace(" ", "")
    if symbol.startswith("TSX:"):
        symbol = f"{symbol.split(':', 1)[1]}.TO"
    if symbol.endswith("-TO"):
        symbol = symbol[:-3] + ".TO"
    return symbol


def format_mcap(value):
    """Render numeric market caps in a compact human-readable format."""
    try:
        n = float(value)
    except (TypeError, ValueError):
        return "N/A"

    if n >= 1e12:
        return f"${n / 1e12:.2f}T"
    if n >= 1e9:
        return f"${n / 1e9:.2f}B"
    if n >= 1e6:
        return f"${n / 1e6:.2f}M"
    if n >= 1e3:
        return f"${n / 1e3:.2f}K"
    return f"${n:.0f}"


def _load_csv_rows(path, expected_cols):
    if not os.path.exists(path):
        return []

    df = pd.read_csv(path)
    if df.empty:
        return []

    for col in expected_cols:
        if col not in df.columns:
            df[col] = ""

    if "marketcap" in df.columns:
        df["marketcap"] = pd.to_numeric(df["marketcap"], errors="coerce").fillna(0)
        df = df.sort_values(by="marketcap", ascending=False)

    return df.to_dict("records")


def load_ai_rows(path=SIGNALS_CSV_PATH):
    rows = _load_csv_rows(path, ["percentage", "confidence", "stock_name", "ticker", "marketcap"])
    for row in rows:
        row["percentage"] = float(pd.to_numeric(row.get("percentage", 0), errors="coerce") or 0)
        row["confidence"] = float(pd.to_numeric(row.get("confidence", 0), errors="coerce") or 0)
    return rows


def load_macd_rows(path=MACD_SIGNALS_CSV_PATH):
    rows = _load_csv_rows(path, ["signal_type", "stock_name", "ticker", "price", "change_pct", "marketcap"])
    for row in rows:
        row["signal_type"] = int(pd.to_numeric(row.get("signal_type", 0), errors="coerce") or 0)
        row["price"] = float(pd.to_numeric(row.get("price", 0), errors="coerce") or 0)
        row["change_pct"] = float(pd.to_numeric(row.get("change_pct", 0), errors="coerce") or 0)
    return rows


def load_rsi_rows(path=RSI_SIGNALS_CSV_PATH):
    rows = _load_csv_rows(path, ["signal_type", "stock_name", "ticker", "rsi", "marketcap"])
    for row in rows:
        row["signal_type"] = int(pd.to_numeric(row.get("signal_type", 0), errors="coerce") or 0)
        row["rsi"] = float(pd.to_numeric(row.get("rsi", 0), errors="coerce") or 0)
    return rows




def load_market_forecast_rows(path=MARKET_FORECAST_CSV_PATH):
    rows = _load_csv_rows(path, ["day", "percentage", "confidence"])
    normalized = []
    for row in rows:
        day = int(pd.to_numeric(row.get("day", 0), errors="coerce") or 0)
        if day <= 0:
            continue
        normalized.append(
            {
                "day": day,
                "percentage": float(pd.to_numeric(row.get("percentage", 0), errors="coerce") or 0),
                "confidence": float(pd.to_numeric(row.get("confidence", 0), errors="coerce") or 0),
            }
        )
    return sorted(normalized, key=lambda item: item["day"])


def load_market_forecast_history(days_to_keep=FORECAST_TABS_DAYS_TO_KEEP):
    history = main.load_recent_sp500_forecast_history(days_to_keep=days_to_keep)
    normalized = []
    for entry in history:
        run_date = entry.get("run_date")
        forecasts = sorted(entry.get("forecasts", []), key=lambda item: int(item.get("day", 0)))
        if not run_date or not forecasts:
            continue
        normalized.append({"run_date": run_date, "forecasts": forecasts})
    return normalized

def launch_signals_ui(csv_path=SIGNALS_CSV_PATH):
    root = tk.Tk()
    root.title("Signals Viewer (AI + MACD + RSI)")
    root.geometry("1200x760")

    colors = {
        "bg": "#111318",
        "panel": "#1A1F27",
        "panel_soft": "#222834",
        "text": "#E8ECF3",
        "muted": "#9AA4B2",
        "accent": "#5DADE2",
        "accent_hover": "#74B9FF",
        "border": "#2D3542",
        "success": "#58D68D",
        "danger": "#FF6B6B",
    }

    root.configure(bg=colors["bg"])
    root.option_add("*Font", "Arial 10")
    root.option_add("*Background", colors["bg"])
    root.option_add("*Foreground", colors["text"])
    root.option_add("*Entry.Background", colors["panel_soft"])
    root.option_add("*Entry.Foreground", colors["text"])
    root.option_add("*Entry.insertBackground", colors["text"])

    style = ttk.Style(root)
    style.theme_use("clam")
    style.configure("Dark.TFrame", background=colors["bg"])
    style.configure("Dark.TNotebook", background=colors["bg"], borderwidth=0)
    style.configure("Dark.TNotebook.Tab", background=colors["panel_soft"], foreground=colors["text"], padding=(12, 8))
    style.map(
        "Dark.TNotebook.Tab",
        background=[("selected", colors["panel"]), ("active", colors["panel_soft"])],
        foreground=[("selected", colors["accent"]), ("active", colors["text"])],
    )
    style.configure(
        "Dark.Treeview",
        background=colors["panel"],
        fieldbackground=colors["panel"],
        foreground=colors["text"],
        bordercolor=colors["border"],
        rowheight=28,
    )
    style.configure(
        "Dark.Treeview.Heading",
        background=colors["panel_soft"],
        foreground=colors["text"],
        relief="flat",
        font=("Arial", 10, "bold"),
    )
    style.map("Dark.Treeview", background=[("selected", "#2F3A4A")], foreground=[("selected", colors["text"])])
    style.configure("Dark.Horizontal.TProgressbar", troughcolor=colors["panel_soft"], background=colors["accent"], bordercolor=colors["border"], lightcolor=colors["accent"], darkcolor=colors["accent"])

    def make_button(parent, text, command, accent=False):
        return tk.Button(
            parent,
            text=text,
            command=command,
            bg=colors["accent"] if accent else colors["panel_soft"],
            fg="#0F1115" if accent else colors["text"],
            activebackground=colors["accent_hover"] if accent else "#2B3341",
            activeforeground="#0F1115" if accent else colors["text"],
            relief="flat",
            bd=0,
            padx=12,
            pady=6,
            cursor="hand2",
        )

    app_shell = tk.Frame(root, bg=colors["bg"])
    app_shell.pack(expand=True, fill="both")

    last_updated_var = tk.StringVar(value="Last updated at: --")
    last_updated_label = tk.Label(
        app_shell,
        textvariable=last_updated_var,
        fg=colors["muted"],
        bg=colors["bg"],
        anchor="e",
    )
    last_updated_label.pack(side="bottom", fill="x", padx=12, pady=(0, 8))

    content_frame = tk.Frame(app_shell, bg=colors["bg"])
    content_frame.pack(side="left", expand=True, fill="both")

    watchlist_frame = tk.Frame(app_shell, bd=1, relief="flat", bg=colors["panel"], highlightthickness=1, highlightbackground=colors["border"], padx=10, pady=10, width=300)
    watchlist_frame.pack(side="right", fill="y")
    watchlist_frame.pack_propagate(False)

    title_label = tk.Label(content_frame, text="Flat tabs: AI, MACD, and RSI", font=("Arial", 13, "bold"), bg=colors["bg"], fg=colors["text"])
    title_label.pack(pady=8)

    market_frame = tk.Frame(content_frame, bd=1, relief="flat", bg=colors["panel"], highlightthickness=1, highlightbackground=colors["border"], padx=10, pady=8)
    market_frame.pack(fill="x", padx=10, pady=(0, 6))

    market_title = tk.Label(market_frame, text="S&P 500 Forecast (next 5 trading days)", font=("Arial", 11, "bold"), bg=colors["panel"], fg=colors["text"])
    market_title.pack(anchor="w")

    market_forecast_var = tk.StringVar(value="No S&P 500 forecast data yet. Run a daily scan to generate it.")
    market_forecast_label = tk.Label(market_frame, textvariable=market_forecast_var, font=("Arial", 10), justify="left", anchor="w", bg=colors["panel"], fg=colors["muted"])
    market_forecast_label.pack(fill="x", pady=(2, 0))

    market_tabs_frame = tk.Frame(market_frame, bg=colors["panel"])
    market_tabs_frame.pack(fill="x", pady=(6, 0))

    ticker_search_var = tk.StringVar(value="")
    ticker_search_frame = tk.Frame(content_frame, bg=colors["bg"])
    ticker_search_frame.pack(fill="x", padx=10, pady=(0, 4))

    ticker_search_inner = tk.Frame(ticker_search_frame, bg=colors["bg"])
    ticker_search_inner.pack(anchor="e")

    ticker_search_label = tk.Label(ticker_search_inner, text="Open ticker:", bg=colors["bg"], fg=colors["muted"])
    ticker_search_label.pack(side="left", padx=(0, 6))

    ticker_search_entry = tk.Entry(ticker_search_inner, textvariable=ticker_search_var, width=14, relief="flat", bd=6)
    ticker_search_entry.pack(side="left")

    ticker_search_status_var = tk.StringVar(value="")
    ticker_search_status_label = tk.Label(ticker_search_frame, textvariable=ticker_search_status_var, fg=colors["muted"], bg=colors["bg"])
    ticker_search_status_label.pack(anchor="e", pady=(2, 0))

    notebook = ttk.Notebook(content_frame, style="Dark.TNotebook")
    notebook.pack(expand=True, fill="both", padx=10, pady=6)

    sort_states = {
        "ai": {"column": None, "ascending": True},
        "macd": {"column": None, "ascending": True},
        "rsi": {"column": None, "ascending": True},
    }
    rows_store = {"ai": [], "macd": [], "rsi": []}
    selected_forecast_date = {"value": None}
    market_forecast_history = {"rows": []}

    def render_market_forecast(rows):
        if not rows:
            market_forecast_var.set("No S&P 500 forecast data yet. Run a daily scan to generate it.")
            return

        day_labels = {1: "Tomorrow", 2: "2 days", 3: "3 days", 4: "4 days", 5: "5 days"}
        chunks = []
        for row in rows:
            day = int(row.get("day", 0))
            chunks.append(
                f"{day_labels.get(day, f'{day} days')}: {row.get('percentage', 0):+.2f}% (conf {row.get('confidence', 0):.1f}%)"
            )

        market_forecast_var.set("   |   ".join(chunks))

    def render_market_tabs():
        for widget in market_tabs_frame.winfo_children():
            widget.destroy()

        history = market_forecast_history["rows"]
        if not history:
            selected_forecast_date["value"] = None
            return

        available_dates = [item["run_date"] for item in history]
        if selected_forecast_date["value"] not in available_dates:
            selected_forecast_date["value"] = available_dates[0]

        for item in history:
            run_date = item["run_date"]
            try:
                label = datetime.strptime(run_date, "%Y-%m-%d").strftime("%b %d")
            except ValueError:
                label = run_date
            is_selected = run_date == selected_forecast_date["value"]
            tk.Button(
                market_tabs_frame,
                text=label,
                relief="flat",
                bd=0,
                padx=10,
                pady=4,
                bg=colors["accent"] if is_selected else colors["panel_soft"],
                fg="#0F1115" if is_selected else colors["text"],
                activebackground=colors["accent_hover"],
                activeforeground="#0F1115",
                command=lambda d=run_date: on_market_tab_selected(d),
            ).pack(side="left", padx=(0, 6))

    def on_market_tab_selected(run_date):
        selected_forecast_date["value"] = run_date
        render_market_tabs()
        selected = next((item for item in market_forecast_history["rows"] if item["run_date"] == run_date), None)
        render_market_forecast(selected.get("forecasts", []) if selected else [])

    def build_tree_tab(tab_title, columns, headings, widths):
        frame = ttk.Frame(notebook, style="Dark.TFrame")
        notebook.add(frame, text=tab_title)
        tree = ttk.Treeview(frame, columns=columns, show="headings", style="Dark.Treeview")
        for col, heading, width in zip(columns, headings, widths):
            tree.heading(col, text=heading)
            tree.column(col, width=width, anchor="center" if col in {"ticker", "percentage", "confidence", "signal_type", "price", "change_pct", "rsi"} else "w")
        tree.pack(expand=True, fill="both")
        return tree

    ai_columns = ("percentage", "confidence", "stock_name", "ticker", "marketcap")
    ai_tree = build_tree_tab(
        "🤖 AI Signals",
        ai_columns,
        ("Predicted %", "Confidence", "Stock", "Ticker", "Market Cap"),
        (120, 110, 320, 120, 160),
    )

    macd_columns = ("signal_type", "stock_name", "ticker", "price", "change_pct", "marketcap")
    macd_tree = build_tree_tab(
        "📈 MACD (Flat)",
        macd_columns,
        ("Type", "Stock", "Ticker", "Price", "Change %", "Market Cap"),
        (90, 280, 120, 110, 110, 160),
    )

    rsi_columns = ("signal_type", "stock_name", "ticker", "rsi", "marketcap")
    rsi_tree = build_tree_tab(
        "🆘 RSI (Flat)",
        rsi_columns,
        ("Type", "Stock", "Ticker", "RSI", "Market Cap"),
        (90, 330, 120, 110, 160),
    )

    portfolio_tab = ttk.Frame(notebook, style="Dark.TFrame")
    notebook.add(portfolio_tab, text="💼 Portfolio")

    portfolio_content = tk.Frame(portfolio_tab, bg=colors["bg"])
    portfolio_content.pack(expand=True, fill="both")

    portfolio_form = tk.Frame(portfolio_content, bg=colors["bg"])
    portfolio_form.pack(pady=(10, 6), anchor="n")

    tk.Label(portfolio_form, text="Ticker", bg=colors["bg"], fg=colors["muted"]).grid(row=0, column=0, sticky="w")
    tk.Label(portfolio_form, text="Position (shares)", bg=colors["bg"], fg=colors["muted"]).grid(row=0, column=1, sticky="w", padx=(10, 0))
    tk.Label(portfolio_form, text="Cost basis ($)", bg=colors["bg"], fg=colors["muted"]).grid(row=0, column=2, sticky="w", padx=(10, 0))

    portfolio_ticker_var = tk.StringVar(value="")
    portfolio_shares_var = tk.StringVar(value="")
    portfolio_cost_var = tk.StringVar(value="")
    portfolio_status_var = tk.StringVar(value="")

    tk.Entry(portfolio_form, textvariable=portfolio_ticker_var, width=12, relief="flat", bd=6).grid(row=1, column=0, sticky="w")
    tk.Entry(portfolio_form, textvariable=portfolio_shares_var, width=14, relief="flat", bd=6).grid(row=1, column=1, sticky="w", padx=(10, 0))
    tk.Entry(portfolio_form, textvariable=portfolio_cost_var, width=14, relief="flat", bd=6).grid(row=1, column=2, sticky="w", padx=(10, 0))

    portfolio_rows = {"items": []}

    portfolio_columns = (
        "id",
        "ticker",
        "stock_name",
        "shares",
        "cost_basis",
        "price",
        "pnl",
        "pnl_pct",
        "signal",
    )
    portfolio_tree = ttk.Treeview(portfolio_content, columns=portfolio_columns, show="headings", style="Dark.Treeview")
    portfolio_tree.heading("id", text="ID")
    portfolio_tree.heading("ticker", text="Ticker")
    portfolio_tree.heading("stock_name", text="Stock")
    portfolio_tree.heading("shares", text="Position")
    portfolio_tree.heading("cost_basis", text="Cost")
    portfolio_tree.heading("price", text="Price")
    portfolio_tree.heading("pnl", text="Current P&L")
    portfolio_tree.heading("pnl_pct", text="P&L %")
    portfolio_tree.heading("signal", text="Signal /100")

    portfolio_tree.column("id", width=45, anchor="center")
    portfolio_tree.column("ticker", width=90, anchor="center")
    portfolio_tree.column("stock_name", width=220, anchor="w")
    portfolio_tree.column("shares", width=90, anchor="e")
    portfolio_tree.column("cost_basis", width=90, anchor="e")
    portfolio_tree.column("price", width=90, anchor="e")
    portfolio_tree.column("pnl", width=120, anchor="e")
    portfolio_tree.column("pnl_pct", width=90, anchor="e")
    portfolio_tree.column("signal", width=130, anchor="center")
    portfolio_tree.pack(expand=True, fill="both", padx=10, pady=(0, 8), anchor="n")

    portfolio_status_label = tk.Label(portfolio_content, textvariable=portfolio_status_var, fg=colors["muted"], bg=colors["bg"], anchor="center")
    portfolio_status_label.pack(fill="x", padx=10, pady=(0, 10))

    watchlist_items = load_watchlist_items()
    watchlist_prediction_cache = {}
    watchlist_prediction_in_flight = set()
    watchlist_prediction_queue = queue.Queue()

    watchlist_title = tk.Label(watchlist_frame, text="Watchlist", font=("Arial", 12, "bold"), bg=colors["panel"], fg=colors["text"])
    watchlist_title.pack(anchor="w")

    watchlist_hint = tk.Label(watchlist_frame, text="Pinned like TradingView", fg=colors["muted"], bg=colors["panel"])
    watchlist_hint.pack(anchor="w", pady=(0, 4))

    def fetch_watchlist_quote(ticker):
        if yf is None:
            return None, None
        try:
            quote_df = yf.Ticker(ticker).history(period="5d", interval="1d", auto_adjust=False)
            if quote_df is None or quote_df.empty:
                quote_df = yf.download(ticker, period="5d", interval="1d", progress=False)
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

    stock_name_cache = {}

    def get_stock_name_for_ticker(ticker):
        symbol = str(ticker or "").upper().strip()
        if not symbol:
            return ""

        cached = stock_name_cache.get(symbol)
        if cached:
            return cached

        for item in watchlist_items:
            if str(item.get("ticker", "")).upper() == symbol:
                candidate = str(item.get("name") or "").strip()
                if candidate:
                    stock_name_cache[symbol] = candidate
                    return candidate

        for table in ("ai", "macd", "rsi"):
            for row in rows_store.get(table, []):
                if str(row.get("ticker", "")).upper() == symbol:
                    candidate = str(row.get("stock_name", "")).strip()
                    if candidate:
                        stock_name_cache[symbol] = candidate
                        return candidate

        if yf is not None:
            try:
                info = yf.Ticker(symbol).info or {}
                candidate = str(info.get("shortName") or info.get("longName") or "").strip()
                if candidate:
                    stock_name_cache[symbol] = candidate
                    return candidate
            except Exception:
                pass

        return ""

    def refresh_portfolio_table():
        portfolio_rows["items"] = load_portfolio_positions()
        for item_id in portfolio_tree.get_children():
            portfolio_tree.delete(item_id)

        if not portfolio_rows["items"]:
            portfolio_status_var.set("No positions yet. Add one above to get buy/sell/hold guidance.")
            return

        for pos in portfolio_rows["items"]:
            ticker = pos.get("ticker", "")
            name = get_stock_name_for_ticker(ticker) or ticker
            price, _ = fetch_watchlist_quote(ticker)
            shares = float(pos.get("shares", 0) or 0)
            cost_basis = float(pos.get("cost_basis", 0) or 0)
            total_cost = shares * cost_basis

            if isinstance(price, (int, float)):
                current_value = shares * float(price)
                pnl = current_value - total_cost
                pnl_pct = (pnl / total_cost * 100.0) if total_cost else 0.0
                ai_prediction = get_ai_prediction_for_ticker(ticker)
                score = compute_recommendation_score(pnl_pct, ai_prediction)
                action = score_to_action(score)
                signal_text = f"{action} {score:.0f}/100"
                price_text = f"${price:.2f}"
                pnl_text = f"${pnl:+.2f}"
                pnl_pct_text = f"{pnl_pct:+.2f}%"
            else:
                price_text = "--"
                pnl_text = "--"
                pnl_pct_text = "--"
                signal_text = "HOLD 50/100"

            portfolio_tree.insert(
                "",
                "end",
                values=(
                    pos.get("id"),
                    ticker,
                    name,
                    f"{shares:.4f}",
                    f"${cost_basis:.2f}",
                    price_text,
                    pnl_text,
                    pnl_pct_text,
                    signal_text,
                ),
            )

        portfolio_status_var.set(
            "Signal score blends current P&L momentum + AI 5-day forecast. 0 = strong sell, 100 = strong buy."
        )

    def add_portfolio_position():
        try:
            insert_portfolio_position(
                portfolio_ticker_var.get(),
                portfolio_shares_var.get(),
                portfolio_cost_var.get(),
            )
        except ValueError as exc:
            portfolio_status_var.set(str(exc))
            return
        except Exception as exc:
            portfolio_status_var.set(f"Unable to save position: {exc}")
            return

        portfolio_ticker_var.set("")
        portfolio_shares_var.set("")
        portfolio_cost_var.set("")
        refresh_portfolio_table()

    def remove_selected_portfolio_position():
        selected = portfolio_tree.selection()
        if not selected:
            portfolio_status_var.set("Select a portfolio row to delete.")
            return

        values = portfolio_tree.item(selected[0]).get("values", [])
        if not values:
            portfolio_status_var.set("Could not read selected portfolio row.")
            return

        try:
            delete_portfolio_position(int(values[0]))
        except Exception as exc:
            portfolio_status_var.set(f"Unable to delete position: {exc}")
            return

        refresh_portfolio_table()

    portfolio_button_row = tk.Frame(portfolio_form, bg=colors["bg"])
    portfolio_button_row.grid(row=1, column=3, padx=(10, 0), sticky="w")

    add_position_btn = make_button(portfolio_button_row, "Add Position", add_portfolio_position, accent=True)
    add_position_btn.pack(side="left")

    delete_position_btn = make_button(portfolio_button_row, "Delete Selected", remove_selected_portfolio_position)
    delete_position_btn.pack(side="left", padx=(6, 0))

    portfolio_ticker_var_entry_bind = lambda _event=None: add_portfolio_position()
    for entry_widget in portfolio_form.winfo_children():
        if isinstance(entry_widget, tk.Entry):
            entry_widget.bind("<Return>", portfolio_ticker_var_entry_bind)

    def get_ai_prediction_for_ticker(ticker):
        for row in rows_store.get("ai", []):
            if str(row.get("ticker", "")).upper() == ticker:
                return float(row.get("percentage", 0.0) or 0.0)
        return None

    def request_watchlist_prediction(ticker):
        symbol = str(ticker).upper()
        if symbol in watchlist_prediction_in_flight:
            return
        watchlist_prediction_in_flight.add(symbol)

        def worker():
            try:
                payload = main.generate_stock_trade_plan(symbol)
                day5 = payload.get("trade_plan", {}).get("day5_predicted_return")
                value = float(day5) * 100 if day5 is not None else None
            except Exception:
                value = None
            watchlist_prediction_queue.put((symbol, value))

        threading.Thread(target=worker, daemon=True).start()

    def process_watchlist_prediction_queue():
        dirty = False
        while True:
            try:
                symbol, value = watchlist_prediction_queue.get_nowait()
            except queue.Empty:
                break
            watchlist_prediction_in_flight.discard(symbol)
            if value is not None:
                watchlist_prediction_cache[symbol] = value
            dirty = True
        if dirty:
            refresh_watchlist_cards()
        root.after(300, process_watchlist_prediction_queue)

    def get_watchlist_prediction_value(ticker):
        symbol = str(ticker).upper()
        ai_prediction = get_ai_prediction_for_ticker(symbol)
        if ai_prediction is not None and ai_prediction > 2.0:
            return ai_prediction

        cached = watchlist_prediction_cache.get(symbol)
        if cached is None:
            request_watchlist_prediction(symbol)
        return cached

    watchlist_cards_canvas = tk.Canvas(
        watchlist_frame,
        bg=colors["panel"],
        highlightthickness=0,
        bd=0,
        relief="flat",
    )
    watchlist_cards_scrollbar = tk.Scrollbar(watchlist_frame, orient="vertical", command=watchlist_cards_canvas.yview)
    watchlist_cards_canvas.configure(yscrollcommand=watchlist_cards_scrollbar.set)
    watchlist_cards_canvas.pack(side="left", expand=True, fill="both", pady=(8, 0))
    watchlist_cards_scrollbar.pack(side="right", fill="y", pady=(8, 0))

    watchlist_cards_frame = tk.Frame(watchlist_cards_canvas, bg=colors["panel"])
    watchlist_window = watchlist_cards_canvas.create_window((0, 0), window=watchlist_cards_frame, anchor="nw")

    def sync_watchlist_scroll_region(_event=None):
        watchlist_cards_canvas.configure(scrollregion=watchlist_cards_canvas.bbox("all"))

    def sync_watchlist_window_width(event):
        watchlist_cards_canvas.itemconfig(watchlist_window, width=event.width)

    watchlist_cards_frame.bind("<Configure>", sync_watchlist_scroll_region)
    watchlist_cards_canvas.bind("<Configure>", sync_watchlist_window_width)

    def _watchlist_on_mousewheel(event):
        if event.delta:
            watchlist_cards_canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")

    def _bind_watchlist_mousewheel(_event=None):
        watchlist_cards_canvas.bind_all("<MouseWheel>", _watchlist_on_mousewheel)

    def _unbind_watchlist_mousewheel(_event=None):
        watchlist_cards_canvas.unbind_all("<MouseWheel>")

    watchlist_cards_canvas.bind("<Enter>", _bind_watchlist_mousewheel)
    watchlist_cards_canvas.bind("<Leave>", _unbind_watchlist_mousewheel)

    def refresh_watchlist_cards():
        for widget in watchlist_cards_frame.winfo_children():
            widget.destroy()

        if not watchlist_items:
            tk.Label(watchlist_cards_frame, text="No stocks added yet.\nUse + Add at the top.", fg=colors["muted"], bg=colors["panel"], justify="left").pack(anchor="w", pady=6)
            return

        for item in watchlist_items:
            ticker = item.get("ticker", "")
            card = tk.Frame(watchlist_cards_frame, bd=1, relief="flat", bg=colors["panel_soft"], highlightthickness=1, highlightbackground=colors["border"], padx=8, pady=8)
            card.pack(fill="x", pady=(0, 6))

            name = item.get("name") or get_stock_name_for_ticker(ticker) or "Unknown name"
            ai_5d = get_watchlist_prediction_value(ticker)
            live_price, day_change = fetch_watchlist_quote(ticker)

            delete_btn = tk.Button(
                card,
                text="✕",
                bg="#3A4150",
                fg=colors["text"],
                activebackground="#4A5264",
                activeforeground=colors["text"],
                relief="flat",
                bd=0,
                padx=4,
                pady=1,
                cursor="hand2",
                font=("Arial", 8, "bold"),
            )

            def remove_from_watchlist(symbol=ticker):
                nonlocal watchlist_items
                watchlist_items = [row for row in watchlist_items if row.get("ticker") != symbol]
                delete_watchlist_item(symbol)
                refresh_watchlist_cards()

            delete_btn.configure(command=remove_from_watchlist)
            delete_btn.place(relx=1.0, x=-6, y=6, anchor="ne")

            header_row = tk.Frame(card, bg=colors["panel_soft"])
            header_row.pack(fill="x", anchor="w")

            tk.Label(header_row, text=ticker, font=("Arial", 11, "bold"), bg=colors["panel_soft"], fg=colors["text"]).pack(side="left")
            tk.Label(
                header_row,
                text=f"  ${live_price:.2f}" if isinstance(live_price, (int, float)) else "  --",
                bg=colors["panel_soft"],
                fg=colors["text"],
                font=("Arial", 10),
            ).pack(side="left")
            if isinstance(day_change, (int, float)):
                change_text = f"  {day_change:+.2f}%"
                change_color = "#2E8B57" if day_change >= 0 else "#C0392B"
                tk.Label(header_row, text=change_text, fg=change_color, bg=colors["panel_soft"], font=("Arial", 10, "bold")).pack(side="left")
            else:
                tk.Label(header_row, text="  --", bg=colors["panel_soft"], fg=colors["text"], font=("Arial", 10)).pack(side="left")

            if ai_5d is None:
                tk.Label(card, text="AI 5D: --", bg=colors["panel_soft"], fg=colors["text"], anchor="w").pack(anchor="w")
            else:
                ai_color = "#2E8B57" if ai_5d >= 0 else "#C0392B"
                tk.Label(card, text=f"AI 5D: {ai_5d:+.2f}%", fg=ai_color, bg=colors["panel_soft"], anchor="w").pack(anchor="w")

            tk.Label(card, text=name, fg=colors["muted"], bg=colors["panel_soft"], font=("Arial", 8)).pack(anchor="w", pady=(4, 0))

            def _open_from_watchlist(_event=None, symbol=ticker):
                if detail_task_in_progress["value"]:
                    ticker_search_status_var.set("Please wait for the current detail request to finish.")
                    return
                ticker_search_status_var.set(f"Loading {symbol}...")
                load_stock_detail(symbol)

            card.bind("<Button-1>", _open_from_watchlist)
            for child in card.winfo_children():
                if child is delete_btn:
                    continue
                child.bind("<Button-1>", _open_from_watchlist)

    def add_to_watchlist_from_prompt():
        ticker = simpledialog.askstring("Add to watchlist", "Enter ticker (e.g. AAPL):", parent=root)
        if ticker is None:
            return
        ticker = normalize_ticker_input(ticker)
        if not ticker:
            messagebox.showinfo("Watchlist", "Please enter a ticker symbol.")
            return
        if any(item.get("ticker") == ticker for item in watchlist_items):
            messagebox.showinfo("Watchlist", f"{ticker} is already in your watchlist.")
            return
        stock_name = get_stock_name_for_ticker(ticker)
        watchlist_items.append({"ticker": ticker, "name": stock_name})
        upsert_watchlist_item(ticker, stock_name)
        refresh_watchlist_cards()

    add_watchlist_btn = make_button(watchlist_frame, "+ Add", add_to_watchlist_from_prompt, accent=True)
    add_watchlist_btn.pack(anchor="w", pady=(0, 2), before=watchlist_cards_canvas)
    process_watchlist_prediction_queue()
    status_var = tk.StringVar(value="")
    status_label = tk.Label(content_frame, textvariable=status_var, fg=colors["accent"], bg=colors["bg"])
    status_label.pack(pady=4)

    train_status_var = tk.StringVar(value="")
    eta_var = tk.StringVar(value="Estimated time left: --")
    progress_value = tk.DoubleVar(value=0.0)
    task_in_progress = {"value": False}
    detail_task_in_progress = {"value": False}
    progress_queue = queue.Queue()

    progress_bar = ttk.Progressbar(content_frame, orient="horizontal", mode="determinate", length=700, variable=progress_value, style="Dark.Horizontal.TProgressbar")
    progress_bar.pack(padx=10, pady=6, fill="x")
    progress_bar.pack_forget()

    train_status_label = tk.Label(content_frame, textvariable=train_status_var, fg=colors["success"], bg=colors["bg"])
    train_status_label.pack(pady=2)

    eta_label = tk.Label(content_frame, textvariable=eta_var, fg=colors["muted"], bg=colors["bg"])
    eta_label.pack(pady=2)

    current_page = {"value": "list"}
    list_page_widgets = []
    list_page_pack_state = {}
    detail_page = tk.Frame(content_frame, bg=colors["bg"])

    detail_header_var = tk.StringVar(value="Stock detail")
    detail_summary_var = tk.StringVar(value="")
    detail_forecast_var = tk.StringVar(value="")
    detail_live_signal_var = tk.StringVar(value="Live 5m signal: waiting for data...")
    detail_status_var = tk.StringVar(value="")
    detail_loading_progress = tk.DoubleVar(value=0)
    live_signal_task_in_progress = {"value": False}

    detail_header_label = tk.Label(detail_page, textvariable=detail_header_var, font=("Arial", 13, "bold"), bg=colors["bg"], fg=colors["text"])
    detail_header_label.pack(anchor="w", padx=10, pady=(10, 6))

    detail_summary_label = tk.Label(detail_page, textvariable=detail_summary_var, justify="left", anchor="w", font=("Arial", 10), bg=colors["bg"], fg=colors["text"])
    detail_summary_label.pack(fill="x", padx=10)

    detail_forecast_label = tk.Label(detail_page, textvariable=detail_forecast_var, justify="left", anchor="w", font=("Arial", 10), bg=colors["bg"], fg=colors["muted"])
    detail_forecast_label.pack(fill="x", padx=10, pady=(6, 6))

    detail_live_signal_label = tk.Label(detail_page, textvariable=detail_live_signal_var, justify="left", anchor="w", font=("Arial", 10, "bold"), bg=colors["bg"], fg=colors["accent"])
    detail_live_signal_label.pack(fill="x", padx=10, pady=(0, 10))

    detail_buttons = tk.Frame(detail_page, bg=colors["bg"])
    detail_buttons.pack(anchor="w", padx=10, pady=(0, 8))

    selected_ticker = {"value": None}
    chart_container = tk.Frame(detail_page, bg=colors["panel"])
    chart_container.pack(expand=True, fill="both", padx=10, pady=(2, 10))
    chart_canvas = {"value": None}

    def clear_detail_chart(message=""):
        if chart_canvas["value"] is not None:
            chart_canvas["value"].get_tk_widget().destroy()
            chart_canvas["value"] = None
        for widget in chart_container.winfo_children():
            widget.destroy()
        if message:
            tk.Label(chart_container, text=message, fg=colors["muted"], bg=colors["panel"], anchor="w", justify="left").pack(anchor="w")

    def render_detail_chart(payload):
        candles = payload.get("daily_candles", [])
        forecast = sorted(payload.get("forecast", []), key=lambda item: int(item.get("day", 0)))
        trade_plan = payload.get("trade_plan", {})

        if Figure is None or FigureCanvasTkAgg is None or mdates is None:
            clear_detail_chart("Chart is unavailable because matplotlib is not installed in this environment.")
            return
        if not candles:
            clear_detail_chart("No candle data available to render chart.")
            return

        clear_detail_chart()

        candle_df = pd.DataFrame(candles)
        candle_df["date"] = pd.to_datetime(candle_df["date"], errors="coerce")
        candle_df = candle_df.dropna(subset=["date", "open", "high", "low", "close"]).sort_values("date")
        if candle_df.empty:
            clear_detail_chart("No valid candle data available to render chart.")
            return

        figure = Figure(figsize=(10.5, 7.2), dpi=100)
        grid = figure.add_gridspec(3, 1, height_ratios=[70, 15, 15], hspace=0.06)
        price_axis = figure.add_subplot(grid[0])
        macd_axis = figure.add_subplot(grid[1], sharex=price_axis)
        rsi_axis = figure.add_subplot(grid[2], sharex=price_axis)

        price_axis.set_title(f"{payload.get('ticker', '')} price action + 5-day projection")
        price_axis.set_ylabel("Price (USD)")

        history_count = len(candle_df)
        x_positions = list(range(history_count))
        candle_width = 0.6
        color_up = "#2E8B57"
        color_down = "#C0392B"

        from matplotlib.patches import Rectangle

        for x_pos, o, h, l, c in zip(x_positions, candle_df["open"], candle_df["high"], candle_df["low"], candle_df["close"]):
            color = color_up if c >= o else color_down
            price_axis.vlines(x_pos, l, h, color=color, linewidth=1.0, zorder=2)
            lower = min(o, c)
            height = max(abs(c - o), 0.02)
            price_axis.add_patch(
                Rectangle(
                    (x_pos - candle_width / 2, lower),
                    candle_width,
                    height,
                    facecolor=color,
                    edgecolor=color,
                    linewidth=0.6,
                    zorder=3,
                )
            )

        last_date = candle_df["date"].iloc[-1]
        last_close = float(candle_df["close"].iloc[-1])
        projected_dates = [last_date + pd.offsets.BDay(int(row.get("day", 0))) for row in forecast if int(row.get("day", 0)) > 0]
        projected_prices = [float(row.get("projected_price", last_close)) for row in forecast if int(row.get("day", 0)) > 0]
        projected_positions = [history_count - 1 + int(row.get("day", 0)) for row in forecast if int(row.get("day", 0)) > 0]

        if projected_positions and projected_prices:
            price_axis.plot(projected_positions, projected_prices, color="#1f77b4", marker="o", linewidth=1.8, label="Predicted next 5 days", zorder=4)
            price_axis.scatter([history_count - 1], [last_close], color="#1f77b4", s=24, zorder=5)

        stop_loss = trade_plan.get("stop_loss")
        take_profit = trade_plan.get("take_profit")
        if isinstance(stop_loss, (int, float)):
            price_axis.axhline(stop_loss, linestyle="--", color="#8E44AD", linewidth=1.2, label=f"Stop loss ${stop_loss:.2f}")
        if isinstance(take_profit, (int, float)):
            price_axis.axhline(take_profit, linestyle="--", color="#D35400", linewidth=1.2, label=f"Take profit ${take_profit:.2f}")

        all_prices = candle_df[["low", "high"]].to_numpy().ravel().tolist() + projected_prices
        if isinstance(stop_loss, (int, float)):
            all_prices.append(float(stop_loss))
        if isinstance(take_profit, (int, float)):
            all_prices.append(float(take_profit))

        min_price = min(all_prices)
        max_price = max(all_prices)
        spread = max(max_price - min_price, 0.5)
        price_axis.set_ylim(min_price - spread * 0.08, max_price + spread * 0.12)

        x_end = projected_positions[-1] if projected_positions else history_count - 1
        price_axis.set_xlim(-1, x_end + 1)
        from matplotlib.ticker import MaxNLocator
        price_axis.yaxis.set_major_locator(MaxNLocator(nbins=8))
        price_axis.grid(True, linestyle=":", linewidth=0.7, alpha=0.6)
        price_axis.legend(loc="upper left", fontsize=9)

        close_series = pd.to_numeric(candle_df["close"], errors="coerce")

        ema_fast = close_series.ewm(span=12, adjust=False).mean()
        ema_slow = close_series.ewm(span=26, adjust=False).mean()
        macd_line = ema_fast - ema_slow
        macd_signal = macd_line.ewm(span=9, adjust=False).mean()
        macd_hist = macd_line - macd_signal

        macd_colors = ["#2E8B57" if value >= 0 else "#C0392B" for value in macd_hist]
        macd_axis.bar(x_positions, macd_hist, width=0.8, color=macd_colors, alpha=0.5, label="Histogram")
        macd_axis.plot(x_positions, macd_line, color="#1f77b4", linewidth=1.3, label="MACD")
        macd_axis.plot(x_positions, macd_signal, color="#E67E22", linewidth=1.3, label="Signal")
        macd_axis.axhline(0, color="gray", linewidth=0.8, linestyle="--")
        macd_axis.set_ylabel("MACD")
        macd_axis.grid(True, linestyle=":", linewidth=0.7, alpha=0.5)
        macd_axis.legend(loc="upper left", fontsize=8)

        delta = close_series.diff()
        gains = delta.clip(lower=0)
        losses = -delta.clip(upper=0)
        avg_gain = gains.rolling(14).mean()
        avg_loss = losses.rolling(14).mean()
        rs = avg_gain / avg_loss.replace(0, pd.NA)
        rsi = 100 - (100 / (1 + rs))

        rsi_axis.plot(x_positions, rsi, color="#8E44AD", linewidth=1.4, label="RSI (14)")
        rsi_axis.axhline(70, color="#C0392B", linestyle="--", linewidth=1.0)
        rsi_axis.axhline(30, color="#2E8B57", linestyle="--", linewidth=1.0)
        rsi_axis.set_ylim(0, 100)
        rsi_axis.set_ylabel("RSI")
        rsi_axis.set_xlabel("Date")
        rsi_axis.grid(True, linestyle=":", linewidth=0.7, alpha=0.5)
        rsi_axis.legend(loc="upper left", fontsize=8)
        tick_positions = sorted(set([0, max(0, history_count // 2), history_count - 1] + projected_positions))
        tick_labels = []
        future_label_map = {
            history_count - 1 + int(row.get("day", 0)): (last_date + pd.offsets.BDay(int(row.get("day", 0)))).strftime("%Y-%m-%d")
            for row in forecast
            if int(row.get("day", 0)) > 0
        }
        for pos in tick_positions:
            if pos < history_count:
                tick_labels.append(candle_df["date"].iloc[pos].strftime("%Y-%m-%d"))
            else:
                tick_labels.append(future_label_map.get(pos, ""))

        rsi_axis.set_xticks(tick_positions)
        rsi_axis.set_xticklabels(tick_labels, rotation=25, ha="right")

        price_axis.tick_params(axis="x", labelbottom=False)
        macd_axis.tick_params(axis="x", labelbottom=False)

        figure.tight_layout()
        tk_canvas = FigureCanvasTkAgg(figure, master=chart_container)
        tk_canvas.draw()
        tk_canvas.get_tk_widget().pack(expand=True, fill="both")
        chart_canvas["value"] = tk_canvas

    def render_live_signal_snapshot(snapshot):
        if not snapshot:
            detail_live_signal_var.set("Live 5m signal: no stored snapshot yet.")
            return

        buy = float(snapshot.get("buy_pct", 0.0) or 0.0)
        hold = float(snapshot.get("hold_pct", 0.0) or 0.0)
        sell = float(snapshot.get("sell_pct", 0.0) or 0.0)
        action = snapshot.get("action", "HOLD")
        confidence = float(snapshot.get("confidence", 0.0) or 0.0)
        captured_at = str(snapshot.get("captured_at_utc", ""))
        market_open = bool(snapshot.get("market_open", False))
        state = "market open" if market_open else "market closed"
        detail_live_signal_var.set(
            f"Live 5m signal ({state}): BUY {buy:.1f}% | HOLD {hold:.1f}% | SELL {sell:.1f}% "
            f"→ {action} (conf {confidence:.1f}%) | updated {captured_at}"
        )

    def refresh_live_signal_for_selected_ticker(force=False):
        ticker = selected_ticker.get("value")
        if not ticker:
            return

        if live_signal_task_in_progress["value"]:
            return

        if not force and not main.is_us_market_open():
            cached = main.load_latest_live_signal_snapshot(ticker)
            render_live_signal_snapshot(cached)
            root.after(LIVE_SIGNAL_REFRESH_MS, refresh_live_signal_for_selected_ticker)
            return

        live_signal_task_in_progress["value"] = True

        def worker():
            try:
                snapshot = main.generate_intraday_signal_mix(ticker)
                main.save_live_signal_snapshot(snapshot)
                progress_queue.put({"stage": "live_signal_loaded", "ticker": ticker, "snapshot": snapshot})
            except Exception as exc:
                progress_queue.put({"stage": "live_signal_error", "ticker": ticker, "message": str(exc)})

        threading.Thread(target=worker, daemon=True).start()
        root.after(LIVE_SIGNAL_REFRESH_MS, refresh_live_signal_for_selected_ticker)

    def show_list_page():
        if current_page["value"] == "list":
            return
        detail_page.pack_forget()
        for widget in list_page_widgets:
            pack_opts = list_page_pack_state.get(widget)
            if pack_opts:
                widget.pack(**pack_opts)
        current_page["value"] = "list"

    def show_detail_page():
        if current_page["value"] == "detail":
            return
        for widget in list_page_widgets:
            if widget.winfo_manager() == "pack":
                list_page_pack_state[widget] = widget.pack_info()
            widget.pack_forget()
        detail_page.pack(expand=True, fill="both", padx=10, pady=6)
        current_page["value"] = "detail"

    def render_stock_detail(payload, listed_5d_pct=None):
        hide_detail_loading_bar()
        ticker = payload.get("ticker", "")
        selected_ticker["value"] = ticker
        trade_plan = payload.get("trade_plan", {})
        forecast = payload.get("forecast", [])
        detail_header_var.set(f"{ticker} - 5 Day Forecast & Trade Plan")

        resistance_level = trade_plan.get("resistance_level")
        sr_target = trade_plan.get("sr_target")
        support_strength = trade_plan.get("support_strength")
        resistance_strength = trade_plan.get("resistance_strength")
        method = trade_plan.get("sr_method", "sr")
        resistance_text = f"${resistance_level:.2f}" if isinstance(resistance_level, (int, float)) else "N/A"
        sr_target_text = f"${sr_target:.2f}" if isinstance(sr_target, (int, float)) else "N/A"
        support_strength_text = f"{support_strength * 100:.1f}%" if isinstance(support_strength, (int, float)) else "N/A"
        resistance_strength_text = f"{resistance_strength * 100:.1f}%" if isinstance(resistance_strength, (int, float)) else "N/A"

        forecast_by_day = {
            int(row.get("day", 0)): float(row.get("predicted_return", 0.0))
            for row in forecast
            if int(row.get("day", 0)) > 0
        }
        day1_return_pct = forecast_by_day.get(1, 0.0) * 100
        day2_return_pct = forecast_by_day.get(2, 0.0) * 100
        day3_return_pct = forecast_by_day.get(3, 0.0) * 100
        model_day5_return_pct = forecast_by_day.get(5, trade_plan.get('day5_predicted_return', trade_plan.get('avg_predicted_return', 0.0))) * 100
        day5_return_pct = float(listed_5d_pct) if listed_5d_pct is not None else float(model_day5_return_pct)
        day5_confidence_pct = trade_plan.get('day5_confidence', trade_plan.get('avg_confidence', 0)) * 100

        summary_lines = [
            f"Current price: ${payload.get('current_price', 0):.2f}",
            f"Predicted return day 1: {day1_return_pct:+.2f}%",
            f"Predicted return day 2: {day2_return_pct:+.2f}%",
            f"Predicted return day 3: {day3_return_pct:+.2f}%",
            f"Predicted return day 5: {day5_return_pct:+.2f}% (conf {day5_confidence_pct:.1f}%)",
            f"Support used for SL: ${trade_plan.get('support_level', 0):.2f} (AI strength: {support_strength_text})",
            f"Nearest resistance: {resistance_text} (AI strength: {resistance_strength_text})",
            f"Recommended stop loss (support - buffer): ${trade_plan.get('stop_loss', 0):.2f}",
            f"Recommended take profit (SR-led): ${trade_plan.get('take_profit', 0):.2f} (SR target: {sr_target_text})",
            f"SR selection method: {method}",
            f"Position size: {trade_plan.get('position_size_pct', 0) * 100:.1f}% of capital",
            f"Estimated shares (assuming $100,000 capital): {trade_plan.get('shares', 0)}",
            f"Average across days 1-5: {trade_plan.get('avg_predicted_return', 0) * 100:+.2f}% | Avg confidence: {trade_plan.get('avg_confidence', 0) * 100:.1f}%",
        ]
        generated_at_utc = str(payload.get("generated_at_utc", "") or "").strip()
        if generated_at_utc:
            summary_lines.append(f"Analysis generated (UTC): {generated_at_utc}")
        if listed_5d_pct is not None:
            summary_lines.insert(5, f"List view 5-day return: {listed_5d_pct:+.2f}% | Model day-5 return: {model_day5_return_pct:+.2f}%")
        detail_summary_var.set("\n".join(summary_lines))

        day_lines = []
        for row in forecast:
            day = int(row.get("day", 0))
            day_lines.append(
                f"Day {day}: {row.get('predicted_return', 0) * 100:+.2f}% (conf {row.get('confidence', 0) * 100:.1f}%) -> ${row.get('projected_price', 0):.2f}"
            )
        detail_forecast_var.set("\n".join(day_lines))
        render_detail_chart(payload)
        cached_snapshot = main.load_latest_live_signal_snapshot(ticker)
        render_live_signal_snapshot(cached_snapshot)
        refresh_live_signal_for_selected_ticker(force=True)
        detail_status_var.set("")
        show_detail_page()

    def load_stock_detail(ticker, listed_5d_pct=None):
        detail_task_in_progress["value"] = True
        ticker_search_status_var.set("")
        cached_payload = main.load_stock_trade_plan_cache(ticker)

        if cached_payload:
            render_stock_detail(cached_payload, listed_5d_pct=listed_5d_pct)
            detail_status_var.set("Loaded cached detail instantly. Refreshing with latest market data...")
        else:
            show_detail_loading_bar()
            detail_header_var.set(f"{ticker} - loading analysis...")
            detail_summary_var.set("Computing on-demand forecast and trade plan...")
            detail_forecast_var.set("")
            detail_live_signal_var.set("Live 5m signal: loading...")
            clear_detail_chart("Loading chart data...")
            detail_status_var.set("")
            show_detail_page()

        def worker():
            try:
                payload = main.generate_stock_trade_plan(ticker)
                main.save_stock_trade_plan_cache(payload)
                progress_queue.put(
                    {
                        "stage": "detail_loaded",
                        "payload": payload,
                        "listed_5d_pct": listed_5d_pct,
                        "used_cache": bool(cached_payload),
                    }
                )
            except Exception as exc:
                progress_queue.put(
                    {
                        "stage": "detail_error",
                        "ticker": ticker,
                        "message": str(exc),
                        "trace": traceback.format_exc(),
                        "had_cached_payload": bool(cached_payload),
                    }
                )

        threading.Thread(target=worker, daemon=True).start()
        handle_progress_updates()

    def add_selected_ticker_to_watchlist():
        ticker = selected_ticker.get("value")
        if not ticker:
            messagebox.showinfo("Watchlist", "Open a stock detail first, then add it.")
            return
        if any(item.get("ticker") == ticker for item in watchlist_items):
            messagebox.showinfo("Watchlist", f"{ticker} is already in your watchlist.")
            return
        stock_name = get_stock_name_for_ticker(ticker)
        watchlist_items.append({"ticker": ticker, "name": stock_name})
        upsert_watchlist_item(ticker, stock_name)
        refresh_watchlist_cards()

    back_btn = make_button(detail_buttons, "← Back to list", show_list_page)
    back_btn.pack(side="left", padx=(0, 8))

    add_watchlist_detail_btn = make_button(detail_buttons, "+ Add to watchlist", add_selected_ticker_to_watchlist)
    add_watchlist_detail_btn.pack(side="left", padx=(0, 8))

    open_tv_btn = make_button(
        detail_buttons,
        "Open in TradingView",
        lambda: open_tradingview(selected_ticker["value"]) if selected_ticker["value"] else None,
    )
    open_tv_btn.pack(side="left", padx=(0, 8))

    detail_status_label = tk.Label(detail_page, textvariable=detail_status_var, fg=colors["accent"], bg=colors["bg"], justify="left", anchor="w")
    detail_status_label.pack(fill="x", padx=10, pady=(0, 6))

    detail_loading_bar = ttk.Progressbar(
        detail_page,
        orient="horizontal",
        mode="indeterminate",
        variable=detail_loading_progress,
        length=520,
        style="Dark.Horizontal.TProgressbar",
    )

    def show_detail_loading_bar():
        detail_loading_progress.set(0)
        if detail_loading_bar.winfo_manager() != "pack":
            detail_loading_bar.pack(fill="x", padx=10, pady=(0, 8), before=detail_status_label)
        detail_loading_bar.start(10)

    def hide_detail_loading_bar():
        detail_loading_bar.stop()
        detail_loading_bar.pack_forget()

    def _bind_open_chart(tree, ticker_index):
        def open_item(item_id):
            if not item_id:
                return
            values = tree.item(item_id).get("values", [])
            if len(values) > ticker_index and values[ticker_index]:
                listed_5d_pct = None
                if tree is ai_tree and values:
                    try:
                        listed_5d_pct = float(str(values[0]).replace("%", "").strip())
                    except (TypeError, ValueError):
                        listed_5d_pct = None
                load_stock_detail(values[ticker_index], listed_5d_pct=listed_5d_pct)

        def on_double_click(event):
            open_item(tree.identify_row(event.y))

        def on_enter_key(_event):
            selected_items = tree.selection()
            if selected_items:
                open_item(selected_items[0])

        tree.bind("<Double-1>", on_double_click)
        tree.bind("<Return>", on_enter_key)

    _bind_open_chart(ai_tree, 3)
    _bind_open_chart(macd_tree, 2)
    _bind_open_chart(rsi_tree, 2)


    def open_ticker_from_search(_event=None):
        if detail_task_in_progress["value"]:
            ticker_search_status_var.set("Please wait for the current detail request to finish.")
            return

        ticker = normalize_ticker_input(ticker_search_var.get())
        if not ticker:
            ticker_search_status_var.set("Enter a ticker symbol (for example: MSFT).")
            return

        ticker_search_status_var.set(f"Loading {ticker}...")
        load_stock_detail(ticker)

    ticker_search_btn = make_button(ticker_search_inner, "Go", open_ticker_from_search, accent=True)
    ticker_search_btn.pack(side="left", padx=(6, 0))

    ticker_search_entry.bind("<Return>", open_ticker_from_search)

    def render_ai(rows):
        for item_id in ai_tree.get_children():
            ai_tree.delete(item_id)
        for row in rows:
            ai_tree.insert(
                "",
                "end",
                values=(
                    f"{row.get('percentage', 0):.2f}%",
                    f"{row.get('confidence', 0):.2f}%",
                    row.get("stock_name", ""),
                    row.get("ticker", ""),
                    format_mcap(row.get("marketcap", 0)),
                ),
            )

    def render_macd(rows):
        for item_id in macd_tree.get_children():
            macd_tree.delete(item_id)
        for row in rows:
            macd_tree.insert(
                "",
                "end",
                values=(
                    row.get("signal_type", ""),
                    row.get("stock_name", ""),
                    row.get("ticker", ""),
                    f"{row.get('price', 0):.2f}",
                    f"{row.get('change_pct', 0):.2f}%",
                    format_mcap(row.get("marketcap", 0)),
                ),
            )

    def render_rsi(rows):
        for item_id in rsi_tree.get_children():
            rsi_tree.delete(item_id)
        for row in rows:
            rsi_tree.insert(
                "",
                "end",
                values=(
                    row.get("signal_type", ""),
                    row.get("stock_name", ""),
                    row.get("ticker", ""),
                    f"{row.get('rsi', 0):.2f}",
                    format_mcap(row.get("marketcap", 0)),
                ),
            )

    def sort_rows(table_name, rows):
        state = sort_states[table_name]
        column = state["column"]
        if not column:
            return rows

        return sorted(rows, key=lambda row: row.get(column, ""), reverse=not state["ascending"])

    def rerender_table(table_name):
        rows = sort_rows(table_name, rows_store[table_name])
        if table_name == "ai":
            render_ai(rows)
        elif table_name == "macd":
            render_macd(rows)
        elif table_name == "rsi":
            render_rsi(rows)

    def bind_sorting(tree, table_name, sortable_columns):
        for column in sortable_columns:
            heading_text = tree.heading(column, "text")

            def on_sort_click(col=column):
                state = sort_states[table_name]
                if state["column"] == col:
                    state["ascending"] = not state["ascending"]
                else:
                    state["column"] = col
                    state["ascending"] = True
                rerender_table(table_name)

            tree.heading(column, text=heading_text, command=on_sort_click)

    def populate_tables():
        rows_store["ai"] = load_ai_rows(csv_path)
        rows_store["macd"] = load_macd_rows()
        rows_store["rsi"] = load_rsi_rows()
        market_history_rows = load_market_forecast_history()
        market_forecast_history["rows"] = market_history_rows

        render_market_tabs()
        if selected_forecast_date["value"]:
            selected = next((item for item in market_history_rows if item["run_date"] == selected_forecast_date["value"]), None)
            render_market_forecast(selected.get("forecasts", []) if selected else [])
        else:
            market_rows = load_market_forecast_rows()
            render_market_forecast(market_rows)

        rerender_table("ai")
        rerender_table("macd")
        rerender_table("rsi")
        refresh_watchlist_cards()
        refresh_portfolio_table()

        total_market_rows = sum(len(item.get("forecasts", [])) for item in market_history_rows)
        status_var.set(
            f"Loaded AI={len(rows_store['ai'])}, MACD={len(rows_store['macd'])}, RSI={len(rows_store['rsi'])}, S&P forecast rows={total_market_rows}. Double-click any row to open on-demand stock analysis."
        )
        last_updated_var.set(f"Last updated at: {datetime.now().strftime('%Y-%m-%d %I:%M:%S %p')}")

    bind_sorting(ai_tree, "ai", ("percentage", "confidence", "ticker", "marketcap"))
    bind_sorting(macd_tree, "macd", ("ticker", "marketcap"))
    bind_sorting(rsi_tree, "rsi", ("ticker", "marketcap"))

    def estimate_eta_seconds(event, stage_start_times):
        stage = event.get("stage", "")
        current = max(event.get("current", 0), 0)
        total = max(event.get("total", 1), 1)
        now = time.time()

        if stage not in stage_start_times:
            stage_start_times[stage] = now

        if current <= 0:
            return None

        elapsed = now - stage_start_times[stage]
        avg_per_unit = elapsed / current
        remaining = max(total - current, 0)
        return int(avg_per_unit * remaining)

    def handle_progress_updates():
        processed_any = False
        while True:
            try:
                event = progress_queue.get_nowait()
            except queue.Empty:
                break

            processed_any = True
            stage = event.get("stage")
            current = max(event.get("current", 0), 0)
            total = max(event.get("total", 1), 1)
            message = event.get("message", "")

            if stage == "download":
                mapped_progress = 5
            elif stage == "prepare":
                mapped_progress = 5 + (current / total) * 45
            elif stage == "train":
                mapped_progress = 50 + (current / total) * 50
            elif stage == "scan":
                mapped_progress = (current / total) * 100
            elif stage == "done":
                mapped_progress = 100
            else:
                mapped_progress = progress_value.get()

            if stage == "detail_loaded":
                detail_task_in_progress["value"] = False
                render_stock_detail(event.get("payload", {}), event.get("listed_5d_pct"))
                if event.get("used_cache"):
                    detail_status_var.set("Detail refreshed with latest data.")
                continue
            if stage == "live_signal_loaded":
                live_signal_task_in_progress["value"] = False
                if event.get("ticker") == selected_ticker.get("value"):
                    render_live_signal_snapshot(event.get("snapshot"))
                continue
            if stage == "live_signal_error":
                live_signal_task_in_progress["value"] = False
                if event.get("ticker") == selected_ticker.get("value"):
                    detail_live_signal_var.set(f"Live 5m signal unavailable: {event.get('message', 'unknown error')}")
                continue
            if stage == "detail_error":
                detail_task_in_progress["value"] = False
                hide_detail_loading_bar()
                ticker = event.get("ticker", "")
                message = event.get("message", "Unknown error")
                if event.get("had_cached_payload") and ticker == selected_ticker.get("value"):
                    detail_status_var.set(
                        f"Live refresh failed, showing cached analysis instead. Reason: {message}"
                    )
                    print(event.get("trace", ""))
                    continue
                detail_header_var.set(f"{ticker} - analysis failed")
                detail_summary_var.set("Could not generate on-demand forecast/trade plan.")
                detail_forecast_var.set("")
                detail_live_signal_var.set("Live 5m signal unavailable.")
                clear_detail_chart("Unable to render chart for this ticker.")
                detail_status_var.set(message)
                show_detail_page()
                print(event.get("trace", ""))
                continue

            progress_value.set(mapped_progress)
            train_status_var.set(message)

            eta_seconds = estimate_eta_seconds(event, handle_progress_updates.stage_start_times)
            if stage == "done":
                eta_var.set("Estimated time left: 0s")
            elif eta_seconds is None:
                eta_var.set("Estimated time left: calculating...")
            else:
                minutes, seconds = divmod(eta_seconds, 60)
                eta_var.set(f"Estimated time left: {minutes}m {seconds}s")

            if stage == "done":
                task_in_progress["value"] = False
                progress_bar.pack_forget()
                train_btn.config(state="normal")
                run_scan_btn.config(state="normal")
                refresh_btn.config(state="normal")
                refresh_watchlist_cards()
                populate_tables()

        if task_in_progress["value"] or detail_task_in_progress["value"] or processed_any:
            root.after(250, handle_progress_updates)

    handle_progress_updates.stage_start_times = {}

    def run_global_training():
        if task_in_progress["value"]:
            return

        task_in_progress["value"] = True
        progress_value.set(0)
        progress_bar.pack(padx=10, pady=6, fill="x")
        train_status_var.set("Starting global model training...")
        eta_var.set("Estimated time left: calculating...")
        train_btn.config(state="disabled")
        run_scan_btn.config(state="disabled")
        refresh_btn.config(state="disabled")
        handle_progress_updates.stage_start_times = {}

        def worker():
            try:
                tickers = main.get_tickers()
                main.train_global_model(tickers, progress_callback=progress_queue.put)
                progress_queue.put({"stage": "done", "current": 1, "total": 1, "message": "Global model training complete."})
            except Exception as exc:
                progress_queue.put({"stage": "done", "current": 1, "total": 1, "message": f"Training failed: {exc}"})

        threading.Thread(target=worker, daemon=True).start()
        handle_progress_updates()

    def start_global_training():
        if task_in_progress["value"]:
            return

        first_confirm = messagebox.askyesno(
            "Train global model",
            "Are you sure you want to train the global model now?",
            parent=root,
        )
        if not first_confirm:
            train_status_var.set("Global model training cancelled.")
            return

        second_confirm = messagebox.askyesno(
            "Confirm training",
            "This can take several minutes. Start global model training now?",
            parent=root,
        )
        if not second_confirm:
            train_status_var.set("Global model training cancelled.")
            return

        run_global_training()

    def start_manual_scan():
        if task_in_progress["value"]:
            return

        task_in_progress["value"] = True
        progress_value.set(0)
        progress_bar.pack(padx=10, pady=6, fill="x")
        train_status_var.set("Starting manual daily scan...")
        eta_var.set("Estimated time left: this run can take a few minutes")
        train_btn.config(state="disabled")
        run_scan_btn.config(state="disabled")
        refresh_btn.config(state="disabled")
        handle_progress_updates.stage_start_times = {}

        def worker():
            try:
                tickers = main.get_tickers()
                main.run_daily(tickers=tickers, progress_callback=progress_queue.put)
                progress_queue.put({"stage": "done", "current": 1, "total": 1, "message": "Manual daily scan complete. CSV files updated."})
            except Exception as exc:
                progress_queue.put({"stage": "done", "current": 1, "total": 1, "message": f"Manual scan failed: {exc}"})

        threading.Thread(target=worker, daemon=True).start()
        handle_progress_updates()

    button_frame = tk.Frame(content_frame, bg=colors["bg"])
    button_frame.pack(pady=6)

    refresh_btn = make_button(button_frame, "Refresh CSV Tabs", populate_tables, accent=True)
    refresh_btn.pack(side="left", padx=5)

    train_btn = make_button(button_frame, "Train Global Model", start_global_training)
    train_btn.pack(side="left", padx=5)

    run_scan_btn = make_button(button_frame, "Run Daily Scan Now", start_manual_scan)
    run_scan_btn.pack(side="left", padx=5)

    def clear_cached_stock_details():
        if not messagebox.askyesno(
            "Delete cached stock details",
            "This will delete all cached individual stock detail data (predictions, stop loss, take profit, chart snapshot).\n\n"
            "It will NOT delete your trained AI model files or the 5-day list CSV outputs. Continue?",
            parent=root,
        ):
            return

        deleted = main.clear_all_stock_trade_plan_cache()
        detail_status_var.set(f"Cleared cached stock details for {deleted} ticker(s).")
        messagebox.showinfo("Cache cleared", f"Deleted cached detail data for {deleted} ticker(s).", parent=root)

    clear_details_btn = make_button(button_frame, "Delete Stock Detail Cache", clear_cached_stock_details)
    clear_details_btn.pack(side="left", padx=5)

    close_btn = make_button(button_frame, "Close", root.destroy)
    close_btn.pack(side="left", padx=5)

    hint = tk.Label(
        content_frame,
        text="Tip: Daily scan now writes AI (buy_signals.csv), MACD (macd_signals.csv), and RSI (rsi_signals.csv).",
        fg=colors["muted"],
        bg=colors["bg"],
    )
    hint.pack(pady=6)

    list_page_widgets.extend(
        [
            title_label,
            market_frame,
            notebook,
            status_label,
            progress_bar,
            train_status_label,
            eta_label,
            button_frame,
            hint,
        ]
    )

    populate_tables()
    root.mainloop()


if __name__ == "__main__":
    launch_signals_ui()
