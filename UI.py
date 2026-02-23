import os
import queue
import threading
import time
import webbrowser
import tkinter as tk
from tkinter import ttk
from datetime import datetime

import pandas as pd

import main

SIGNALS_CSV_PATH = "buy_signals.csv"
MACD_SIGNALS_CSV_PATH = "macd_signals.csv"
RSI_SIGNALS_CSV_PATH = "rsi_signals.csv"
MARKET_FORECAST_CSV_PATH = "sp500_forecast.csv"
TV_LAYOUT_ID = "ClEM8BLT"
FORECAST_TABS_DAYS_TO_KEEP = 5


def open_tradingview(ticker):
    """Open TradingView chart for a ticker, handling TSX suffixes."""
    symbol = str(ticker).upper()
    if symbol.endswith('.TO'):
        tv_symbol = f"TSX:{symbol.replace('.TO', '')}"
    else:
        tv_symbol = symbol

    url = f"https://www.tradingview.com/chart/{TV_LAYOUT_ID}/?symbol={tv_symbol}"
    webbrowser.open(url)


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

    title_label = tk.Label(root, text="Flat tabs: AI, MACD, and RSI", font=("Arial", 13, "bold"))
    title_label.pack(pady=8)

    market_frame = tk.Frame(root, bd=1, relief="groove", padx=8, pady=6)
    market_frame.pack(fill="x", padx=10, pady=(0, 6))

    market_title = tk.Label(market_frame, text="S&P 500 Forecast (next 5 trading days)", font=("Arial", 11, "bold"))
    market_title.pack(anchor="w")

    market_forecast_var = tk.StringVar(value="No S&P 500 forecast data yet. Run a daily scan to generate it.")
    market_forecast_label = tk.Label(market_frame, textvariable=market_forecast_var, font=("Arial", 10), justify="left", anchor="w")
    market_forecast_label.pack(fill="x", pady=(2, 0))

    market_tabs_frame = tk.Frame(market_frame)
    market_tabs_frame.pack(fill="x", pady=(6, 0))

    notebook = ttk.Notebook(root)
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
                relief="sunken" if is_selected else "raised",
                padx=10,
                command=lambda d=run_date: on_market_tab_selected(d),
            ).pack(side="left", padx=(0, 6))

    def on_market_tab_selected(run_date):
        selected_forecast_date["value"] = run_date
        render_market_tabs()
        selected = next((item for item in market_forecast_history["rows"] if item["run_date"] == run_date), None)
        render_market_forecast(selected.get("forecasts", []) if selected else [])

    def build_tree_tab(tab_title, columns, headings, widths):
        frame = ttk.Frame(notebook)
        notebook.add(frame, text=tab_title)
        tree = ttk.Treeview(frame, columns=columns, show="headings")
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

    status_var = tk.StringVar(value="")
    status_label = tk.Label(root, textvariable=status_var, fg="blue")
    status_label.pack(pady=4)

    train_status_var = tk.StringVar(value="")
    eta_var = tk.StringVar(value="Estimated time left: --")
    progress_value = tk.DoubleVar(value=0.0)
    task_in_progress = {"value": False}
    progress_queue = queue.Queue()

    progress_bar = ttk.Progressbar(root, orient="horizontal", mode="determinate", length=700, variable=progress_value)
    progress_bar.pack(padx=10, pady=6, fill="x")
    progress_bar.pack_forget()

    train_status_label = tk.Label(root, textvariable=train_status_var, fg="green")
    train_status_label.pack(pady=2)

    eta_label = tk.Label(root, textvariable=eta_var, fg="gray")
    eta_label.pack(pady=2)

    def _bind_open_chart(tree, ticker_index):
        def open_item(item_id):
            if not item_id:
                return
            values = tree.item(item_id).get("values", [])
            if len(values) > ticker_index and values[ticker_index]:
                open_tradingview(values[ticker_index])

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

        total_market_rows = sum(len(item.get("forecasts", [])) for item in market_history_rows)
        status_var.set(
            f"Loaded AI={len(rows_store['ai'])}, MACD={len(rows_store['macd'])}, RSI={len(rows_store['rsi'])}, S&P forecast rows={total_market_rows}. Double-click any row to open TradingView."
        )

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
                populate_tables()

        if task_in_progress["value"] or processed_any:
            root.after(250, handle_progress_updates)

    handle_progress_updates.stage_start_times = {}

    def start_global_training():
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

    button_frame = tk.Frame(root)
    button_frame.pack(pady=6)

    refresh_btn = tk.Button(button_frame, text="Refresh CSV Tabs", command=populate_tables)
    refresh_btn.pack(side="left", padx=5)

    train_btn = tk.Button(button_frame, text="Train Global Model", command=start_global_training)
    train_btn.pack(side="left", padx=5)

    run_scan_btn = tk.Button(button_frame, text="Run Daily Scan Now", command=start_manual_scan)
    run_scan_btn.pack(side="left", padx=5)

    close_btn = tk.Button(button_frame, text="Close", command=root.destroy)
    close_btn.pack(side="left", padx=5)

    hint = tk.Label(
        root,
        text="Tip: Daily scan now writes AI (buy_signals.csv), MACD (macd_signals.csv), and RSI (rsi_signals.csv).",
        fg="gray",
    )
    hint.pack(pady=6)

    populate_tables()
    root.mainloop()


if __name__ == "__main__":
    launch_signals_ui()
