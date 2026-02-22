import os
import queue
import threading
import time
import webbrowser
import tkinter as tk
from tkinter import ttk

import pandas as pd

import main

SIGNALS_CSV_PATH = "buy_signals.csv"
TV_LAYOUT_ID = "ClEM8BLT"


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


def load_signals_from_csv(path=SIGNALS_CSV_PATH):
    """Load and normalize signal rows from CSV."""
    if not os.path.exists(path):
        return []

    df = pd.read_csv(path)
    if df.empty:
        return []

    expected_cols = ["percentage", "stock_name", "ticker", "marketcap"]
    for col in expected_cols:
        if col not in df.columns:
            df[col] = ""

    df["marketcap"] = pd.to_numeric(df["marketcap"], errors="coerce").fillna(0)
    df["percentage"] = pd.to_numeric(df["percentage"], errors="coerce").fillna(0)
    df = df.sort_values(by="marketcap", ascending=False)

    return df.to_dict("records")


def launch_signals_ui(csv_path=SIGNALS_CSV_PATH):
    root = tk.Tk()
    root.title("Buy Signals Viewer (CSV)")
    root.geometry("900x600")

    title_label = tk.Label(
        root,
        text=f"Saved Buy Signals from {csv_path}",
        font=("Arial", 13, "bold"),
    )
    title_label.pack(pady=8)

    columns = ("percentage", "stock_name", "ticker", "marketcap")
    tree = ttk.Treeview(root, columns=columns, show='headings')

    tree.heading("percentage", text="Predicted %")
    tree.heading("stock_name", text="Stock")
    tree.heading("ticker", text="Ticker")
    tree.heading("marketcap", text="Market Cap")

    tree.column("percentage", width=120, anchor="center")
    tree.column("stock_name", width=300)
    tree.column("ticker", width=120, anchor="center")
    tree.column("marketcap", width=160, anchor="e")

    tree.pack(expand=True, fill='both', padx=10, pady=10)

    status_var = tk.StringVar(value="")
    status_label = tk.Label(root, textvariable=status_var, fg="blue")
    status_label.pack(pady=4)

    train_status_var = tk.StringVar(value="")
    eta_var = tk.StringVar(value="Estimated time left: --")
    progress_value = tk.DoubleVar(value=0.0)
    task_in_progress = {"value": False}
    progress_queue = queue.Queue()

    progress_bar = ttk.Progressbar(
        root,
        orient="horizontal",
        mode="determinate",
        length=600,
        variable=progress_value,
    )
    progress_bar.pack(padx=10, pady=6, fill="x")

    train_status_label = tk.Label(root, textvariable=train_status_var, fg="green")
    train_status_label.pack(pady=2)

    eta_label = tk.Label(root, textvariable=eta_var, fg="gray")
    eta_label.pack(pady=2)

    def populate_table():
        for item_id in tree.get_children():
            tree.delete(item_id)

        rows = load_signals_from_csv(csv_path)
        if not rows:
            status_var.set("No saved signals found. Run the scanner to generate buy_signals.csv.")
            return

        for row in rows:
            percentage_text = f"{row.get('percentage', 0):.2f}%"
            tree.insert(
                '',
                'end',
                values=(
                    percentage_text,
                    row.get('stock_name', ''),
                    row.get('ticker', ''),
                    format_mcap(row.get('marketcap', 0)),
                ),
            )

        status_var.set(f"Loaded {len(rows)} signals. Double-click a row to open TradingView.")

    def on_double_click(event):
        item_id = tree.identify_row(event.y)
        if not item_id:
            return

        values = tree.item(item_id).get("values", [])
        if len(values) < 3:
            return

        ticker = values[2]
        if ticker:
            open_tradingview(ticker)

    tree.bind("<Double-1>", on_double_click)

    button_frame = tk.Frame(root)
    button_frame.pack(pady=6)

    refresh_btn = tk.Button(button_frame, text="Refresh from CSV", command=populate_table)
    refresh_btn.pack(side="left", padx=5)

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

            # Map stage progress into global 0-100 progress bar
            if stage == "download":
                mapped_progress = 5
            elif stage == "prepare":
                mapped_progress = 5 + (current / total) * 45
            elif stage == "train":
                mapped_progress = 50 + (current / total) * 50
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
                train_btn.config(state="normal")
                run_scan_btn.config(state="normal")
                refresh_btn.config(state="normal")

        if task_in_progress["value"] or processed_any:
            root.after(250, handle_progress_updates)

    handle_progress_updates.stage_start_times = {}

    def start_global_training():
        if task_in_progress["value"]:
            return

        task_in_progress["value"] = True
        progress_value.set(0)
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
                progress_queue.put({
                    "stage": "done",
                    "current": 1,
                    "total": 1,
                    "message": "Global model training complete.",
                })
            except Exception as exc:
                progress_queue.put({
                    "stage": "done",
                    "current": 1,
                    "total": 1,
                    "message": f"Training failed: {exc}",
                })

        threading.Thread(target=worker, daemon=True).start()
        handle_progress_updates()

    def start_manual_scan():
        if task_in_progress["value"]:
            return

        task_in_progress["value"] = True
        progress_value.set(0)
        train_status_var.set("Starting manual daily scan...")
        eta_var.set("Estimated time left: this run can take a few minutes")
        train_btn.config(state="disabled")
        run_scan_btn.config(state="disabled")
        refresh_btn.config(state="disabled")
        handle_progress_updates.stage_start_times = {}

        def worker():
            try:
                tickers = main.get_tickers()
                main.run_daily(tickers=tickers)
                progress_queue.put({
                    "stage": "done",
                    "current": 1,
                    "total": 1,
                    "message": "Manual daily scan complete. Buy signals CSV updated.",
                })
            except Exception as exc:
                progress_queue.put({
                    "stage": "done",
                    "current": 1,
                    "total": 1,
                    "message": f"Manual scan failed: {exc}",
                })

        threading.Thread(target=worker, daemon=True).start()
        handle_progress_updates()

    train_btn = tk.Button(button_frame, text="Train Global Model", command=start_global_training)
    train_btn.pack(side="left", padx=5)

    run_scan_btn = tk.Button(button_frame, text="Run Daily Scan Now", command=start_manual_scan)
    run_scan_btn.pack(side="left", padx=5)

    close_btn = tk.Button(button_frame, text="Close", command=root.destroy)
    close_btn.pack(side="left", padx=5)

    hint = tk.Label(
        root,
        text="Tip: You can close and reopen this UI anytime; it reads persisted CSV data.",
        fg="gray",
    )
    hint.pack(pady=6)

    populate_table()
    root.mainloop()


if __name__ == "__main__":
    launch_signals_ui()
