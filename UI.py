import os
import webbrowser
import tkinter as tk
from tkinter import ttk

import pandas as pd

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
