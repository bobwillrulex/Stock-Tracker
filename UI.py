import os
import queue
import threading
import time
import traceback
import webbrowser
import tkinter as tk
from tkinter import ttk
from datetime import datetime

import pandas as pd

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
    detail_task_in_progress = {"value": False}
    progress_queue = queue.Queue()

    progress_bar = ttk.Progressbar(root, orient="horizontal", mode="determinate", length=700, variable=progress_value)
    progress_bar.pack(padx=10, pady=6, fill="x")
    progress_bar.pack_forget()

    train_status_label = tk.Label(root, textvariable=train_status_var, fg="green")
    train_status_label.pack(pady=2)

    eta_label = tk.Label(root, textvariable=eta_var, fg="gray")
    eta_label.pack(pady=2)

    current_page = {"value": "list"}
    list_page_widgets = []
    list_page_pack_state = {}
    detail_page = tk.Frame(root)

    detail_header_var = tk.StringVar(value="Stock detail")
    detail_summary_var = tk.StringVar(value="")
    detail_forecast_var = tk.StringVar(value="")
    detail_status_var = tk.StringVar(value="")
    detail_loading_progress = tk.DoubleVar(value=0)

    detail_header_label = tk.Label(detail_page, textvariable=detail_header_var, font=("Arial", 13, "bold"))
    detail_header_label.pack(anchor="w", padx=10, pady=(10, 6))

    detail_summary_label = tk.Label(detail_page, textvariable=detail_summary_var, justify="left", anchor="w", font=("Arial", 10))
    detail_summary_label.pack(fill="x", padx=10)

    detail_forecast_label = tk.Label(detail_page, textvariable=detail_forecast_var, justify="left", anchor="w", font=("Arial", 10))
    detail_forecast_label.pack(fill="x", padx=10, pady=(6, 10))

    detail_buttons = tk.Frame(detail_page)
    detail_buttons.pack(anchor="w", padx=10, pady=(0, 8))

    selected_ticker = {"value": None}
    chart_container = tk.Frame(detail_page)
    chart_container.pack(expand=True, fill="both", padx=10, pady=(2, 10))
    chart_canvas = {"value": None}

    def clear_detail_chart(message=""):
        if chart_canvas["value"] is not None:
            chart_canvas["value"].get_tk_widget().destroy()
            chart_canvas["value"] = None
        for widget in chart_container.winfo_children():
            widget.destroy()
        if message:
            tk.Label(chart_container, text=message, fg="gray", anchor="w", justify="left").pack(anchor="w")

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

        figure = Figure(figsize=(10.5, 4.8), dpi=100)
        axis = figure.add_subplot(111)
        axis.set_title(f"{payload.get('ticker', '')} price action + 5-day projection")
        axis.set_xlabel("Date")
        axis.set_ylabel("Price (USD)")

        x_dates = candle_df["date"].tolist()
        x_nums = mdates.date2num(x_dates)
        candle_width = 0.6
        color_up = "#2E8B57"
        color_down = "#C0392B"

        from matplotlib.patches import Rectangle

        for x_num, o, h, l, c in zip(x_nums, candle_df["open"], candle_df["high"], candle_df["low"], candle_df["close"]):
            color = color_up if c >= o else color_down
            axis.vlines(x_num, l, h, color=color, linewidth=1.0, zorder=2)
            lower = min(o, c)
            height = max(abs(c - o), 0.02)
            axis.add_patch(
                Rectangle(
                    (x_num - candle_width / 2, lower),
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

        if projected_dates and projected_prices:
            axis.plot(projected_dates, projected_prices, color="#1f77b4", marker="o", linewidth=1.8, label="Predicted next 5 days", zorder=4)
            axis.scatter([last_date], [last_close], color="#1f77b4", s=24, zorder=5)

        stop_loss = trade_plan.get("stop_loss")
        take_profit = trade_plan.get("take_profit")
        if isinstance(stop_loss, (int, float)):
            axis.axhline(stop_loss, linestyle="--", color="#8E44AD", linewidth=1.2, label=f"Stop loss ${stop_loss:.2f}")
        if isinstance(take_profit, (int, float)):
            axis.axhline(take_profit, linestyle="--", color="#D35400", linewidth=1.2, label=f"Take profit ${take_profit:.2f}")

        all_prices = candle_df[["low", "high"]].to_numpy().ravel().tolist() + projected_prices
        if isinstance(stop_loss, (int, float)):
            all_prices.append(float(stop_loss))
        if isinstance(take_profit, (int, float)):
            all_prices.append(float(take_profit))

        min_price = min(all_prices)
        max_price = max(all_prices)
        spread = max(max_price - min_price, 0.5)
        axis.set_ylim(min_price - spread * 0.08, max_price + spread * 0.12)

        x_end = projected_dates[-1] if projected_dates else last_date
        axis.set_xlim(candle_df["date"].iloc[0] - pd.Timedelta(days=1), x_end + pd.Timedelta(days=1))
        axis.xaxis.set_major_locator(mdates.AutoDateLocator(minticks=6, maxticks=10))
        axis.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
        axis.tick_params(axis="x", labelrotation=25)
        from matplotlib.ticker import MaxNLocator
        axis.yaxis.set_major_locator(MaxNLocator(nbins=8))
        axis.grid(True, linestyle=":", linewidth=0.7, alpha=0.6)
        axis.legend(loc="upper left", fontsize=9)

        figure.tight_layout()
        tk_canvas = FigureCanvasTkAgg(figure, master=chart_container)
        tk_canvas.draw()
        tk_canvas.get_tk_widget().pack(expand=True, fill="both")
        chart_canvas["value"] = tk_canvas

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

    def render_stock_detail(payload):
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

        summary_lines = [
            f"Current price: ${payload.get('current_price', 0):.2f}",
            f"Support used for SL: ${trade_plan.get('support_level', 0):.2f} (AI strength: {support_strength_text})",
            f"Nearest resistance: {resistance_text} (AI strength: {resistance_strength_text})",
            f"Recommended stop loss (support - buffer): ${trade_plan.get('stop_loss', 0):.2f}",
            f"Recommended take profit (SR-led): ${trade_plan.get('take_profit', 0):.2f} (SR target: {sr_target_text})",
            f"SR selection method: {method}",
            f"Position size: {trade_plan.get('position_size_pct', 0) * 100:.1f}% of capital",
            f"Estimated shares (assuming $100,000 capital): {trade_plan.get('shares', 0)}",
            f"Avg predicted return: {trade_plan.get('avg_predicted_return', 0) * 100:+.2f}% | Avg confidence: {trade_plan.get('avg_confidence', 0) * 100:.1f}%",
        ]
        detail_summary_var.set("\n".join(summary_lines))

        day_lines = []
        for row in forecast:
            day = int(row.get("day", 0))
            day_lines.append(
                f"Day {day}: {row.get('predicted_return', 0) * 100:+.2f}% (conf {row.get('confidence', 0) * 100:.1f}%) -> ${row.get('projected_price', 0):.2f}"
            )
        detail_forecast_var.set("\n".join(day_lines))
        render_detail_chart(payload)
        detail_status_var.set("")
        show_detail_page()

    def load_stock_detail(ticker):
        detail_task_in_progress["value"] = True
        show_detail_loading_bar()
        detail_header_var.set(f"{ticker} - loading analysis...")
        detail_summary_var.set("Computing on-demand forecast and trade plan...")
        detail_forecast_var.set("")
        clear_detail_chart("Loading chart data...")
        detail_status_var.set("")
        show_detail_page()

        def worker():
            try:
                payload = main.generate_stock_trade_plan(ticker)
                progress_queue.put({"stage": "detail_loaded", "payload": payload})
            except Exception as exc:
                progress_queue.put({"stage": "detail_error", "ticker": ticker, "message": str(exc), "trace": traceback.format_exc()})

        threading.Thread(target=worker, daemon=True).start()
        handle_progress_updates()

    back_btn = tk.Button(detail_buttons, text="← Back to list", command=show_list_page)
    back_btn.pack(side="left", padx=(0, 8))

    open_tv_btn = tk.Button(
        detail_buttons,
        text="Open in TradingView",
        command=lambda: open_tradingview(selected_ticker["value"]) if selected_ticker["value"] else None,
    )
    open_tv_btn.pack(side="left", padx=(0, 8))

    detail_status_label = tk.Label(detail_page, textvariable=detail_status_var, fg="blue", justify="left", anchor="w")
    detail_status_label.pack(fill="x", padx=10, pady=(0, 6))

    detail_loading_bar = ttk.Progressbar(
        detail_page,
        orient="horizontal",
        mode="indeterminate",
        variable=detail_loading_progress,
        length=520,
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
                load_stock_detail(values[ticker_index])

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
            f"Loaded AI={len(rows_store['ai'])}, MACD={len(rows_store['macd'])}, RSI={len(rows_store['rsi'])}, S&P forecast rows={total_market_rows}. Double-click any row to open on-demand stock analysis."
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

            if stage == "detail_loaded":
                detail_task_in_progress["value"] = False
                render_stock_detail(event.get("payload", {}))
                continue
            if stage == "detail_error":
                detail_task_in_progress["value"] = False
                hide_detail_loading_bar()
                ticker = event.get("ticker", "")
                message = event.get("message", "Unknown error")
                detail_header_var.set(f"{ticker} - analysis failed")
                detail_summary_var.set("Could not generate on-demand forecast/trade plan.")
                detail_forecast_var.set("")
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
                populate_tables()

        if task_in_progress["value"] or detail_task_in_progress["value"] or processed_any:
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
