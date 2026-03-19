import threading
import base64
import io
from datetime import datetime

from flask import Flask, flash, jsonify, redirect, render_template, request, url_for

import main
import pandas as pd
from UI import (
    compute_recommendation_score,
    delete_portfolio_position,
    delete_watchlist_item,
    format_mcap,
    insert_portfolio_position,
    load_ai_rows,
    load_macd_rows,
    load_market_forecast_history,
    load_market_forecast_rows,
    load_portfolio_positions,
    load_fvg_rows,
    load_rsi_rows,
    load_watchlist_items,
    normalize_ticker_input,
    score_to_action,
    upsert_watchlist_item,
)

app = Flask(__name__)
app.secret_key = "stock-tracker-flask-ui"

_state = {
    "scan_running": False,
    "scan_message": "Idle",
    "scan_started_at": None,
    "last_scan_completed_at": None,
    "last_scan_failed_at": None,
}


def _to_float(value, default=0.0):
    try:
        return float(value)
    except Exception:
        return default


def _latest_price_and_change(ticker):
    try:
        return main._fetch_latest_price_and_change(ticker)
    except Exception:
        return None, None


def _ai_prediction_map():
    data = {}
    for row in load_ai_rows():
        ticker = str(row.get("ticker", "")).upper()
        if ticker:
            data[ticker] = _to_float(row.get("percentage"))
    return data


def _enrich_portfolio_rows():
    enriched = []
    ai_map = _ai_prediction_map()
    name_map = {}
    for row in load_ai_rows() + load_macd_rows() + load_rsi_rows():
        ticker = str(row.get("ticker", "")).upper()
        if ticker and ticker not in name_map:
            name_map[ticker] = str(row.get("stock_name", "")).strip()

    for pos in load_portfolio_positions():
        ticker = pos["ticker"].upper()
        price, _ = _latest_price_and_change(ticker)
        shares = _to_float(pos.get("shares"))
        cost_basis = _to_float(pos.get("cost_basis"))
        total_cost = shares * cost_basis

        pnl = None
        pnl_pct = None
        signal = "HOLD 50/100"
        if isinstance(price, (int, float)) and total_cost > 0:
            current_value = shares * float(price)
            pnl = current_value - total_cost
            pnl_pct = (pnl / total_cost) * 100
            score = compute_recommendation_score(pnl_pct, ai_map.get(ticker))
            signal = f"{score_to_action(score)} {score:.0f}/100"

        enriched.append(
            {
                **pos,
                "stock_name": name_map.get(ticker, ticker),
                "price": price,
                "pnl": pnl,
                "pnl_pct": pnl_pct,
                "signal": signal,
            }
        )

    return enriched


def _enrich_watchlist_rows():
    rows = []
    ai_map = _ai_prediction_map()
    for item in load_watchlist_items():
        ticker = item["ticker"].upper()
        price, change = _latest_price_and_change(ticker)
        rows.append(
            {
                "ticker": ticker,
                "name": item.get("name") or ticker,
                "price": price,
                "change_pct": change,
                "ai_5d": ai_map.get(ticker),
                "tv_symbol": main._to_tv_symbol(ticker),
            }
        )
    return rows


def _scan_job():
    _state["scan_running"] = True
    _state["scan_started_at"] = datetime.utcnow().isoformat()
    _state["scan_message"] = "Running full market scan..."
    try:
        main.run_daily()
        completed_at = datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")
        _state["last_scan_completed_at"] = completed_at
        _state["last_scan_failed_at"] = None
        _state["scan_message"] = f"Last completed at {completed_at}"
    except Exception as exc:
        failed_at = datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")
        _state["last_scan_failed_at"] = failed_at
        _state["scan_message"] = f"Scan failed: {exc}"
    finally:
        _state["scan_running"] = False
        _state["scan_started_at"] = None


def _build_live_signal_text(snapshot):
    if not snapshot:
        return "No live signal snapshot available."

    buy = _to_float(snapshot.get("buy_pct"))
    hold = _to_float(snapshot.get("hold_pct"))
    sell = _to_float(snapshot.get("sell_pct"))
    action = str(snapshot.get("action") or "HOLD")
    confidence = _to_float(snapshot.get("confidence"))
    captured_at = str(snapshot.get("captured_at_utc") or snapshot.get("generated_at") or "")
    market_open = bool(snapshot.get("market_open", False))
    state = "market open" if market_open else "market closed"
    return (
        f"Live 5m signal ({state}): BUY {buy:.1f}% | HOLD {hold:.1f}% | SELL {sell:.1f}% "
        f"→ {action} (conf {confidence:.1f}%) | updated {captured_at}"
    )


def _build_detail_chart_base64(payload):
    try:
        import matplotlib
        matplotlib.use("Agg")
        from matplotlib.figure import Figure
        from matplotlib.patches import Rectangle
    except Exception:
        return None

    candles = payload.get("daily_candles", [])
    forecast = sorted(payload.get("forecast", []), key=lambda item: int(item.get("day", 0)))
    trade_plan = payload.get("trade_plan", {})
    if not candles:
        return None

    candle_df = pd.DataFrame(candles)
    candle_df["date"] = pd.to_datetime(candle_df["date"], errors="coerce")
    candle_df = candle_df.dropna(subset=["date", "open", "high", "low", "close"]).sort_values("date")
    if candle_df.empty:
        return None

    figure = Figure(figsize=(11, 7), dpi=120)
    grid = figure.add_gridspec(3, 1, height_ratios=[70, 15, 15], hspace=0.06)
    price_axis = figure.add_subplot(grid[0])
    macd_axis = figure.add_subplot(grid[1], sharex=price_axis)
    rsi_axis = figure.add_subplot(grid[2], sharex=price_axis)

    history_count = len(candle_df)
    x_positions = list(range(history_count))
    for x_pos, o, h, l, c in zip(x_positions, candle_df["open"], candle_df["high"], candle_df["low"], candle_df["close"]):
        color = "#2E8B57" if c >= o else "#C0392B"
        price_axis.vlines(x_pos, l, h, color=color, linewidth=1.0, zorder=2)
        lower = min(o, c)
        height = max(abs(c - o), 0.02)
        price_axis.add_patch(Rectangle((x_pos - 0.3, lower), 0.6, height, facecolor=color, edgecolor=color, linewidth=0.6, zorder=3))

    last_date = candle_df["date"].iloc[-1]
    last_close = float(candle_df["close"].iloc[-1])
    projected_positions = [history_count - 1 + int(row.get("day", 0)) for row in forecast if int(row.get("day", 0)) > 0]
    projected_prices = [float(row.get("projected_price", last_close)) for row in forecast if int(row.get("day", 0)) > 0]
    if projected_positions and projected_prices:
        price_axis.plot(projected_positions, projected_prices, color="#1f77b4", marker="o", linewidth=1.8, label="Predicted next 5 days", zorder=4)
        price_axis.scatter([history_count - 1], [last_close], color="#1f77b4", s=24, zorder=5)

    stop_loss = trade_plan.get("stop_loss")
    take_profit = trade_plan.get("take_profit")
    if isinstance(stop_loss, (int, float)):
        price_axis.axhline(stop_loss, linestyle="--", color="#8E44AD", linewidth=1.2, label=f"Stop loss ${stop_loss:.2f}")
    if isinstance(take_profit, (int, float)):
        price_axis.axhline(take_profit, linestyle="--", color="#D35400", linewidth=1.2, label=f"Take profit ${take_profit:.2f}")

    price_axis.set_title(f"{payload.get('ticker', '')} price action + 5-day projection")
    price_axis.set_ylabel("Price (USD)")
    price_axis.grid(True, linestyle=":", linewidth=0.7, alpha=0.6)
    price_axis.legend(loc="upper left", fontsize=8)

    close_series = pd.to_numeric(candle_df["close"], errors="coerce")
    ema_fast = close_series.ewm(span=12, adjust=False).mean()
    ema_slow = close_series.ewm(span=26, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    macd_signal = macd_line.ewm(span=9, adjust=False).mean()
    macd_hist = macd_line - macd_signal
    macd_axis.bar(x_positions, macd_hist, width=0.8, color=["#2E8B57" if v >= 0 else "#C0392B" for v in macd_hist], alpha=0.5)
    macd_axis.plot(x_positions, macd_line, color="#1f77b4", linewidth=1.2, label="MACD")
    macd_axis.plot(x_positions, macd_signal, color="#E67E22", linewidth=1.2, label="Signal")
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
    rsi_axis.plot(x_positions, rsi, color="#8E44AD", linewidth=1.3, label="RSI (14)")
    rsi_axis.axhline(70, color="#C0392B", linestyle="--", linewidth=1.0)
    rsi_axis.axhline(30, color="#2E8B57", linestyle="--", linewidth=1.0)
    rsi_axis.set_ylim(0, 100)
    rsi_axis.set_ylabel("RSI")
    rsi_axis.grid(True, linestyle=":", linewidth=0.7, alpha=0.5)
    rsi_axis.legend(loc="upper left", fontsize=8)

    tick_positions = sorted(set([0, max(0, history_count // 2), history_count - 1] + projected_positions))
    future_label_map = {
        history_count - 1 + int(row.get("day", 0)): (last_date + pd.offsets.BDay(int(row.get("day", 0)))).strftime("%Y-%m-%d")
        for row in forecast
        if int(row.get("day", 0)) > 0
    }
    tick_labels = [
        candle_df["date"].iloc[pos].strftime("%Y-%m-%d") if pos < history_count else future_label_map.get(pos, "")
        for pos in tick_positions
    ]
    rsi_axis.set_xticks(tick_positions)
    rsi_axis.set_xticklabels(tick_labels, rotation=25, ha="right")
    rsi_axis.set_xlabel("Date")
    price_axis.tick_params(axis="x", labelbottom=False)
    macd_axis.tick_params(axis="x", labelbottom=False)

    figure.tight_layout()
    buffer = io.BytesIO()
    figure.savefig(buffer, format="png", bbox_inches="tight")
    buffer.seek(0)
    return base64.b64encode(buffer.read()).decode("utf-8")


@app.context_processor
def inject_scan_state():
    return {"scan_state": _state}


@app.route("/")
def dashboard():
    forecast_history = load_market_forecast_history()
    selected_date = request.args.get("forecast_date")
    selected = None
    if forecast_history:
        if not selected_date:
            selected = forecast_history[0]
            selected_date = selected["run_date"]
        else:
            selected = next((r for r in forecast_history if r["run_date"] == selected_date), forecast_history[0])
            selected_date = selected["run_date"]

    return render_template(
        "dashboard.html",
        ai_rows=load_ai_rows(),
        forecast_rows=selected["forecasts"] if selected else load_market_forecast_rows(),
        forecast_history=forecast_history,
        selected_date=selected_date,
        scan_state=_state,
        format_mcap=format_mcap,
    )


@app.route("/rules")
def rules():
    return render_template(
        "rules.html",
        macd_rows=load_macd_rows(),
        rsi_rows=load_rsi_rows(),
        fvg_rows=load_fvg_rows(),
        scan_state=_state,
    )


@app.route("/portfolio")
def portfolio():
    return render_template(
        "portfolio.html",
        positions=_enrich_portfolio_rows(),
        watchlist=_enrich_watchlist_rows(),
        scan_state=_state,
    )


@app.post("/portfolio/add")
def add_portfolio():
    try:
        insert_portfolio_position(request.form.get("ticker"), request.form.get("shares"), request.form.get("cost_basis"))
        flash("Position added.", "success")
    except Exception as exc:
        flash(str(exc), "error")
    return redirect(url_for("portfolio"))


@app.post("/portfolio/delete/<int:position_id>")
def remove_portfolio(position_id):
    delete_portfolio_position(position_id)
    flash("Position removed.", "success")
    return redirect(url_for("portfolio"))


@app.post("/watchlist/add")
def add_watchlist():
    ticker = normalize_ticker_input(request.form.get("ticker", ""))
    name = request.form.get("name", "")
    if not ticker:
        flash("Ticker is required.", "error")
    else:
        upsert_watchlist_item(ticker, name)
        flash(f"Added {ticker} to watchlist.", "success")
    return redirect(url_for("portfolio"))


@app.post("/watchlist/delete/<ticker>")
def remove_watchlist(ticker):
    delete_watchlist_item(ticker)
    flash(f"Removed {ticker.upper()} from watchlist.", "success")
    return redirect(url_for("portfolio"))


@app.route("/stock/<ticker>")
def stock_detail(ticker):
    symbol = normalize_ticker_input(ticker)
    payload = main.load_stock_trade_plan_cache(symbol) or main.generate_stock_trade_plan(symbol)
    if not payload:
        flash("Could not load stock detail.", "error")
        return redirect(url_for("dashboard"))
    live = main.load_latest_live_signal_snapshot(symbol)
    forecast = sorted(payload.get("forecast", []), key=lambda row: int(row.get("day", 0)))
    trade_plan = payload.get("trade_plan", {})
    support_strength = trade_plan.get("support_strength")
    resistance_strength = trade_plan.get("resistance_strength")
    detail_summary = {
        "day1_pct": next((row.get("predicted_return", 0.0) for row in forecast if int(row.get("day", 0)) == 1), 0.0) * 100,
        "day2_pct": next((row.get("predicted_return", 0.0) for row in forecast if int(row.get("day", 0)) == 2), 0.0) * 100,
        "day3_pct": next((row.get("predicted_return", 0.0) for row in forecast if int(row.get("day", 0)) == 3), 0.0) * 100,
        "day5_pct": trade_plan.get("day5_predicted_return", trade_plan.get("avg_predicted_return", 0.0)) * 100,
        "day5_confidence_pct": trade_plan.get("day5_confidence", trade_plan.get("avg_confidence", 0.0)) * 100,
        "support_strength_pct": support_strength * 100 if isinstance(support_strength, (int, float)) else None,
        "resistance_strength_pct": resistance_strength * 100 if isinstance(resistance_strength, (int, float)) else None,
    }
    chart_image = _build_detail_chart_base64(payload)
    atr_value = payload.get("atr")
    if atr_value is None:
        atr_value = payload.get("trade_plan", {}).get("atr")
    return render_template(
        "stock_detail.html",
        payload=payload,
        forecast=forecast,
        live=live,
        live_text=_build_live_signal_text(live),
        atr_value=atr_value,
        detail_summary=detail_summary,
        chart_image=chart_image,
    )


@app.post("/scan")
def run_scan():
    if not _state["scan_running"]:
        threading.Thread(target=_scan_job, daemon=True).start()
    else:
        flash("Scan already in progress.", "error")
    return redirect(request.referrer or url_for("dashboard"))


@app.route("/status")
def status():
    return jsonify(_state)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
