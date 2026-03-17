import threading
from datetime import datetime

from flask import Flask, flash, jsonify, redirect, render_template, request, url_for

import main
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
        macd_rows=load_macd_rows(),
        rsi_rows=load_rsi_rows(),
        forecast_rows=selected["forecasts"] if selected else load_market_forecast_rows(),
        forecast_history=forecast_history,
        selected_date=selected_date,
        scan_state=_state,
        format_mcap=format_mcap,
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
    atr_value = payload.get("atr")
    if atr_value is None:
        atr_value = payload.get("trade_plan", {}).get("atr")
    return render_template("stock_detail.html", payload=payload, live=live, atr_value=atr_value)


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
