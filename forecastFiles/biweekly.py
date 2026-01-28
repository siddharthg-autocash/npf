# forecastFiles/biweekly.py
import pandas as pd
import numpy as np
from datetime import timedelta
from forecastFiles.core import WEEKDAY_MAP, apply_adjustment

def _extract_biweekly_anchor_and_noise(df, pct_threshold=0.15, anchor_percentile=90):
    """
    Robust anchor detection:
      1) Try percentile-based anchor selection (top anchor_percentile% by abs amount).
      2) If that yields nothing, fall back to original pct_threshold * median rule.
    Returns (last_anchor_datetime, noise_sum_after_anchor)
    """
    df_sorted = df.sort_values("date").copy()
    abs_amounts = df_sorted["amount"].abs()

    if df_sorted.empty:
        raise ValueError("empty df passed to anchor detection")

    # Attempt percentile-based detection first (preferred)
    try:
        perc = float(anchor_percentile)
        if 0 < perc < 100:
            thr = np.percentile(abs_amounts.values, perc)
            anchor_rows = df_sorted[abs_amounts >= thr]
            if not anchor_rows.empty:
                last_anchor = anchor_rows["date"].iloc[-1]
                noise_sum = df_sorted[df_sorted["date"] > last_anchor]["amount"].sum()
                return last_anchor, float(noise_sum)
    except Exception:
        # percentile calc failed for some reason -> fall back
        pass

    # Fallback: original median * pct_threshold approach (keeps backward compatibility)
    median_amt = abs_amounts.median()
    if median_amt == 0 or np.isnan(median_amt):
        return df_sorted["date"].iloc[-1], 0.0

    threshold = pct_threshold * median_amt
    is_anchor = abs_amounts >= threshold
    anchor_rows = df_sorted[is_anchor]

    if anchor_rows.empty:
        return df_sorted["date"].iloc[-1], 0.0

    last_anchor = anchor_rows["date"].iloc[-1]
    noise_sum = df_sorted[df_sorted["date"] > last_anchor]["amount"].sum()
    return last_anchor, float(noise_sum)


def _generate_raw_lattice(last_anchor, start, end, weekday_override=None, back_steps=1, forward_padding_days=14):
    """
    Generate raw biweekly lattice candidates.
    - start from last_anchor + 14 days, then extend forward in 14-day steps up to end + padding.
    - also optionally include 1 back-step (previous biweek) which can be important when adjustment moves dates forward.
    """
    dates = []

    if weekday_override:
        target = WEEKDAY_MAP.get(weekday_override.lower(), last_anchor.weekday())
    else:
        target = last_anchor.weekday()

    # primary start
    primary = pd.to_datetime(last_anchor) + timedelta(days=14)

    # optionally include earlier step(s) to allow adjustments to bring them into window
    first = primary - timedelta(days=14 * back_steps)

    current = first
    limit = pd.to_datetime(end) + timedelta(days=forward_padding_days)

    while current <= limit:
        # align this lattice point to desired weekday with minimal forward shift (0..6)
        delta = (target - current.weekday()) % 7
        aligned = current + timedelta(days=delta)
        dates.append(pd.to_datetime(aligned))
        current += timedelta(days=14)

    # remove duplicates and sort
    dates = sorted(pd.to_datetime(list(dict.fromkeys(dates))))
    return dates


def _aggregate_collapsed_dates(dates, vals):
    df = pd.DataFrame({"date": pd.to_datetime(dates), "val": vals})
    if df.empty:
        return [], []
    agg = df.groupby("date", as_index=True)["val"].sum().sort_index()
    return list(agg.index), list(agg.values)


def run_biweekly_forecast(df, model_params, forecast_start, forecast_end, holiday_set, raw_df=None):
    """
    Robust biweekly with debug output.
    Returns the usual payload plus debug fields:
      - debug: {
            "anchors_detected": [...],
            "raw_lattice": [...],
            "candidates_before_adjust": [...],
            "candidates_after_adjust": [...],
            "excluded_reasons": [ ... ],
            "base_amount_from_anchors": x or null
        }
    """
    forecast_start = pd.to_datetime(forecast_start)
    forecast_end = pd.to_datetime(forecast_end)

    anchor_source = raw_df if raw_df is not None else df

    # 1) anchor detection (raw)
    last_anchor, noise = _extract_biweekly_anchor_and_noise(anchor_source, pct_threshold=0.15)

    # Also produce full list of detected anchors (for debug)
    abs_amounts = anchor_source["amount"].abs()
    median_amt = abs_amounts.median() if not anchor_source.empty else 0.0
    anchors_only = pd.DataFrame()
    if median_amt > 0 and not np.isnan(median_amt):
        thr = 0.15 * median_amt
        anchors_only = anchor_source[abs_amounts >= thr].sort_values("date")

    # 2) Build a robust raw lattice. include one back-step to account for adjustments that push into window.
    raw_lattice = _generate_raw_lattice(last_anchor, forecast_start, forecast_end, model_params.get("weekday"), back_steps=1)

    debug = {
        "anchors_detected": anchors_only["date"].dt.strftime("%Y-%m-%d").tolist() if not anchors_only.empty else [],
        "median_abs_amount": float(median_amt) if not np.isnan(median_amt) else None,
        "last_anchor": pd.to_datetime(last_anchor).strftime("%Y-%m-%d"),
        "noise_sum": float(noise),
        "raw_lattice": [d.strftime("%Y-%m-%d") for d in raw_lattice],
        "candidates_before_adjust": [],
        "candidates_after_adjust": [],
        "excluded_reasons": []
    }

    if not raw_lattice:
        return {"status": "no_future_dates", "debug": debug}

    # 3) compute base_amount preferably from anchors_only, else fallback to filtered df
    base_amount = None
    base_source = None
    if not anchors_only.empty:
        n_candidates = model_params.get("n_candidates", [3])
        if isinstance(n_candidates, int):
            n_candidates = [n_candidates]
        try:
            k = int(min(n_candidates))
        except Exception:
            k = 3
        k = max(1, min(k, len(anchors_only)))
        base_amount = float(anchors_only.sort_values("date")["amount"].iloc[-k:].mean())
        base_source = "anchors"
    else:
        # fallback to filtered transactions (df)
        ts = df.set_index("date")["amount"].sort_index()
        s = ts.dropna()
        n_candidates = model_params.get("n_candidates", [3])
        if isinstance(n_candidates, int):
            n_candidates = [n_candidates]
        try:
            n = int(min(n_candidates))
        except Exception:
            n = 3
        if len(s) == 0:
            base_amount = 0.0
        else:
            n_used = max(1, min(n, len(s)))
            base_amount = float(s.iloc[-n_used:].mean())
        base_source = "filtered_transactions"

    debug["base_amount_from_anchors"] = None if base_source != "anchors" else base_amount
    debug["base_amount_used"] = float(base_amount)
    debug["base_source"] = base_source

    # 4) For each raw lattice candidate, compute value and apply adjustment BEFORE filtering by forecast window.
    adj_option = model_params.get("adjustmentOption")
    candidates_before = []
    candidates_after = []

    for idx, cand in enumerate(raw_lattice):
        val = base_amount + noise if idx == 0 else base_amount
        candidates_before.append({"raw_candidate": pd.to_datetime(cand).strftime("%Y-%m-%d"), "value": float(val)})

        # apply adjustment (returns DatetimeIndex and values list)
        final_idx, final_vals = apply_adjustment([cand], [val], adj_option, holiday_set)
        if len(final_idx) == 0:
            # was dropped by adjustment (dropTransaction)
            debug["excluded_reasons"].append({
                "raw_candidate": pd.to_datetime(cand).strftime("%Y-%m-%d"),
                "reason": "dropped_by_adjustment"
            })
            continue

        adj_dt = pd.to_datetime(final_idx[0])
        adj_val = float(final_vals[0])
        candidates_after.append({"adjusted_candidate": adj_dt.strftime("%Y-%m-%d"), "value": adj_val, "raw_candidate": pd.to_datetime(cand).strftime("%Y-%m-%d")})

    debug["candidates_before_adjust"] = candidates_before
    debug["candidates_after_adjust"] = candidates_after

    # 5) Now filter by forecast window using the ADJUSTED date
    adjusted_dates = []
    adjusted_vals = []
    for item in candidates_after:
        adj_dt = pd.to_datetime(item["adjusted_candidate"])
        val = item["value"]
        if adj_dt >= forecast_start and adj_dt <= forecast_end:
            adjusted_dates.append(adj_dt)
            adjusted_vals.append(val)
        else:
            debug["excluded_reasons"].append({
                "adjusted_candidate": item["adjusted_candidate"],
                "raw_candidate": item["raw_candidate"],
                "reason": "outside_forecast_window_after_adjust"
            })

    if not adjusted_dates:
        # nothing inside forecast window
        return {"status": "no_future_dates", "debug": debug}

    # 6) Deduplicate collisions (sum values)
    final_dates, final_vals = _aggregate_collapsed_dates(adjusted_dates, adjusted_vals)

    # 7) Prepare output (same shape as others) with debug
    out = {
        "status": "success",
        "cadence_anchor": pd.to_datetime(last_anchor).strftime("%Y-%m-%d"),
        "noise_offset_applied": float(noise),
        "index": [d.strftime("%Y-%m-%d") for d in final_dates],
        "forecast": final_vals,
        "debug": debug
    }
    return out
