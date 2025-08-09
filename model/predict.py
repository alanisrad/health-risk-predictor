import os
import joblib
import pandas as pd

# ---- SHAP availability + logging ----
try:
    import shap
    _HAS_SHAP = True
except Exception as e:
    _HAS_SHAP = False
    print("DEBUG shap import failed:", e, flush=True)

print("DEBUG _HAS_SHAP ->", _HAS_SHAP, flush=True)

# ---- Load model ----
model_path = os.path.join(os.path.dirname(__file__), "illness_risk_model.joblib")
model = joblib.load(model_path)

# ---- Feature & label definitions ----
FEATURE_ORDER = [
    "age", "vitamin_d", "a1c", "ferritin", "glucose",
    "creatinine", "ldl", "vitamin_b12", "tsh"
]

PREDICTION_LABELS = [
    "vitamin_d_deficiency",
    "anemia_risk",
    "prediabetes_risk",
    "thyroid_flag",
    "high_cholesterol_risk"
]

# NEW: neutral defaults to fill missing values (best-effort mode)
DEFAULTS = {
    "age": 40, "vitamin_d": 30, "a1c": 5.4, "ferritin": 80, "glucose": 95,
    "creatinine": 0.9, "ldl": 110, "vitamin_b12": 500, "tsh": 2.0
}

def risk_label(score: float) -> str:
    if score >= 0.7:
        return "High Risk"
    elif score >= 0.4:
        return "Moderate"
    else:
        return "Low"

def _positive_class_index(est):
    classes = list(est.classes_)
    if True in classes:  return classes.index(True)
    if 1 in classes:     return classes.index(1)
    return 0

def _confidence_from_prob(p: float) -> float:
    # Distance from 0.5 → 0..100%
    return round(abs(p - 0.5) * 200.0, 1)

# ---------- RULE OVERLAY (primary drivers by thresholds) ----------
def _rule_factors(values: dict, label: str, p: float):
    """
    Returns a small list of directional strings (↑/↓ feature with value context)
    prioritizing the *rule that defines the condition*. This makes explanations
    intuitive even when SHAP is quirky.
    """
    v = values
    show_up = (p >= 0.7)
    show_down = (p < 0.4)
    out = []

    def add(txt):
        if txt not in out:
            out.append(txt)

    if label == "vitamin_d_deficiency":
        # Risk rule: vitamin_d < 20
        if show_up and v["vitamin_d"] < 20:
            add(f"↑ vitamin_d (low {v['vitamin_d']})")
        elif show_down and v["vitamin_d"] >= 30:
            add(f"↓ vitamin_d (optimal {v['vitamin_d']})")
        elif not show_up and not show_down:
            if v["vitamin_d"] < 20:
                add(f"↑ vitamin_d (low {v['vitamin_d']})")
            else:
                add(f"↓ vitamin_d ({v['vitamin_d']})")

    elif label == "anemia_risk":
        # Risk rule: ferritin < 25
        if show_up and v["ferritin"] < 25:
            add(f"↑ ferritin (low {v['ferritin']})")
        elif show_down and v["ferritin"] >= 50:
            add(f"↓ ferritin ({v['ferritin']})")
        elif not show_up and not show_down:
            if v["ferritin"] < 25:
                add(f"↑ ferritin (low {v['ferritin']})")
            else:
                add(f"↓ ferritin ({v['ferritin']})")

    elif label == "prediabetes_risk":
        # Risk rule: 5.7 < a1c < 6.5
        if show_up and (5.7 < v["a1c"] < 6.5):
            add(f"↑ a1c ({v['a1c']})")
        elif show_down and v["a1c"] <= 5.6:
            add(f"↓ a1c ({v['a1c']})")
        elif not show_up and not show_down:
            if 5.7 < v["a1c"] < 6.5:
                add(f"↑ a1c ({v['a1c']})")
            else:
                add(f"↓ a1c ({v['a1c']})")

    elif label == "thyroid_flag":
        # Risk rule: tsh < 0.3 or > 4.5
        if show_up and (v["tsh"] < 0.3 or v["tsh"] > 4.5):
            add(f"↑ tsh ({v['tsh']})")
        elif show_down and (0.4 <= v["tsh"] <= 4.5):
            add(f"↓ tsh ({v['tsh']})")
        elif not show_up and not show_down:
            if v["tsh"] < 0.3 or v["tsh"] > 4.5:
                add(f"↑ tsh ({v['tsh']})")
            else:
                add(f"↓ tsh ({v['tsh']})")

    elif label == "high_cholesterol_risk":
        # Risk rule: ldl > 130
        if show_up and v["ldl"] > 130:
            add(f"↑ ldl ({v['ldl']})")
        elif show_down and v["ldl"] <= 100:
            add(f"↓ ldl ({v['ldl']})")
        elif not show_up and not show_down:
            if v["ldl"] > 130:
                add(f"↑ ldl ({v['ldl']})")
            else:
                add(f"↓ ldl ({v['ldl']})")

    return out

# ---------- SHAP (directional) ----------
def _shap_row_for_estimator(est, input_df):
    """
    Return (row, features) where `row` is 1D SHAP values (per-feature) for the positive class.
    Return (None, None) if only class scores were produced (no per-feature attributions).
    """
    try:
        # Fast path for tree models; use raw (log-odds) for stable directions
        try:
            explainer = shap.TreeExplainer(est, model_output="raw")
            sv = explainer.shap_values(input_df, check_additivity=False)
        except Exception:
            exp = shap.Explainer(est, feature_names=FEATURE_ORDER)(input_df)
            sv = getattr(exp, "values", exp)

        # Normalize shapes
        if isinstance(sv, list):
            # binary: [neg, pos]
            pos_idx = _positive_class_index(est)
            arr = sv[pos_idx if pos_idx < len(sv) else -1]  # (n_samples, n_features)
            row = arr[0]
        else:
            arr = sv
            if getattr(arr, "ndim", 0) == 2:
                # (n_samples, n_features)  OR (n_samples, n_classes) -> reject latter
                if arr.shape[1] != len(FEATURE_ORDER):
                    return None, None
                row = arr[0]
            elif getattr(arr, "ndim", 0) == 3:
                # (n_samples, n_classes, n_features)
                pos_idx = _positive_class_index(est)
                if pos_idx >= arr.shape[1]:
                    pos_idx = arr.shape[1] - 1
                row = arr[0, pos_idx, :]
            else:
                return None, None

        if len(row) != len(FEATURE_ORDER):
            print(f"DEBUG SHAP feature length mismatch: {len(row)} vs {len(FEATURE_ORDER)}", flush=True)
            return None, None

        return row, FEATURE_ORDER
    except Exception as e:
        print("DEBUG SHAP error ->", e, flush=True)
        return None, None

def _top_factors_with_shap_directional(est, input_df, p: float, top_k=3):
    """
    Direction-aware top factors:
      High (p>=0.7): only ↑ (risk-increasing)
      Low  (p<0.4) : only ↓ (risk-decreasing)
      Moderate     : strongest |impact| (mix)
    """
    row, feats = _shap_row_for_estimator(est, input_df)
    if row is None:
        return None

    pairs = [(feats[i], float(row[i])) for i in range(len(row))]

    if p >= 0.7:
        filtered = [(f, v) for f, v in pairs if v > 0]   # risk-up
    elif p < 0.4:
        filtered = [(f, v) for f, v in pairs if v < 0]   # risk-down
    else:
        filtered = pairs                                  # mix

    if not filtered:  # edge case
        filtered = pairs

    filtered.sort(key=lambda x: abs(x[1]), reverse=True)
    tops = []
    for f, v in filtered[:top_k]:
        direction = "↑" if v > 0 else "↓"
        tops.append(f"{direction} {f}")
    return tops

# ---------- Perturbation fallback (directional) ----------
def _top_factors_perturbation(est, input_df, p: float, top_k=3, rel_step=0.05):
    """
    Model-agnostic local explanation by finite differences.
    Honors the same directional filtering based on p.
    """
    import numpy as np

    pos_idx = _positive_class_index(est)
    base_p = float(est.predict_proba(input_df)[0][pos_idx])

    x0 = input_df.iloc[0].copy()
    impacts = []
    for feat in FEATURE_ORDER:
        val = float(x0[feat])
        step = abs(val) * rel_step if val != 0 else rel_step

        x_up = x0.copy(); x_up[feat] = val + step
        x_dn = x0.copy(); x_dn[feat] = max(0.0, val - step)

        p_up = float(est.predict_proba(pd.DataFrame([x_up], columns=FEATURE_ORDER))[0][pos_idx])
        p_dn = float(est.predict_proba(pd.DataFrame([x_dn], columns=FEATURE_ORDER))[0][pos_idx])

        # symmetric local effect
        delta = ((p_up - base_p) + (base_p - p_dn)) / 2.0
        impacts.append((feat, delta))

    # Directional filtering
    if p >= 0.7:
        filtered = [(f, d) for f, d in impacts if d > 0]
    elif p < 0.4:
        filtered = [(f, d) for f, d in impacts if d < 0]
    else:
        filtered = impacts

    if not filtered:
        filtered = impacts

    filtered.sort(key=lambda t: abs(t[1]), reverse=True)
    return [("↑ " if d > 0 else "↓ ") + f for f, d in filtered[:top_k]]

def _top_factors_fallback(est, top_k=3):
    """Last resort: global importances (no direction)."""
    try:
        importances = getattr(est, "feature_importances_", None)
        if importances is None:
            return None
        pairs = list(zip(FEATURE_ORDER, importances))
        pairs.sort(key=lambda x: x[1], reverse=True)
        return [f"• {f}" for f, _ in pairs[:top_k]]
    except Exception:
        return None

# ---------- Main predict ----------
def predict_illness_scores(input_dict: dict) -> dict:
    # NEW: best-effort imputation so model/explainers never see None/NaN
    clean = {k: (input_dict.get(k) if input_dict.get(k) is not None else DEFAULTS[k])
             for k in FEATURE_ORDER}
    input_df = pd.DataFrame([clean], columns=FEATURE_ORDER).astype(float)

    results = {}
    for label, est in zip(PREDICTION_LABELS, model.estimators_):
        # predict prob of positive class
        proba_row = est.predict_proba(input_df)[0]
        pos_idx = _positive_class_index(est)
        p = float(proba_row[pos_idx])

        conf = _confidence_from_prob(p)

        # RULE overlay first (use the same clean values for consistency)
        rule_list = _rule_factors(clean, label, p)

        # ML explanations (directional)
        top_factors = None
        if _HAS_SHAP:
            top_factors = _top_factors_with_shap_directional(est, input_df, p, top_k=3)
        if not top_factors:
            top_factors = _top_factors_perturbation(est, input_df, p, top_k=3)
        if not top_factors:
            top_factors = _top_factors_fallback(est, top_k=3)

        # Merge rule + ML, dedupe by feature name (keep arrows)
        merged = list(rule_list)
        seen_feats = {f.split()[-1] for f in merged}  # crude: last token is feature name
        for item in top_factors or []:
            feat = item.split()[-1]
            if feat not in seen_feats:
                merged.append(item)
                seen_feats.add(feat)

        results[label] = {
            "score": round(p * 100.0, 1),   # 0–100%
            "label": risk_label(p),         # Low / Moderate / High Risk
            "confidence": conf,             # 0–100%
            "top_factors": merged[:3]       # keep it short
        }

        # Debug per condition
        print(
            f"DEBUG {label} -> score={results[label]['score']}%, "
            f"conf={conf}%, factors={results[label]['top_factors']}",
            flush=True
        )

    return results

