"""
Football Valuation Engine (Streamlit)

Mode A  Player Explorer: valuation vs Transfermarkt, season trend, price drivers in %.
Mode B  What-If Simulator: start from a real player or a typical one, change a few
        headline inputs, and everything derivable is filled in automatically
        (90s, per-90 rates, G+A, xG family, league from club, count stats scaled to minutes...).
"""
from __future__ import annotations

import re
from collections import Counter
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
import streamlit as st

st.set_page_config(page_title="Football Valuation Engine", page_icon="⚽", layout="wide")

# ==============================================================================
# 0. CONFIG
# ==============================================================================
MODEL_DIR = Path("models")
ASSETS = {
    "Outfield": dict(
        model="xgboost_valuation_model.joblib", prep="valuation_preprocessor.joblib",
        feats="feature_order.joblib",
        data=["data/final_model_data.csv", "data/cleaned_model_data.csv", "cleaned_model_data.csv"]),
    "Goalkeeper": dict(
        model="gk_xgboost_model.joblib", prep="gk_preprocessor.joblib",
        feats="gk_feature_order.joblib",
        data=["data/final_gk_data.csv", "final_gk_data.csv", "data/cleaned_model_data.csv"]),
}
MODE_A, MODE_B = "🔍 Player Explorer", "🎛️ What-If Simulator"

ROLE_ALIASES = {  # semantic role -> accepted column names (lower case)
    "name": ["name", "player", "player_name", "matched_fbref_name"],
    "team": ["team", "club", "squad"],
    "league": ["league", "competition", "comp"],
    "foot": ["foot"],
    "position": ["position", "pos"],
}
KNOWN_CATS = {"team", "league", "foot", "agent_name", "position", "club", "squad", "comp", "competition"}
ROLE_SUFFIX = {"team": "team", "league": "league", "foot": "foot", "position": "pos"}

STAT_LABELS = {
    "CrdY": "Yellow cards", "CrdR": "Red cards", "G+A": "Goals + assists", "MP": "Matches played",
    "Min": "Minutes", "90s": "90s played", "Gls": "Goals", "Ast": "Assists", "G-PK": "Non-penalty goals",
    "PK": "Penalty goals", "PKatt": "Penalty attempts", "Sh": "Shots", "SoT": "Shots on target",
    "xG": "xG", "npxG": "Non-penalty xG", "xAG": "xAG", "xA": "xA", "npxG+xAG": "npxG + xAG",
    "G+A-PK": "Non-penalty G + A", "CS": "Clean sheets", "CS%": "Clean sheet %", "Starts": "Starts",
    "Save%": "Save %", "PSxG+/-": "PSxG +/-", "PSxG": "PSxG", "GA": "Goals against", "GA90": "Goals against / 90",
    "SoTA": "Shots on target against", "Tkl": "Tackles", "Int": "Interceptions", "Tkl+Int": "Tackles + interceptions",
    "Fls": "Fouls", "Fld": "Fouled", "Off": "Offsides", "Crs": "Crosses", "PrgC": "Progressive carries",
    "PrgP": "Progressive passes", "PrgR": "Progressive passes received", "age_at_val": "Age",
    "contract_days_left": "Contract days left", "international_caps": "International caps",
}
CORE_GROUPS = {"performance", "standard", "playing time", "expected", "per 90 minutes"}

# Headline widgets: state name -> candidate FBref stat names
HEADLINES = {
    "Outfield": {"goals": ["Gls"], "assists": ["Ast"], "xg": ["xG"], "xag": ["xAG", "xA"]},
    "Goalkeeper": {"cs": ["CS"], "save": ["Save%"], "psxg": ["PSxG+/-"]},
}
# name -> (min, max, is_int)
LIMITS = {"goals": (0, 100, True), "assists": (0, 60, True), "cs": (0, 38, True),
          "xg": (0.0, 80.0, False), "xag": (0.0, 60.0, False),
          "save": (0.0, 100.0, False), "psxg": (-20.0, 25.0, False)}
FIT_SPECS = {  # name -> (x stat, [y stats], x is per-90)
    "xg": ("Gls", ["xG"], True), "xag": ("Ast", ["xAG", "xA"], True), "psxg": ("Save%", ["PSxG+/-"], False)}


# ==============================================================================
# 1. NAME HELPERS
# ==============================================================================
def raw_name(col: str) -> str:
    for p in ("num__", "cat__", "remainder__"):
        if col.startswith(p):
            return col[len(p):]
    return col


def split_group(raw: str):
    """'Per 90 Minutes_Gls' -> ('Per 90 Minutes', 'Gls'); snake_case names have no group."""
    if "_" in raw and raw[:1].isupper():
        g, s = raw.split("_", 1)
        return g, s
    return None, raw


def is_per90_group(g) -> bool:
    return bool(g) and ("per 90" in g.lower() or "per90" in g.lower())


def stat_key(stat: str) -> str:
    s = stat.lower().replace("+", "plus").replace("-", "minus").replace("%", "pct").replace("/", "per")
    return re.sub(r"[^a-z0-9]", "", s)


K = stat_key


def find_key(columns, key):
    """First non-per-90 column whose stat name matches `key` exactly (no substring guessing)."""
    for c in columns:
        g, s = split_group(raw_name(c))
        if stat_key(s) == key and not is_per90_group(g):
            return c
    return None


def clean_ui_name(col: str) -> str:
    raw = raw_name(col)
    if col.startswith("cat__"):
        t = raw.replace("_", " ").strip()
        return t[:1].upper() + t[1:]
    grp, stat = split_group(raw)
    label = STAT_LABELS.get(stat) or (stat.replace("_", " ").strip().title() if grp is None else stat)
    if is_per90_group(grp):
        return label + " / 90"
    if grp and grp.lower() not in CORE_GROUPS:
        label += f" ({grp})"
    return label


def fmt_eur(x) -> str:
    if x is None or not np.isfinite(x):
        return "N/A"
    a = abs(x)
    if a >= 1e6:
        return f"€{x / 1e6:,.1f}M"
    if a >= 1e3:
        return f"€{x / 1e3:,.0f}K"
    return f"€{x:,.0f}"


def num(x, default=np.nan):
    v = pd.to_numeric(x, errors="coerce")
    return default if pd.isna(v) else float(v)


def with_current(opts, cur):
    return opts if cur is None or cur in opts else [cur] + list(opts)


def to_num(s):
    return pd.to_numeric(s, errors="coerce")


# ==============================================================================
# 2. CACHED LOADERS
# ==============================================================================
@st.cache_resource(show_spinner="Loading model…")
def load_assets(ptype: str):
    a = ASSETS[ptype]
    model = joblib.load(MODEL_DIR / a["model"])
    prep = joblib.load(MODEL_DIR / a["prep"])
    feats = list(joblib.load(MODEL_DIR / a["feats"]))
    return model, prep, feats, shap.Explainer(model)


def _clean_season(s):
    s = str(s).replace(".0", "")
    if len(s) == 4 and s.isdigit():
        return f"{int(s) - 1}/{s[2:]}" if s.startswith("20") else f"20{s[:2]}/{s[2:]}"
    return s


@st.cache_data(show_spinner="Loading player data…")
def load_player_database(ptype: str) -> pd.DataFrame:
    df = None
    for path in ASSETS[ptype]["data"]:
        if Path(path).exists():
            df = pd.read_csv(path)
            break
    if df is None:
        st.error(f"No dataset found for {ptype}. Looked for: {', '.join(ASSETS[ptype]['data'])}")
        st.stop()

    low = {c.lower(): c for c in df.columns}
    pick = lambda names: next((low[n] for n in names if n in low), None)  # noqa: E731
    date_c = pick(["date", "date_val", "valuation_date"])
    dob_c = pick(["date_of_birth", "dob"])
    exp_c = pick(["contract_expiration_date", "expiration_date"])

    if "age_at_val" not in df.columns:
        if dob_c and date_c:
            df["age_at_val"] = (pd.to_datetime(df[date_c], errors="coerce")
                                - pd.to_datetime(df[dob_c], errors="coerce")).dt.days / 365.25
        else:
            df["age_at_val"] = np.nan
    df["age_at_val"] = df["age_at_val"].fillna(df["age_at_val"].median()).fillna(26.0)

    if "contract_days_left" not in df.columns:
        if exp_c and date_c:
            df["contract_days_left"] = (pd.to_datetime(df[exp_c], errors="coerce")
                                        - pd.to_datetime(df[date_c], errors="coerce")).dt.days
        else:
            df["contract_days_left"] = 0
    df["contract_days_left"] = df["contract_days_left"].fillna(0)

    target = next((c for c in df.columns if "market_value_in_eur" in c.lower()), None)
    if target:
        df["actual_market_value"] = df[target]
    df["season_display"] = df["season"].apply(_clean_season) if "season" in df.columns else "Unknown"
    return df.reset_index(drop=True)


@st.cache_data(show_spinner="Analysing model features…")
def build_plan(_df: pd.DataFrame, features: tuple, ptype: str, n_rows: int) -> dict:
    """Works out, once per dataset, how every model feature behaves so the simulator
    can fill it in automatically (count vs per-90 vs ratio vs category vs admin)."""
    df, cols = _df, list(_df.columns)

    def src(c):
        if c in df.columns:
            return c
        r = raw_name(c)
        return r if r in df.columns else None

    roles = {r: next((c for c in cols if c.lower() in al), None) for r, al in ROLE_ALIASES.items()}
    min_c, n90_c = find_key(cols, "min"), find_key(cols, "90s")
    if n90_c:
        n90 = to_num(df[n90_c])
    elif min_c:
        n90 = to_num(df[min_c]) / 90.0
    else:
        n90 = pd.Series(np.nan, index=df.index)
    valid = n90 >= 5
    if valid.sum() < 50:
        valid = n90 > 0
    if valid.sum() < 50:
        valid = pd.Series(True, index=df.index)
    n90_safe = n90.where(n90 > 0)

    kinds, key_of, special, srcmap = {}, {}, {}, {}
    for c in features:
        s = src(c)
        srcmap[c] = s
        raw = raw_name(c)
        g, stat = split_group(raw)
        key_of[c] = stat_key(stat)
        low = raw.lower()
        numeric = s is not None and pd.api.types.is_numeric_dtype(df[s])
        if low in KNOWN_CATS or c.startswith("cat__") or (s is not None and not numeric):
            kinds[c] = "cat"
        elif low == "age_at_val":
            kinds[c], special[c] = "static", "age"
        elif low == "contract_days_left":
            kinds[c], special[c] = "static", "contract"
        elif "caps" in low:
            kinds[c], special[c] = "static", "caps"
        elif "season" in low or "date" in low or low == "id" or low.endswith("_id"):
            kinds[c] = "admin"
        elif key_of[c] == "min" and not is_per90_group(g):
            kinds[c], special[c] = "time", "mins"
        elif key_of[c] == "90s":
            kinds[c], special[c] = "time", "n90"
        elif g is None or g.lower().startswith("team success"):
            kinds[c] = "static"
        elif is_per90_group(g):
            kinds[c] = "per90"
        elif "%" in stat or "/" in stat:
            kinds[c] = "static"
        else:  # a raw FBref total: it is a "count" if it grows with minutes played
            cr = to_num(df[s]).corr(n90) if numeric else np.nan
            kinds[c] = "count" if np.isfinite(cr) and cr > 0.25 else "static"

    # typical values: per-90 rate for counts, plain median for everything else
    R = {}
    for c in features:
        if kinds[c] in ("count", "per90", "static"):
            s = srcmap[c]
            if s is None:
                R[c] = pd.Series(0.0, index=df.index)
            else:
                v = to_num(df[s])
                R[c] = v / n90_safe if kinds[c] == "count" else v
    R = pd.DataFrame(R)
    Rv = R[valid]
    overall = Rv.median().fillna(0.0).to_dict() if len(R.columns) else {}
    by_pos, pc = {}, roles["position"]
    if pc and len(R.columns):
        ps = df.loc[valid, pc].astype(str)
        for p, idx in ps.groupby(ps).groups.items():
            if len(idx) >= 15:
                by_pos[p] = Rv.loc[idx].median().fillna(pd.Series(overall)).to_dict()

    # simple per-90 line fits used to auto-estimate xG / xAG / PSxG+/-
    def fit(mask, xs, ys, x_per90):
        xc = find_key(cols, stat_key(xs))
        yc = next((find_key(cols, stat_key(y)) for y in ys if find_key(cols, stat_key(y))), None)
        if xc is None or yc is None:
            return None
        x = to_num(df[xc]) / n90_safe if x_per90 else to_num(df[xc])
        y = to_num(df[yc]) / n90_safe
        m = mask & x.notna() & y.notna() & np.isfinite(x) & np.isfinite(y)
        if m.sum() < 30:
            return None
        slope, icpt = np.polyfit(x[m], y[m], 1)
        return float(slope), float(icpt)

    fits = {"__all__": {n: f for n, sp in FIT_SPECS.items() if (f := fit(valid, *sp))}}
    for p in by_pos:
        mask = valid & (df[pc].astype(str) == p)
        fits[p] = {n: f for n, sp in FIT_SPECS.items() if (f := fit(mask, *sp))}

    pk_c, g_c = find_key(cols, K("PK")), find_key(cols, K("Gls"))
    pk_share = 0.08
    if pk_c and g_c and to_num(df[g_c]).sum() > 0:
        pk_share = float(np.clip(to_num(df[pk_c]).sum() / to_num(df[g_c]).sum(), 0, 0.3))

    admin = {}
    for c in features:
        if kinds[c] == "admin":
            v = to_num(df[srcmap[c]]) if srcmap[c] else pd.Series(dtype=float)
            admin[c] = float(v.max()) if v.notna().any() else 0.0

    cat_default = {}
    for c in features:
        if kinds[c] == "cat":
            vc = df[srcmap[c]].dropna().astype(str).value_counts() if srcmap[c] else pd.Series(dtype=int)
            cat_default[c] = "Unknown" if "Unknown" in vc.index else (vc.index[0] if len(vc) else "Unknown")

    role_opts = {}
    for r in ("team", "league", "foot", "position"):
        if roles[r]:
            vc = df[roles[r]].dropna().astype(str).value_counts()
            role_opts[r] = sorted(vc.index) if r in ("team", "league") else vc.index.tolist()

    team_league = {}
    if roles["team"] and roles["league"]:
        tl = df[[roles["team"], roles["league"]]].dropna().astype(str)
        team_league = tl.groupby(roles["team"])[roles["league"]].agg(lambda s: s.mode().iat[0]).to_dict()

    default_team = role_opts.get("team", [None])[0]
    if roles["team"] and "actual_market_value" in df.columns:  # a typical club, not the richest
        tv = df.groupby(roles["team"])["actual_market_value"].mean().dropna().sort_values()
        if len(tv):
            default_team = str(tv.index[len(tv) // 2])

    role_col = {v: k for k, v in roles.items() if v and k != "name"}
    cat_role = {c: role_col.get(srcmap[c]) for c in features if kinds[c] == "cat"}

    col_by_key = {}
    for c in features:
        if kinds[c] in ("count", "static"):
            col_by_key.setdefault(key_of[c], c)
    totals = {}
    for c in features:
        if kinds[c] == "count":
            totals.setdefault(key_of[c], c)
    twins = {c: totals[key_of[c]] for c in features if kinds[c] == "per90" and key_of[c] in totals}

    hk = {"Outfield": {K(x) for x in ["Gls", "Ast", "G+A", "G-PK", "PK", "PKatt", "G+A-PK", "xG", "npxG",
                                       "xAG", "xA", "npxG+xAG"]},
          "Goalkeeper": {K(x) for x in ["Save%", "CS", "PSxG+/-", "GA", "GA90", "PSxG", "PSxG/SoT", "CS%", "Saves"]}}[ptype]
    editable = [c for c in features if kinds[c] in ("count", "static", "per90") and c not in special
                and key_of[c] not in hk and c not in twins]

    hl_src = {}
    for name, stats in HEADLINES[ptype].items():
        hl_src[name] = next((find_key(cols, K(s)) for s in stats if find_key(cols, K(s))), None)

    return dict(
        ptype=ptype, features=list(features), kinds=kinds, key_of=key_of, special=special, src=srcmap,
        roles=roles, role_opts=role_opts, team_league=team_league, default_team=default_team,
        overall=overall, by_pos=by_pos, fits=fits, pk_share=pk_share, admin=admin,
        cat_default=cat_default, cat_role=cat_role, col_by_key=col_by_key, twins=twins,
        editable=editable, hl_src=hl_src, min_col=min_c, n90_col=n90_c,
        generic_cats=[c for c in features if kinds[c] == "cat" and not cat_role.get(c)],
        cat_opts={c: df[srcmap[c]].dropna().astype(str).value_counts().index[:300].tolist()
                  for c in features if kinds[c] == "cat" and srcmap[c] and not cat_role.get(c)},
    )


# ==============================================================================
# 3. MODEL HELPERS
# ==============================================================================
def prepare_X(plan, rows: pd.DataFrame):
    X = rows.copy()
    missing = [c for c in plan["features"] if c not in X.columns]
    for c in missing:
        X[c] = "Unknown" if plan["kinds"][c] == "cat" else 0.0
    X = X[plan["features"]]
    for c in plan["features"]:
        if plan["kinds"][c] != "cat":
            X[c] = pd.to_numeric(X[c], errors="coerce")
    return X, missing


def encode(prep, X: pd.DataFrame) -> pd.DataFrame:
    arr = prep.transform(X)
    if hasattr(arr, "toarray"):
        arr = arr.toarray()
    return pd.DataFrame(arr, columns=prep.get_feature_names_out(), index=X.index)


def predict_eur(model, enc: pd.DataFrame) -> np.ndarray:
    return np.expm1(model.predict(enc))


def explain_row(explainer, enc_row: pd.DataFrame):
    ex = explainer(enc_row)
    raw = list(enc_row.columns)
    exp = shap.Explanation(values=ex.values[0], base_values=float(np.ravel(ex.base_values)[0]),
                           data=ex.data[0], feature_names=[clean_ui_name(c) for c in raw])
    return exp, raw


def driver_frame(exp, raw, k=10) -> pd.DataFrame:
    v = exp.values
    order = np.argsort(-np.abs(v))
    rows = []
    for i in order[:k]:
        lab = exp.feature_names[i]
        if raw[i].startswith("cat__"):
            lab = lab if exp.data[i] >= 0.5 else f"not {lab}"
        else:
            lab = f"{lab} = {exp.data[i]:,.4g}"
        rows.append((lab, v[i]))
    if len(order) > k:
        rows.append((f"{len(order) - k} other features", v[order[k:]].sum()))
    out = pd.DataFrame(rows, columns=["Driver", "log_effect"])
    out["pct"] = np.expm1(out["log_effect"]) * 100
    return out


def plot_drivers(fr: pd.DataFrame):
    fr = fr.iloc[::-1]
    fig, ax = plt.subplots(figsize=(8, 0.42 * len(fr) + 1.2))
    ax.barh(fr["Driver"], fr["pct"], color=np.where(fr["pct"] >= 0, "#2e8b57", "#c8453b"))
    for y, p in enumerate(fr["pct"]):
        ax.annotate(f"{p:+.0f}%", (p, y), xytext=(4 if p >= 0 else -4, 0), textcoords="offset points",
                    ha="left" if p >= 0 else "right", va="center", fontsize=9)
    ax.axvline(0, color="#888", lw=0.8)
    ax.margins(x=0.18)
    ax.set_xlabel("Effect on predicted value")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    fig.tight_layout()
    return fig


def show_drivers(model_exp, raw, base_eur, pred_eur, k=10):
    fr = driver_frame(model_exp, raw, k)
    st.caption(f"Starting from the model's average player ({fmt_eur(base_eur)}), each bar scales the value "
               f"up or down. Together they give {fmt_eur(pred_eur)}.")
    fig = plot_drivers(fr)
    st.pyplot(fig, clear_figure=True)
    plt.close(fig)
    with st.expander("Technical view (SHAP waterfall, log scale)"):
        shap.plots.waterfall(model_exp, max_display=12, show=False)
        fig2 = plt.gcf()
        st.pyplot(fig2)
        plt.close(fig2)


# ==============================================================================
# 4. SIMULATOR LOGIC (pure functions)
# ==============================================================================
def get_profile(plan, pos):
    return plan["by_pos"].get(pos, plan["overall"])


def row_n90(plan, row):
    if plan["n90_col"]:
        v = num(row.get(plan["n90_col"]))
    elif plan["min_col"]:
        v = num(row.get(plan["min_col"])) / 90.0
    else:
        v = np.nan
    return v if np.isfinite(v) else None


def profile_from_row(plan, row, fallback):
    """A real player's own stats as the baseline (counts become per-90 rates)."""
    n90 = row_n90(plan, row)
    prof = dict(fallback)
    for c in plan["features"]:
        k, s = plan["kinds"][c], plan["src"][c]
        if k in ("count", "per90", "static") and s is not None:
            v = num(row.get(s))
            if not np.isfinite(v):
                continue
            if k == "count":
                if n90 and n90 >= 3:
                    prof[c] = v / n90
            else:
                prof[c] = v
    return prof


def typical_headlines(plan, prof, n90):
    def tot(key, default):
        c = plan["col_by_key"].get(key)
        if c is None:
            return default
        v = prof.get(c, 0.0)
        return v * n90 if plan["kinds"][c] == "count" else v
    if plan["ptype"] == "Outfield":
        return {"goals": tot(K("Gls"), 3.0), "assists": tot(K("Ast"), 2.0)}
    return {"cs": tot(K("CS"), 8.0), "save": tot(K("Save%"), 70.0)}


def estimate(plan, pos, name, x, n90, fallback):
    f = plan["fits"].get(pos, {}).get(name) or plan["fits"].get("__all__", {}).get(name)
    if f is None:
        return fallback
    slope, icpt = f
    if name in ("xg", "xag"):
        return max((slope * x / n90 + icpt) * n90, 0.0)
    return (slope * x + icpt) * n90


def resolve_x(plan, inp, n90):
    pos = inp["pos"]
    if plan["ptype"] == "Outfield":
        if inp["auto_x"]:
            return {"xg": estimate(plan, pos, "xg", inp["goals"], n90, 0.85 * inp["goals"]),
                    "xag": estimate(plan, pos, "xag", inp["assists"], n90, 0.85 * inp["assists"])}
        return {"xg": inp["xg"], "xag": inp["xag"]}
    if inp["auto_x"]:
        return {"psxg": estimate(plan, pos, "psxg", inp["save"], n90, 0.0)}
    return {"psxg": inp["psxg"]}


def headline_totals(plan, inp, n90, keyval):
    """Everything that follows arithmetically from the few headline inputs."""
    x = resolve_x(plan, inp, n90)
    if plan["ptype"] == "Outfield":
        g, a = inp["goals"], inp["assists"]
        pk = g * plan["pk_share"]
        pkatt = pk / 0.78
        npxg = max(x["xg"] - 0.76 * pkatt, 0.0)
        return {K("Gls"): g, K("Ast"): a, K("G+A"): g + a, K("G-PK"): g - pk, K("PK"): pk,
                K("PKatt"): pkatt, K("G+A-PK"): g + a - pk, K("xG"): x["xg"], K("npxG"): npxg,
                K("xAG"): x["xag"], K("xA"): x["xag"], K("npxG+xAG"): npxg + x["xag"]}
    sv, cs, ps = inp["save"], inp["cs"], x["psxg"]
    h = {K("Save%"): sv, K("CS"): cs, K("PSxG+/-"): ps}
    mp = keyval.get(K("MP"))
    if mp:
        h[K("CS%")] = min(100.0, cs / max(mp, 1.0) * 100)
    sota = keyval.get(K("SoTA"))
    if sota:
        ga = sota * (1 - sv / 100.0)
        h.update({K("GA"): ga, K("GA90"): ga / n90, K("Saves"): sota - ga,
                  K("PSxG"): ga + ps, K("PSxG/SoT"): (ga + ps) / sota})
    return h


def assemble(plan, inp):
    """User-facing inputs -> the full model feature row, plus how each value was obtained."""
    feats, kinds, key_of = plan["features"], plan["kinds"], plan["key_of"]
    n90 = max(inp["mins"] / 90.0, 0.25)
    prof = inp["profile"]
    out, how = {}, {}
    for c in feats:
        k = kinds[c]
        if k == "cat":
            out[c] = inp["cats"].get(c, plan["cat_default"][c])
            how[c] = "your choice" if c in inp["cats"] else "most common value"
        elif k == "admin":
            out[c], how[c] = plan["admin"][c], "latest season in data"
        elif k == "count":
            out[c], how[c] = prof.get(c, 0.0) * n90, "typical rate × minutes"
        else:
            out[c], how[c] = prof.get(c, 0.0), "typical value"
    sp = {"age": inp["age"], "contract": inp["contract"], "caps": inp["caps"],
          "mins": inp["mins"], "n90": inp["mins"] / 90.0}
    for c, s in plan["special"].items():
        out[c], how[c] = sp[s], "your input"
    for c, v in inp["extras"].items():
        if c in out:
            out[c], how[c] = float(v), "your override"
    keyval = {key_of[c]: out[c] for c in feats if kinds[c] in ("count", "static")}
    h = headline_totals(plan, inp, n90, keyval)
    for c in feats:
        if c in plan["special"] or kinds[c] not in ("count", "static", "per90") or key_of[c] not in h:
            continue
        out[c] = h[key_of[c]] / n90 if kinds[c] == "per90" else h[key_of[c]]
        how[c] = "from your headline stats"
    for pc, tc in plan["twins"].items():
        if key_of[pc] not in h and pc not in inp["extras"]:
            out[pc], how[pc] = out[tc] / n90, "total ÷ 90s"
    return out, how


# ==============================================================================
# 5. SIMULATOR STATE + CALLBACKS
# ==============================================================================
def sim_prefix(ptype):
    return "of_" if ptype == "Outfield" else "gk_"


def set_hl(p, name, v):
    lo, hi, is_int = LIMITS[name]
    st.session_state[p + name] = int(np.clip(round(v), lo, hi)) if is_int else float(np.clip(v, lo, hi))


def apply_typical(plan, p, only_untouched=True):
    ss = st.session_state
    n90 = max(ss[p + "mins"] / 90.0, 0.25)
    for name, v in typical_headlines(plan, get_profile(plan, ss.get(p + "pos")), n90).items():
        if not (only_untouched and name in ss[p + "touched"]):
            set_hl(p, name, v)


def sync_manual_x(plan, p):
    """Seed the manual xG / xAG / PSxG fields with the current auto estimate."""
    ss = st.session_state
    n90, pos = max(ss[p + "mins"] / 90.0, 0.25), ss.get(p + "pos")
    if plan["ptype"] == "Outfield":
        g, a = ss[p + "goals"], ss[p + "assists"]
        set_hl(p, "xg", round(estimate(plan, pos, "xg", g, n90, 0.85 * g), 1))
        set_hl(p, "xag", round(estimate(plan, pos, "xag", a, n90, 0.85 * a), 1))
    else:
        set_hl(p, "psxg", round(estimate(plan, pos, "psxg", ss[p + "save"], n90, 0.0), 1))


def ensure_state(plan, p):
    ss = st.session_state
    if ss.get(p + "ready") and (p + "age") in ss:
        return
    ss[p + "touched"], ss[p + "template"] = set(), None
    ss[p + "ver"] = ss.get(p + "ver", 0) + 1
    ss[p + "scenarios"] = ss.get(p + "scenarios", [])
    ss[p + "age"], ss[p + "contract"], ss[p + "caps"], ss[p + "mins"] = 24, 730, 10, 2000
    ro = plan["role_opts"]
    if "position" in ro:
        ss[p + "pos"] = ro["position"][0]
    if "team" in ro:
        ss[p + "team"] = plan["default_team"]
    if "league" in ro:
        ss[p + "league"] = plan["team_league"].get(plan["default_team"], ro["league"][0])
    if "foot" in ro:
        ss[p + "foot"] = "right" if "right" in ro["foot"] else ro["foot"][0]
    ss[p + "auto_x"] = True
    ss[p + "save"] = 72.0
    apply_typical(plan, p, only_untouched=False)
    sync_manual_x(plan, p)
    ss[p + "ready"] = True


def reset_sim(plan, p):
    st.session_state[p + "ready"] = False
    ensure_state(plan, p)


def touch(p, name):
    st.session_state[p + "touched"].add(name)


def on_context(plan, p):  # minutes or position changed -> refresh untouched headline defaults
    apply_typical(plan, p)


def on_team(plan, p):
    lg = plan["team_league"].get(st.session_state[p + "team"])
    if lg:
        st.session_state[p + "league"] = lg


def on_auto(plan, p):
    if not st.session_state[p + "auto_x"]:
        sync_manual_x(plan, p)


def load_template(plan, p, df, idx, label):
    ss = st.session_state
    ensure_state(plan, p)  # may be called from the explorer, before the simulator ever ran
    row = df.loc[idx]
    ss[p + "age"] = int(np.clip(round(num(row.get("age_at_val"), 24)), 16, 40))
    ss[p + "contract"] = int(np.clip(num(row.get("contract_days_left"), 730), 0, 2500))
    caps_col = next((c for c, s in plan["special"].items() if s == "caps"), None)
    if caps_col and plan["src"][caps_col]:
        ss[p + "caps"] = int(np.clip(num(row.get(plan["src"][caps_col]), 0), 0, 200))
    n90 = row_n90(plan, row)
    ss[p + "mins"] = int(np.clip(round((n90 * 90 if n90 else 2000) / 30) * 30, 90, 4500))
    for role, suffix in ROLE_SUFFIX.items():
        col = plan["roles"].get(role)
        if col and pd.notna(row.get(col)):
            ss[p + suffix] = str(row[col])
    for name, col in plan["hl_src"].items():
        v = num(row.get(col)) if col else np.nan
        if np.isfinite(v):
            set_hl(p, name, v)
    ss[p + "touched"] = set(plan["hl_src"])
    ss[p + "auto_x"] = False
    base = get_profile(plan, ss.get(p + "pos"))
    ss[p + "template"] = {"label": label, "profile": profile_from_row(plan, row, base),
                          "actual": num(row.get("actual_market_value"))}
    ss[p + "ver"] += 1


def go_simulate(plan, p, df, idx, label):
    load_template(plan, p, df, idx, label)
    st.session_state["mode"] = MODE_B


def collect_inputs(plan, p, extras, cat_extra):
    ss = st.session_state
    tpl = ss.get(p + "template")
    cats = {}
    for c, role in plan["cat_role"].items():
        if role and (p + ROLE_SUFFIX[role]) in ss:
            cats[c] = ss[p + ROLE_SUFFIX[role]]
    cats.update(cat_extra)
    pos = ss.get(p + "pos")
    inp = dict(age=ss[p + "age"], contract=ss[p + "contract"], caps=ss[p + "caps"], mins=ss[p + "mins"],
               pos=pos, cats=cats, extras=extras, auto_x=ss[p + "auto_x"],
               profile=tpl["profile"] if tpl else get_profile(plan, pos))
    for name in HEADLINES[plan["ptype"]]:
        inp[name] = ss[p + name]
    return inp


PERSISTED = ["age", "contract", "caps", "mins", "team", "league", "foot", "pos", "goals", "assists",
             "xg", "xag", "save", "cs", "psxg", "auto_x"]


def restore_state(p):
    """Streamlit drops widget state while a widget isn't on screen (e.g. in the explorer).
    Put back whatever we saved the last time the simulator was drawn."""
    for n, v in st.session_state.get(p + "saved", {}).items():
        st.session_state.setdefault(p + n, v)


SENS = {
    "Outfield": {"Age": ("age", list(range(17, 38))),
                 "Contract days left": ("contract", list(range(0, 2001, 100))),
                 "Minutes played": ("mins", list(range(300, 4501, 300))),
                 "Goals": ("goals", list(range(0, 31, 2)))},
    "Goalkeeper": {"Age": ("age", list(range(18, 40))),
                   "Contract days left": ("contract", list(range(0, 2001, 100))),
                   "Minutes played": ("mins", list(range(300, 4501, 300))),
                   "Save %": ("save", [55 + 2.5 * i for i in range(13)])},
}


# ==============================================================================
# 6. MODE A: PLAYER EXPLORER
# ==============================================================================
def render_explorer(plan, df, model, prep, explainer, ptype):
    p = sim_prefix(ptype)
    st.title("Player explorer")
    st.caption("Compare the model's estimate with Transfermarkt and see what drives the number.")
    name_col, roles = plan["roles"]["name"], plan["roles"]
    if not name_col:
        st.error("No player name column found in the dataset.")
        st.stop()

    sub = df
    if roles["league"] or roles["team"]:
        with st.expander("Filter the player list", expanded=False):
            f1, f2 = st.columns(2)
            if roles["league"]:
                lg = f1.selectbox("League", ["All"] + plan["role_opts"]["league"])
                if lg != "All":
                    sub = sub[sub[roles["league"]].astype(str) == lg]
            if roles["team"]:
                tm = f2.selectbox("Club", ["All"] + sorted(sub[roles["team"]].dropna().astype(str).unique()))
                if tm != "All":
                    sub = sub[sub[roles["team"]].astype(str) == tm]
    names = sorted(sub[name_col].dropna().unique())
    if not names:
        st.warning("No players match these filters.")
        st.stop()
    c1, c2 = st.columns([2, 1])
    selected = c1.selectbox("Player", names)

    recs = df[df[name_col] == selected]
    if "season" in recs.columns:
        recs = recs.sort_values("season", ascending=False)
    seasons = recs["season_display"].unique().tolist()
    season = c2.selectbox("Season", seasons) if len(seasons) > 1 else seasons[0]
    row = recs[recs["season_display"] == season].iloc[0:1]

    X, missing = prepare_X(plan, row)
    if missing:
        st.warning("Model features missing from the data (filled with defaults): "
                   + ", ".join(clean_ui_name(m) for m in missing[:12]) + ("…" if len(missing) > 12 else ""))
    enc = encode(prep, X)
    pred = float(predict_eur(model, enc)[0])
    actual = num(row["actual_market_value"].iloc[0]) if "actual_market_value" in row.columns else np.nan

    m = st.columns(4)
    m[0].metric("Age", f"{row['age_at_val'].iloc[0]:.1f}")
    team_col = roles["team"]
    m[1].metric("Club that season", str(row[team_col].iloc[0]) if team_col else "N/A")
    m[2].metric("Transfermarkt value", fmt_eur(actual))
    delta = (pred / actual - 1) * 100 if np.isfinite(actual) and actual > 0 else None
    m[3].metric("Model estimate", fmt_eur(pred), delta=f"{delta:+.0f}% vs Transfermarkt" if delta is not None else None,
                delta_color="off")
    if delta is not None:
        if abs(delta) < 15:
            st.info("The model agrees with the market within 15%.")
        elif delta > 0:
            st.info(f"The model values him {delta:.0f}% above Transfermarkt. Stats and profile suggest the market may be low.")
        else:
            st.info(f"The model values him {-delta:.0f}% below Transfermarkt. The market may be paying for something the stats don't show.")

    st.button("🎛️ Simulate this player", on_click=go_simulate,
              args=(plan, p, df, row.index[0], f"{selected} · {season}"),
              help="Opens the simulator with this player's real numbers pre-filled.")

    tab1, tab2 = st.tabs(["What drives the price", "Value over time"])
    with tab1:
        exp, raw = explain_row(explainer, enc)
        show_drivers(exp, raw, float(np.expm1(exp.base_values)), pred)
    with tab2:
        if len(recs) > 1:
            Xa, _ = prepare_X(plan, recs)
            hist = pd.DataFrame({"Season": recs["season_display"].values,
                                 "Model estimate": predict_eur(model, encode(prep, Xa)) / 1e6})
            if "actual_market_value" in recs.columns:
                hist["Transfermarkt"] = recs["actual_market_value"].values / 1e6
            hist = hist.groupby("Season").mean().sort_index()
            st.line_chart(hist, y_label="€ millions")
        else:
            st.caption("Only one season on record for this player.")


# ==============================================================================
# 7. MODE B: WHAT-IF SIMULATOR
# ==============================================================================
def render_simulator(plan, df, model, prep, explainer, ptype):
    ss, p, roles = st.session_state, sim_prefix(ptype), plan["roles"]
    restore_state(p)
    ensure_state(plan, p)
    st.title("What-if simulator")
    st.caption("Set the few things that matter. Everything that follows from them (90s, per-90 rates, "
               "G+A, xG family, league, count stats) is filled in for you.")

    left, right = st.columns([1.05, 1], gap="large")
    extras, cat_extra = {}, {}

    # ---------------- left: inputs ----------------
    with left:
        with st.container(border=True):
            st.markdown("**Start from**")
            tpl = ss.get(p + "template")
            if tpl:
                st.success(f"{tpl['label']} (Transfermarkt: {fmt_eur(tpl['actual'])}). Their real stats are the baseline.")
                st.button("Back to a typical player", on_click=reset_sim, args=(plan, p))
            name_col = roles["name"]
            if name_col:
                a, b, c = st.columns([2, 1, 1])
                names = ["Typical player"] + sorted(df[name_col].dropna().unique())
                pick = a.selectbox("Player", names, key=p + "tpl_name", label_visibility="collapsed")
                if pick != "Typical player":
                    recs = df[df[name_col] == pick]
                    if "season" in recs.columns:
                        recs = recs.sort_values("season", ascending=False)
                    seasons = recs["season_display"].unique().tolist()
                    sea = b.selectbox("Season", seasons, key=p + "tpl_season", label_visibility="collapsed")
                    idx = recs[recs["season_display"] == sea].index[0]
                    c.button("Load", on_click=load_template, args=(plan, p, df, idx, f"{pick} · {sea}"),
                             use_container_width=True)

        with st.container(border=True):
            st.markdown("**Profile**")
            c1, c2 = st.columns(2)
            c1.slider("Age", 16, 40, key=p + "age")
            c2.number_input("Contract days left", 0, 2500, step=30, key=p + "contract")
            c2.caption(f"≈ {ss[p + 'contract'] / 365.25:.1f} years")
            if "team" in plan["role_opts"]:
                d1, d2 = st.columns(2)
                d1.selectbox("Club", with_current(plan["role_opts"]["team"], ss[p + "team"]),
                             key=p + "team", on_change=on_team, args=(plan, p))
                if "league" in plan["role_opts"]:
                    d2.selectbox("League", with_current(plan["role_opts"]["league"], ss[p + "league"]),
                                 key=p + "league")
                    d2.caption("Set from the club; change it if the club isn't in the data.")
            e1, e2, e3 = st.columns(3)
            if "position" in plan["role_opts"]:
                e1.selectbox("Position", with_current(plan["role_opts"]["position"], ss[p + "pos"]),
                             key=p + "pos", on_change=on_context, args=(plan, p))
            if "foot" in plan["role_opts"]:
                e2.selectbox("Foot", with_current(plan["role_opts"]["foot"], ss[p + "foot"]), key=p + "foot")
            e3.number_input("International caps", 0, 200, key=p + "caps")
            st.slider("Minutes played", 90, 4500, step=30, key=p + "mins", on_change=on_context, args=(plan, p))
            st.caption(f"= {ss[p + 'mins'] / 90:.1f} 90s. Match counts, per-90 rates and count stats follow this.")

        with st.container(border=True):
            st.markdown("**Output**")
            n90 = max(ss[p + "mins"] / 90.0, 0.25)
            if ptype == "Outfield":
                g1, g2 = st.columns(2)
                g1.number_input("Goals", 0, 100, step=1, key=p + "goals", on_change=touch, args=(p, "goals"))
                g2.number_input("Assists", 0, 60, step=1, key=p + "assists", on_change=touch, args=(p, "assists"))
                st.checkbox("Estimate xG and xAG from goals and assists", key=p + "auto_x",
                            on_change=on_auto, args=(plan, p))
                if ss[p + "auto_x"]:
                    est = resolve_x(plan, dict(pos=ss.get(p + "pos"), auto_x=True, goals=ss[p + "goals"],
                                               assists=ss[p + "assists"], ptype=ptype), n90)
                    st.caption(f"xG ≈ {est['xg']:.1f}, xAG ≈ {est['xag']:.1f} (typical conversion for this position). "
                               f"G+A = {ss[p + 'goals'] + ss[p + 'assists']}, "
                               f"{(ss[p + 'goals'] + ss[p + 'assists']) / n90:.2f} per 90.")
                else:
                    x1, x2 = st.columns(2)
                    x1.number_input("xG", 0.0, 80.0, step=0.1, key=p + "xg")
                    x2.number_input("xAG", 0.0, 60.0, step=0.1, key=p + "xag")
            else:
                st.slider("Save %", 0.0, 100.0, step=0.5, key=p + "save", on_change=touch, args=(p, "save"))
                g1, g2 = st.columns(2)
                g1.number_input("Clean sheets", 0, 38, step=1, key=p + "cs", on_change=touch, args=(p, "cs"))
                st.checkbox("Estimate PSxG +/- from save %", key=p + "auto_x", on_change=on_auto, args=(plan, p))
                if ss[p + "auto_x"]:
                    est = resolve_x(plan, dict(pos=ss.get(p + "pos"), auto_x=True, save=ss[p + "save"],
                                               ptype=ptype), n90)
                    st.caption(f"PSxG +/- ≈ {est['psxg']:+.1f}")
                else:
                    g2.number_input("PSxG +/-", -20.0, 25.0, step=0.1, key=p + "psxg")

        if plan["editable"] or plan["generic_cats"]:
            deep = st.toggle("Deep scouting: override individual stats", key=p + "deep")
            if deep:
                with st.container(border=True):
                    if plan["editable"]:
                        st.caption("Leave a cell blank to keep the auto-filled value. Per-90 columns follow their totals.")
                        ed = pd.DataFrame({"Stat": [clean_ui_name(c) for c in plan["editable"]],
                                           "Override": np.full(len(plan["editable"]), np.nan)},
                                          index=plan["editable"])
                        res = st.data_editor(ed, key=f"{p}ed_{ss[p + 'ver']}", disabled=["Stat"], hide_index=True,
                                             height=340, column_config={"Override": st.column_config.NumberColumn(
                                                 "Override", format="%.3f")})
                        extras = {c: float(v) for c, v in res["Override"].items() if pd.notna(v)}
                    for c in plan["generic_cats"]:
                        opts = plan["cat_opts"].get(c, [])
                        dflt = plan["cat_default"][c]
                        opts = with_current(opts, dflt)
                        cat_extra[c] = st.selectbox(clean_ui_name(c), opts, index=opts.index(dflt), key=f"{p}cat_{c}")

    # ---------------- right: result ----------------
    with right:
        inp = collect_inputs(plan, p, extras, cat_extra)
        try:
            feats, how = assemble(plan, inp)
            X, _ = prepare_X(plan, pd.DataFrame([feats]))
            enc = encode(prep, X)
            pred = float(predict_eur(model, enc)[0])
        except Exception as e:  # noqa: BLE001
            st.error(f"Prediction failed: {e}")
            st.stop()

        tpl = ss.get(p + "template")
        delta = None
        if tpl and np.isfinite(tpl["actual"]) and tpl["actual"] > 0:
            delta = f"{(pred / tpl['actual'] - 1) * 100:+.0f}% vs {tpl['label'].split(' · ')[0]} today"
        st.metric("Predicted market value", fmt_eur(pred), delta=delta, delta_color="off")

        s1, s2 = st.columns([2, 1])
        scn_name = s1.text_input("Scenario name", key=p + "scn_name", placeholder="e.g. Moves to a bigger club",
                                 label_visibility="collapsed")
        if s2.button("Save scenario", use_container_width=True):
            n = len(ss[p + "scenarios"]) + 1
            ss[p + "scenarios"].append({
                "Scenario": f"#{n} {scn_name}".strip(), "Value (€M)": round(pred / 1e6, 2), "Age": inp["age"],
                "Club": ss.get(p + "team", "-"), "Minutes": inp["mins"], "Contract days": inp["contract"],
                "Output": (f"{inp['goals']}G {inp['assists']}A" if ptype == "Outfield"
                           else f"{inp['save']:.1f}% saves, {inp['cs']} CS")})

        t1, t2, t3, t4 = st.tabs(["Price drivers", "Sensitivity", "Scenarios", "What was auto-filled"])
        with t1:
            exp, raw = explain_row(explainer, enc)
            show_drivers(exp, raw, float(np.expm1(exp.base_values)), pred)
        with t2:
            var = st.selectbox("Vary", list(SENS[ptype]), key=p + "sens_var")
            field, grid = SENS[ptype][var]
            rows = [assemble(plan, {**inp, field: v})[0] for v in grid]
            Xs, _ = prepare_X(plan, pd.DataFrame(rows))
            curve = pd.DataFrame({"Predicted value (€M)": predict_eur(model, encode(prep, Xs)) / 1e6},
                                 index=pd.Index(grid, name=var))
            st.line_chart(curve, y_label="€ millions")
            st.caption("Everything else held at the current inputs.")
        with t3:
            if ss[p + "scenarios"]:
                sc = pd.DataFrame(ss[p + "scenarios"])
                st.dataframe(sc, hide_index=True)
                st.bar_chart(sc.set_index("Scenario")["Value (€M)"])
                st.button("Clear scenarios", on_click=lambda: ss.__setitem__(p + "scenarios", []))
            else:
                st.caption("Change inputs and save a scenario to compare, e.g. staying vs moving clubs.")
        with t4:
            counts = Counter(how.values())
            st.caption(", ".join(f"{v} {k}" for k, v in counts.most_common()))
            tbl = pd.DataFrame({"Feature": [clean_ui_name(c) for c in plan["features"]],
                                "Value": [str(round(feats[c], 3)) if isinstance(feats[c], (int, float, np.floating))
                                          else str(feats[c]) for c in plan["features"]],
                                "Source": [how[c] for c in plan["features"]]})
            st.dataframe(tbl, hide_index=True, height=320)

    ss[p + "saved"] = {n: ss[p + n] for n in PERSISTED if p + n in ss}


# ==============================================================================
# 8. MAIN
# ==============================================================================
st.sidebar.title("⚽ Valuation engine")
mode = st.sidebar.radio("Mode", [MODE_A, MODE_B], key="mode")
ptype = st.sidebar.radio("Player category", ["Outfield", "Goalkeeper"], key="ptype")

try:
    model, prep, features, explainer = load_assets(ptype)
except FileNotFoundError as err:
    st.error(f"Missing model file: {err.filename}")
    st.stop()
df_players = load_player_database(ptype)
plan = build_plan(df_players, tuple(features), ptype, len(df_players))
st.sidebar.caption(f"{len(df_players):,} player-seasons · {len(features)} model features")

if mode == MODE_A:
    render_explorer(plan, df_players, model, prep, explainer, ptype)
else:
    render_simulator(plan, df_players, model, prep, explainer, ptype)