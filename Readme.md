# Football Valuation Engine

A Streamlit app that estimates a footballer's market value with an XGBoost model and explains the estimate with SHAP. It covers outfield players and goalkeepers, each with its own model.

The app has two modes:

- **Player Explorer** compares the model's estimate with Transfermarkt for any player and season, and shows what pushes the price up or down.
- **What-If Simulator** lets you build a player, or start from a real one, change a few inputs, and see the new value. Everything that can be derived from those inputs is filled in automatically.

## Features

### Player Explorer

- Search for a player, optionally narrowed by league and club, and pick a season.
- Model estimate next to the Transfermarkt value, with a plain-language note on whether the model sees the player as under- or overvalued.
- Price drivers shown as percentage effects (for example, `Age = 24: +18%`) instead of raw log-scale SHAP values. The standard SHAP waterfall is available in an expander.
- Value over time: model estimate against Transfermarkt across all of a player's seasons.
- One click to send the player to the simulator with their real numbers pre-filled.

### What-If Simulator

- **Start from a real player or a typical one.** Loading a real player makes their actual stats the baseline, so you can ask "what if he were three years older, or played for a different club?"
- **Auto-fill.** You set a handful of headline inputs and the app derives the rest (see [How auto-fill works](#how-auto-fill-works)).
- **Live results.** The prediction updates as you change inputs. There is no calculate button.
- **Price drivers** for the simulated player.
- **Sensitivity curves** showing predicted value against age, contract length, minutes, and goals (outfield) or save % (goalkeepers).
- **Scenarios.** Save several what-ifs, such as staying versus moving clubs, and compare them in a table and bar chart.
- **Auto-filled tab.** Lists every model feature, its value, and where the value came from (your input, a typical value, a derived value, or an override).
- **Deep scouting.** Optional table to override any individual stat. A blank cell keeps the auto-filled value.

## How auto-fill works

When the app starts it inspects each model feature once and classifies it as a count, per-90 rate, ratio, category, or administrative field. It then fills in values like this:

| You set | The app derives |
|---|---|
| Minutes played | 90s played, matches, starts, and every count stat (scaled by minutes from the typical per-90 rate for the position) |
| Goals and assists | G+A, non-penalty goals, penalty goals and attempts, and all per-90 versions |
| Goals and assists (xG estimate on) | xG, npxG, xAG, npxG+xAG, using a per-position fit from your data |
| Save % (goalkeepers) | PSxG +/-, goals against, saves, PSxG, and related ratios, when the needed columns exist |
| Club | League |
| Any total you override in Deep scouting | The matching per-90 column |
| Nothing (not asked) | Position-typical values for all other stats, and the latest season for season-style features |

Per-90 columns are never asked for, because they follow from their totals.

## Project structure

```
Football/
├── app.py
├── README.md
├── requirements.txt
├── models/
│   ├── xgboost_valuation_model.joblib
│   ├── valuation_preprocessor.joblib
│   ├── feature_order.joblib
│   ├── gk_xgboost_model.joblib
│   ├── gk_preprocessor.joblib
│   └── gk_feature_order.joblib
└── data/
    ├── final_model_data.csv        (outfield; cleaned_model_data.csv also accepted)
    └── final_gk_data.csv           (goalkeepers)
```

The app looks for the data files in these locations, in order:

- Outfield: `data/final_model_data.csv`, `data/cleaned_model_data.csv`, `cleaned_model_data.csv`
- Goalkeeper: `data/final_gk_data.csv`, `final_gk_data.csv`, `data/cleaned_model_data.csv`

## Setup

Requires Python 3.9 or newer and Streamlit 1.38 or newer.

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

Suggested `requirements.txt`:

```
streamlit>=1.38
pandas
numpy
matplotlib
joblib
scikit-learn
xgboost
shap
```

Use the same `scikit-learn` and `xgboost` versions you trained with. Mismatched versions are the most common cause of model loading errors.

## What the app expects from your models and data

**Models**

- The XGBoost model is trained on `log1p(market value in EUR)`. The app converts predictions back with `expm1`.
- The preprocessor is a fitted scikit-learn transformer that takes a DataFrame with the columns in `feature_order`, and it must support `get_feature_names_out()`.
- `feature_order.joblib` is the list of raw input columns, in order.

**Data**

- A player name column, one of `name`, `player`, `player_name`, or `matched_fbref_name`.
- A target column whose name contains `market_value_in_eur`, which is used as the Transfermarkt value.
- `age_at_val` and `contract_days_left`, or the date columns needed to compute them (a valuation date, date of birth, and contract expiration date).
- Optional but used when present: `team`, `league`, `position`, `foot`, and `season`.
- FBref-style stat names such as `Performance_Gls`, `Per 90 Minutes_Gls`, `Expected_xG`, and `Playing Time_Min`.

Auto-fill relies on the FBref naming pattern `Group_Stat` to pair totals with their per-90 versions and to recognize goals, assists, and xG. Columns that don't follow it still work, but they are filled with typical values rather than derived from your inputs. The "What was auto-filled" tab in the simulator shows which is which.

## Reading the results

- **Price drivers** are multiplicative. Starting from the model's average player, each bar scales the value up or down, and together they give the prediction. Because the model works on `log1p(value)`, the percentages are approximate for small values.
- **Model estimate versus Transfermarkt** is a comparison, not a verdict. A gap can mean the market is mispricing a player, or that the model is missing something the data doesn't contain, such as injuries, reputation, or a transfer context.
- **Sensitivity curves** change one input and hold everything else constant. Real players rarely change one thing alone.

## Limitations

- Ratio stats such as shot accuracy do not update when you override their components in Deep scouting.
- xG, xAG, and goalkeeper PSxG +/- estimates are simple per-position line fits on per-90 rates. They are reasonable defaults, not predictions of real performance.
- Count stats are scaled linearly with minutes, which ignores role and tactics.
- Predictions are only as good as the underlying model. Extreme inputs outside the range of the training data may produce unreliable values.

## Troubleshooting

| Problem | Likely cause |
|---|---|
| `Missing model file` | A file is missing from `models/`, or you launched the app from a different folder. Run `streamlit run app.py` from the project root. |
| `No dataset found` | None of the expected CSV paths exist. Check the [project structure](#project-structure). |
| `Prediction failed` in the simulator | The preprocessor rejected a value. Check that `feature_order.joblib` matches the columns the preprocessor was fitted on. |
| Warning about missing model features in the explorer | The dataset lacks some columns the model expects. They are filled with defaults, so predictions for those rows may be off. |
| A stat doesn't follow your inputs | Its column name doesn't match the FBref `Group_Stat` pattern. See the auto-filled tab. |

## Tech stack

Streamlit, pandas, NumPy, scikit-learn, XGBoost, SHAP, Matplotlib, joblib.