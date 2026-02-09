import pandas as pd

def add_econ_percentage(df:pd.DataFrame)->pd.DataFrame:
    total_col = "Total: All usual residents aged 16 years and over"
    econ_active = "Economically active (excluding full-time students)"
    econ_inactive = "Economically inactive"
    retired = "Economically inactive: Retired"
    student = "Economically inactive: Student"
    home_family = "Economically inactive: Looking after home or family"
    long_sick = "Economically inactive: Long-term sick or disabled"
    unemployed = "Economically active (excluding full-time students): Unemployed"
    se_with = "Economically active (excluding full-time students):In employment:Self-employed with employees"
    se_without = "Economically active (excluding full-time students):In employment:Self-employed without employees"

    preds=[total_col,econ_active,econ_inactive,retired,student,home_family,long_sick,unemployed,se_with,se_without]

    out = df.copy()
    for c in preds:
        if c in out.columns:
            out[c] = pd.to_numeric(out[c],errors="coerce")
        else:
            out[c] = pd.NA  # indicates missing cols

    out["total16plus"] = out[total_col].replace(0,pd.NA)
    out["pct_econ_active"] = out[econ_active]/out["total16plus"]*100
    out["pct_econ_inactive"] = out[econ_inactive]/out["total16plus"]*100
    out["pct_retired"] = out[retired]/out["total16plus"]*100
    out["pct_students"] = out[student]/out["total16plus"]*100
    out["pct_home_family"] = out[home_family]/out["total16plus"]*100
    out["pct_long_term_sick"] = out[long_sick]/out["total16plus"]*100
    out["pct_unemployed"] = out[unemployed]/out["total16plus"]*100
    out["self_employed_total"] = out[se_with]+out[se_without]
    out["pct_self_employed"] = out["self_employed_total"]/out["total16plus"]*100

    pct_cols = ["pct_econ_active","pct_econ_inactive","pct_retired","pct_students","pct_home_family","pct_long_term_sick","pct_unemployed","pct_self_employed"]
    out[pct_cols] = out[pct_cols].clip(lower=0, upper=100)

    return out