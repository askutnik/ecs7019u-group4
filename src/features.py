import pandas as pd

def add_econ_percentage(df:pd.DataFrame)->pd.DataFrame:
    #take census economic activity dataframe and add percentage columns
    total_col = "Total: All usual residents aged 16 years and over"
    df = df.copy()
    df["total16plus"] = df[total_col].replace(0, pd.NA)
    #aliasing
    econ_active = "Economically active (excluding full-time students)"
    econ_inactive = "Economically inactive"
    retired = "Economically inactive: Retired"
    student = "Economically inactive: Student"
    home_family = "Economically inactive: Looking after home or family"
    long_sick = "Economically inactive: Long-term sick or disabled"
    unemployed = "Economically active (excluding full-time students): Unemployed"
    #example percentages
    df["pct_econ_active"] = df[econ_active]/df["total16plus"]*100
    df["pct_econ_inactive"] = df[econ_inactive]/df["total16plus"]*100
    df["pct_retired"] = df[retired]/df["total16plus"]*100
    df["pct_students"] = df[student]/df["total16plus"]*100
    df["pct_home_family"] = df[home_family]/df["total16plus"]*100
    df["pct_long_term_sick"] = df[long_sick]/df["total16plus"]*100
    df["pct_unemployed"] = df[unemployed]/df["total16plus"]*100
    #self employed
    se_with = "Economically active (excluding full-time students):In employment:Self-employed with employees"
    se_without = "Economically active (excluding full-time students):In employment:Self-employed without employees"
    df["self_employed_total"] = df[se_with] + df[se_without]
    df["pct_self_employed"] = df["self_employed_total"] / df["total16plus"]*100
    return df
