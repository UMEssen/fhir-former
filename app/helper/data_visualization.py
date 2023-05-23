import pandas as pd
import pytz
import matplotlib.pyplot as plt
import seaborn as sns


def plot_top_15(df, column, year):
    top_15 = df[df["condition_year"] == year][column].value_counts().head(15)
    return top_15


def plot_side_by_side(df, years):
    plt.figure(figsize=(20, 10))

    for i, year in enumerate(years, 1):
        plt.subplot(2, 3, i)

        top_15_icd_code = plot_top_15(df, "icd_code", year)
        top_15_icd_code_root = plot_top_15(df, "icd_code_root", year)

        x1 = top_15_icd_code.index
        y1 = top_15_icd_code.values

        x2 = top_15_icd_code_root.index
        y2 = top_15_icd_code_root.values

        bar_width = 0.4
        r1 = range(len(x1))
        r2 = [x + bar_width for x in r1]

        plt.bar(r1, y1, width=bar_width, label="icd_code")
        plt.bar(r2, y2, width=bar_width, label="icd_code_root")

        plt.xticks([r + bar_width / 2 for r in range(len(x1))], x1, rotation=90)
        plt.title(f"Top 15 ICD Codes & ICD Code Roots in {year}")
        plt.xlabel("ICD Codes & ICD Code Roots")
        plt.ylabel("Frequency")
        plt.legend()

    plt.tight_layout()
    plt.show()


def check_icd_occurrence(config):
    conds_df = pd.read_feather(config["condition_path"])

    # sample_size = 0.2  # Adjust this value to change the sample size (0.1 = 10%)
    # conds_df = conds_df.sample(frac=sample_size)
    conds_df = conds_df[~pd.isna(conds_df.icd_code_root)]
    conds_df["condition_date"] = pd.to_datetime(conds_df["condition_date"])

    conds_df.reset_index(drop=True).to_feather("app/helper/conditions_dt.ftr")

    conds_df = pd.read_feather("app/helper/conditions_dt.ftr")

    conds_df["condition_year"] = conds_df["condition_date"].apply(
        lambda x: pd.to_datetime(x).year if not pd.isnull(x) else x
    )

    conds_df = conds_df.dropna(subset=["condition_date"])
    conds_df = conds_df[
        conds_df["condition_date"].apply(lambda x: isinstance(x, pd.Timestamp))
    ]

    print(conds_df["condition_date"].unique())
    print(conds_df["condition_date"].apply(type).unique())

    print(conds_df.columns)
    print(conds_df.shape)

    print(conds_df["condition_date"].min())
    print(conds_df["condition_date"].max())

    # conds_df.to_feather('app/helper/conditions.ftr')

    # conds_df = pd.read_feather('app/helper/conditions.ftr')
    print(conds_df.columns)
    print(conds_df.shape)
    print(len(conds_df))

    start_date = pd.Timestamp("2017-01-01", tz=pytz.FixedOffset(120))
    end_date = pd.Timestamp("2022-12-31", tz=pytz.FixedOffset(120))
    filtered_df = conds_df[
        (conds_df["condition_date"] >= start_date)
        & (conds_df["condition_date"] <= end_date)
    ]
    filtered_df.reset_index(drop=True).to_feather("app/helper/conditions_filtered.ftr")

    # years = [2017, 2018, 2019, 2020, 2021, 2022]
    years = [2017]
    plot_side_by_side(filtered_df, years)


def main(config):
    check_icd_occurrence(config)
