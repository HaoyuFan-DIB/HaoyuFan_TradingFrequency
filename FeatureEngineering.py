# Functions used to generate features from the timebar data. These features
# will be used in the model that predict future trend to make decision.

# The two types of features generated are:
# 1. The momentum, or slope, over a certain period that directly show the trend;
# 2. The turnover rate, which is the ratio of traded quantity to total quantity,
#    that monitors the activity level of the market;

# Both features will be generated across multiple periods to show short- and
# long-term effect. These periods include: 1h, 2h, 4h (half day), 1d, 2d, 3d,
# 1w, 2w, 4w (1month), 8w (2month), 13w (3month/1quarter), and 26w (half year).

# Features on the hour and day level are calculated using index, where 1 hour = 4 bars
# and 1 day = 26 bars. Features on the week level are calculated using clock time.

import numpy as np
import pandas as pd
import os
from datetime import date, timedelta
from GetPath import RootPath
import matplotlib.pyplot as plt


def calc_norm_slope(x):
    """
    Function to calculate normalized slope. Slopes are obtained fom np.polyfit and then
    normalized by the median value of input x.
    :param x: pandas series, data feed from pandas.series.rolling(), dtype=float
    :return: normalized slope
    """
    x = x.to_numpy()
    x = x[~np.isnan(x)]
    if len(x) < 2:
        return np.nan
    else:
        slope = np.polyfit(np.arange(len(x)), x, 1)[0]
        slope_normalized = slope / np.median(x) * 100
        return slope_normalized


def get_feature(df, lookback=4):
    """
    To get the slope and turn-over percentage in a rolling window of given size.
    :param df: pandas DataFrame that contains the data in column "ClosePrice" and "Quantity"
    :param lookback: int or str, passed to pd.series.rolling() as window size.
    :return: pd.series, with the calculated slopes and turn-over rate.
    """

    total_volumn = 86.65 * 1000000
    slope = df["ClosePrice"].rolling(lookback, min_periods=2, closed="both").apply(calc_norm_slope)
    turnover = df["Quantity"].rolling(lookback, min_periods=2, closed="both").apply(np.nanmean) / total_volumn * 100

    return slope, turnover


if __name__ == "__main__":
    df = pd.read_csv(os.path.join(RootPath, "Data", "TimeBar_2016.csv"))
    for year in np.arange(5) + 2017:
        df_add = pd.read_csv(os.path.join(RootPath, "Data", "TimeBar_%i.csv" % year))
        df = df.append(df_add, ignore_index=True)

    # hour signals
    df_hour = df[["Date", "MJD", "BarIdx", "OpenTime", "CloseTime", "ClosePrice", "Quantity"]]
    for hour in [1, 2, 4]:
        print("Generating features for %i hours" % hour)
        df_hour["Slope_%ih" % hour], df_hour["TurnOver_%ih" % hour] = get_feature(df_hour, lookback=hour*4 - 1)
        df_hour["Slope_%ih" % hour].iloc[0:hour*4] = np.nan
        df_hour["TurnOver_%ih" % hour].iloc[0:hour*4] = np.nan

    with open(os.path.join(RootPath, "Data", "Features_Hour.csv"), "w") as f:
        f.write(df_hour.to_csv(index=False))

    # day signals
    df_day = df[["Date", "MJD", "BarIdx", "OpenTime", "CloseTime", "ClosePrice", "Quantity"]]
    for day in [1, 2, 3]:
        print("Generating features for %i days" % day)
        df_day["Slope_%id" % day], df_day["TurnOver_%id" % day] = get_feature(df_day, lookback=day*26 - 1)
        df_day["Slope_%id" % day].iloc[0:day*26] = np.nan
        df_day["TurnOver_%id" % day].iloc[0:day*26] = np.nan

    with open(os.path.join(RootPath, "Data", "Features_Day.csv"), "w") as f:
        f.write(df_day.to_csv(index=False))

    # week signals
    df_week = df[["Date", "MJD", "BarIdx", "OpenTime", "CloseTime", "ClosePrice", "Quantity"]]
    df_week["TimeStamp"] = pd.to_datetime(df_week["Date"].astype(str) + " " + df_week["CloseTime"])
    df_week.set_index("TimeStamp", inplace=True)
    for week in [1, 2, 4, 8, 13, 26]:
        print("Generating features for %i weeks" % week)
        df_week["Slope_%iw" % week], df_week["TurnOver_%iw" % week] = get_feature(df_week, lookback="%id" % (week * 7))

    df_week.reset_index(drop=False, inplace=True)
    for week in [1, 2, 4, 8, 13, 26]:
        boundary = df_week["TimeStamp"].iloc[0] + pd.offsets.Week(week)
        df_week["Slope_%iw" % week].loc[df_week["TimeStamp"] <= boundary] = np.nan
        df_week["TurnOver_%iw" % week].loc[df_week["TimeStamp"] <= boundary] = np.nan

    df_week.drop(columns=["TimeStamp"], inplace=True)
    with open(os.path.join(RootPath, "Data", "Features_Week.csv"), "w") as f:
        f.write(df_week.to_csv(index=False))