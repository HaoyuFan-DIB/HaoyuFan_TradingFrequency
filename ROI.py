# Functions to simulate changes on asset if applied the predicted label.
# Asset may be in the form of cash or stock value. Stock value changes with market,
# while cash is not affected.
# At the end of each time bar, if predicted label is 1, i.e. price would go up in the next
# trading period, transfer assets to stock so its value changes with market. Ohterwise,
# if predicted label is 0, i.e. pries would drop, transfer asset into cash so the value
# remains constant. That is, using the predicted label as a switch to change or no-change
# the asset, and the change amount is determined by the market
# Also, assume all cash can be convert into stock, which is not possible in reality...
import numpy as np
import pandas as pd
import os
from datetime import date, timedelta
from GetPath import RootPath


def execute_pred_label(df, trade_frequency="1d"):
    """
    Function to execute the trading decision based on the predicted label.
    :param df: pandas DataFrame, output of ModelEvaluation functions.
    :param trade_frequency: str, usually read from the name of csv filename
    :return: df, same as input with the additional "Asset" column that monitor
             the changes in cash/stock value.
    """
    unit_to_idx = {"h": 4, "d": 26, "w": 26 * 5}
    trade_idx = int(trade_frequency[0:-1]) * unit_to_idx[trade_frequency[-1]]

    # Get the price after one trading period, remove nan value in the tail
    df["FuturePrice"] = df["ClosePrice"].shift(-1 * trade_idx)
    df = df.loc[df["FuturePrice"].notna()]
    # changes = market_change ** predict_label = (future / current) ** predict_label
    df["Change"] = (df["FuturePrice"] / df["ClosePrice"]) ** df["Label_Predicted"]

    # Add Asset column to simulate the trade results
    # Add a default number of 10000 in the first trading period as the initial asset
    df["Asset"] = np.nan
    start_idx = 0
    stop_idx = trade_idx
    df["Asset"].iloc[start_idx:stop_idx] = 10000

    # Apply execution result to next trading period, in an iteration manner
    while True:
        start_asset = df["Asset"].iloc[start_idx:stop_idx].to_numpy()
        change_ratio = df["Change"].iloc[start_idx:stop_idx].to_numpy()
        start_idx = start_idx + trade_idx
        stop_idx = stop_idx + trade_idx
        if stop_idx > len(df) - 1:
            break
        df["Asset"].iloc[start_idx:stop_idx] = start_asset * change_ratio

    # The left over, if present, is not a whole trading period
    if start_idx <= len(df) - 1:
        df["Asset"].iloc[start_idx:] = (start_asset * change_ratio)[0:len(df) - start_idx]

    return df.drop(columns=["FuturePrice", "Change"])


def review_ROI(df, review_period="4w"):
    """
    Function to check ROI over a given review period. This is recorded in the new "ARR_period" column
    :param df: pandas DataFrame, the output of execute_pred_label function
    :param review_period: str, in an int-unit period, unit should be "w" for week or "y" for year.
                          If review period is month, it should be converted to w first, i.e.
                          1m = 4w, 2m = 8w, 3m = 13w, 6m = 26w.
    :return: df, same as input, with added "ARR_period" column
    """

    # All period units converted to week and year, including months.
    # 1d = 26 bar, 1w = 5d
    # 1m = 4w, 2m = 8w, 3m = 13w, 6m = 26w
    # 1y = 52w
    # index used for shifting
    unit_to_idx = {"w": 5 * 26, "y": 52 * 5 * 26}
    review_period_idx = int(review_period[0:-1]) * unit_to_idx[review_period[-1]]

    # year used to calculate annualized rate of return
    unit_to_year = {"w": 1 / 52, "y": 1}
    review_period_year = int(review_period[0:-1]) * unit_to_year[review_period[-1]]

    # ROI rate is asset_new / asset_now - 1
    # Also convert to annualized rate, in percentage
    df["ARR_%s" % review_period] = (df["Asset"].shift(-1 * review_period_idx) / df["Asset"] - 1)
    df["ARR_%s" % review_period] = df["ARR_%s" % review_period] * 100 / review_period_year

    return df

if __name__ == "__main__":
    if not os.path.exists(os.path.join(RootPath, "Result", "ROI")):
        os.mkdir(os.path.join(RootPath, "Result", "ROI"))

    period_to_review = ["4w", "8w", "13w", "26w", "1y", "2y", "3y"]
    ROI_summary = pd.DataFrame(columns=["TradeFrequency", "ReviewPeriod",
                                        "ARR_mean", "ARR_median",
                                        "ARR_std", "ARR_max", "ARR_min"])

    for root, dir, files in os.walk(os.path.join(RootPath, "Result", "Label")):
        for file in files:
            if not ".csv" in file:
                continue

            trade_frequency = file.split("_")[0]
            if os.path.exists(os.path.join(RootPath, "Result", "ROI", "%s_ROI.csv") % trade_frequency):
                df = pd.read_csv(os.path.join(RootPath, "Result", "ROI", "%s_ROI.csv") % trade_frequency)
            else:
                df = pd.read_csv(os.path.join(root, file))
                df = execute_pred_label(df, trade_frequency=trade_frequency)

            for period in period_to_review:
                if "ARR_%s" % period not in df.columns:
                    df = review_ROI(df, review_period=period)

                result_dict = {"TradeFrequency": trade_frequency,
                               "ReviewPeriod": period,
                               "ARR_mean": np.nanmean(df["ARR_%s" % period].to_numpy()),
                               "ARR_median": np.nanmedian(df["ARR_%s" % period].to_numpy()),
                               "ARR_std": np.nanstd(df["ARR_%s" % period].to_numpy()),
                               "ARR_max": np.nanmax(df["ARR_%s" % period].to_numpy()),
                               "ARR_min": np.nanmin(df["ARR_%s" % period].to_numpy())}
                ROI_summary = ROI_summary.append(result_dict, ignore_index=True)

            with open(os.path.join(RootPath, "Result", "ROI", "%s_ROI.csv" % trade_frequency), "w") as f:
                f.write(df.to_csv(index=False))

    # Also, add Sharp Ratio as an index
    # Sharp_Ratio = (ARR_product - ARR_risk-free) / std_ARR_product
    # Current risk-free return, such as from the U.S. Treasury Yield, is ~ 2.8%.
    ROI_summary["Sharp_Ratio"] = (ROI_summary["ARR_median"] - 2.8) / ROI_summary["ARR_std"]


    with open(os.path.join(RootPath, "Result", "ROI_Summary.csv"), "w") as f:
        f.write(ROI_summary.to_csv(index=False))




