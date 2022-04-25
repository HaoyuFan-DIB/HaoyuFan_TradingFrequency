# Contains functions to get data feed from algoseek
import numpy as np
import pandas as pd
import os
from datetime import date, timedelta
import boto3
import io
import copy
from GetPath import RootPath, access_key_id, secret_access_key


def get_date_list(start_date="2016-01-01", end_date="2021-12-31"):
    """
    Function to generate a pandas dataframe contains information on market date. The dataframe
    is then used to get data feed from algoseek.
    :param start_date: str, in the format of "yyyy-mm-dd";
    :param end_date: str, in the format of "yyyy-mm-dd";
    :return: df, pandas dataframe with the following columns:
             Date: str, in the format of yyyymmdd
             MJD: int, number of days passed since Nov. 17th, 1858
             EarlyClose, bool, if market closes early at 13:00 on that day
    """

    # set up start and end date
    start_date = start_date.split("-")
    start_date = date(int(start_date[0]), int(start_date[1]), int(start_date[2]))
    end_date = end_date.split("-")
    end_date = date(int(end_date[0]), int(end_date[1]), int(end_date[2]))

    # create output df, fill in date, mjd, and weekday
    df = pd.DataFrame(columns=["Date", "MJD", "WeekDay"])
    df["Date"] = pd.date_range(start_date, end_date, freq="d")
    df["MJD"] = pd.date_range(start_date, end_date, freq="d").to_julian_date() - 2400000.5
    df["MJD"] = df["MJD"].astype(int)
    df["WeekDay"] = df["Date"].dt.day_of_week

    # convert Date to str and prepare to merge
    df["Date"] = df["Date"].astype(str).str.replace("-", "")

    # readin holidays.csv, a short list of market closure dates due to holidays
    holiday_df = pd.read_csv(os.path.join(RootPath, "Data", "CalendarData", "holidays.csv"))
    holiday_df["TradeDate"] = holiday_df["TradeDate"].astype(str)
    holiday_df["IsHoliday"] = True
    df = pd.merge(left=df, right=holiday_df,
                  how="left",
                  left_on="Date", right_on="TradeDate")

    # Similarly for earlycloses.csv for early closures (usually the day before holiday)
    earlyclose_df = pd.read_csv(os.path.join(RootPath, "Data", "CalendarData", "earlycloses.csv"))
    earlyclose_df["TradeDate"] = earlyclose_df["TradeDate"].astype(str)
    earlyclose_df["EarlyClose"] = True
    df = pd.merge(left=df, right=earlyclose_df,
                  how="left",
                  left_on="Date", right_on="TradeDate")

    # Filtering out weekends and holidays, return Date, MJD, and EarlyClose columns
    df = df.loc[(df["WeekDay"].isin([0, 1, 2, 3, 4])) & (df["IsHoliday"].isna())]
    df["EarlyClose"].fillna(False, inplace=True)

    return df[["Date", "MJD", "EarlyClose"]]


def get_data_algoseek(key, bucket):
    """
    get data feed from algoseek and filter out non-trade events
    :param key: str, stock name, yyyymmdd/X/xxxx.csv.gz
    :param bucket: str, bucket name, should be "us-equity-taq-yyyy"
    :return: df, pandas dataframe, processed df with non-trade events filtered out
    """

    # Connect to algoseek and get data feed
    session = boto3.Session(aws_access_key_id=access_key_id,
                            aws_secret_access_key=secret_access_key,
                            region_name='us-east-1')
    s3_client = session.client('s3')
    response = s3_client.get_object(Bucket=bucket, Key=key, RequestPayer='requester')
    df = pd.read_csv(io.BytesIO(response['Body'].read()),compression='gzip')

    # pre_process df
    # filter event type and possible nan data in price and quantity, then drop unrelated columns
    df = df.loc[df["EventType"] == "TRADE NB"]
    df = df.loc[pd.notna(df["Price"]) & pd.notna(df["Quantity"])]
    df = df.loc[df["Quantity"] > 0]
    df = df[["Timestamp", "Price", "Quantity"]]

    # format timestamp, drop sub-second part, convert to timedelta (relative to opening time)
    df["Timestamp"] = df["Timestamp"].str.split(".").str[0]
    df["Timestamp_timedelta"] = pd.to_timedelta(df["Timestamp"])
    df["Timestamp_timedelta"] = df["Timestamp_timedelta"] - timedelta(hours=9, minutes=30)

    return df.reset_index(drop=True)


def process_day(df, early_close=False):
    """
    Turn data of entire trade day into a single time bar
    !!! REPLACED BY new function process_timebar()
    :param df: pandas dataframe, from get_data_algoseek
    :param early_close: bool, if market close early
    :return: dictionary to be append with the following keys:
             OpenTime, str, hh:mm:ss
             CloseTime, str, hh:mm:ss
             OpenPrice, float, price of first trade
             ClosePrice, float, price of last trade
             MinPrice, float, min price within the day
             MaxPrice, float, max price within the day
             Quantity, int, total quantity of stock traded
    """

    if early_close:
        close_hour = 13
    else:
        close_hour = 16

    open_time = timedelta(hours=9, minutes=30)
    close_time = timedelta(hours=close_hour)
    df = df.loc[(df["Timestamp_timedelta"] >= open_time) & (df["Timestamp_timedelta"] <= close_time)]
    dict_out = {"OpenTime": df["Timestamp"].iloc[0],
                "CloseTime": df["Timestamp"].iloc[-1],
                "OpenPrice": df["Price"].iloc[0],
                "ClosePrice": df["Price"].iloc[-1],
                "MinPrice": df["Price"].min(),
                "MaxPrice": df["Price"].max(),
                "Quantity": df["Quantity"].sum()}

    return dict_out


def process_hour(df, early_close=False):
    """
    Turn data of one trade day into multiple time bars of one hour
    !!! REPLACED BY new function process_timebar()
    :param df: pandas dataframe, from get_data_algoseek
    :param early_close: bool, if market close early
    :return: pandas dataframe to be append with the following columns:
             OpenTime, str, hh:mm:ss
             CloseTime, str, hh:mm:ss
             OpenPrice, float, price of first trade
             ClosePrice, float, price of last trade
             MinPrice, float, min price within the hour
             MaxPrice, float, max price within the hour
             Quantity, int, total quantity of stock traded within the hour
    """
    if early_close:
        open_hours = np.arange(4) + 9
    else:
        open_hours = np.arange(7) + 9
    close_hours = open_hours + 1

    df = df.loc[(df["Timestamp_timedelta"] >= timedelta(hours=9, minutes=30)) &
                (df["Timestamp_timedelta"] <= timedelta(hours=int(close_hours[-1])))]
    df["Hour"] = df["Timestamp"].str[0:2]
    if early_close:
        df["Hour"].loc[df["Hour"] == "13"] = "12"
    else:
        df["Hour"].loc[df["Hour"] == "16"] = "15"

    df_out = pd.DataFrame(columns=["OpenTime",
                                   "CloseTime",
                                   "OpenPrice",
                                   "ClosePrice",
                                   "MinPrice",
                                   "MaxPrice",
                                   "Quantity"])

    for hour, subdf in df.groupby(by="Hour"):
        dict_sub = {"OpenTime": subdf["Timestamp"].iloc[0],
                    "CloseTime": subdf["Timestamp"].iloc[-1],
                    "OpenPrice": subdf["Price"].iloc[0],
                    "ClosePrice": subdf["Price"].iloc[-1],
                    "MinPrice": subdf["Price"].min(),
                    "MaxPrice": subdf["Price"].max(),
                    "Quantity": subdf["Quantity"].sum()}
        df_out = df_out.append(dict_sub, ignore_index=True)

    return df_out


def process_timebar(df, barsize=timedelta(hours=1), early_close=False):
    """
    resample tick data from df into time bar of given size
    :param df: pandas dataframe, output of get_data_algoseek
    :param barsize: time.timedelta, size of time bar, default is 1 hour
    :param early_close: bool, if market closes early on that day, in which case, NaN will be used to fill up
    :return:
    """

    # filter tick data within market hour.
    # Market usually opens between 9:30 to 16:00, that is 6 hours and 30 minutes
    market_period = timedelta(hours=6, minutes=30)
    max_bar_idx = market_period // barsize  # bar index at 16:00

    # this index CANNOT be reached, so if market_period does not divide barsize, the max_bar_idx need to be increased
    if market_period % barsize != timedelta(0):
        max_bar_idx = max_bar_idx + 1

    # market closes early on some dates at 13:00, that is 3 hours and 30 minutes of market time
    if early_close:
        market_period = timedelta(hours=3, minutes=30)

    df = df.loc[(df["Timestamp_timedelta"] >= timedelta(0)) &
                (df["Timestamp_timedelta"] <= market_period)]

    # Get bar index for groupby
    df["BarIdx"] = df["Timestamp_timedelta"] // barsize
    # it is likely that market_perid can be divided by barsize, and a few trades will be left in the last bin
    # We want to check and correct that
    last_bar_id = market_period // barsize
    last_bar_replace = (market_period - timedelta(seconds=1)) // barsize
    if last_bar_id != last_bar_replace:
        df["BarIdx"].loc[df["BarIdx"] == last_bar_id] = last_bar_replace

    # Prepare output data
    df_out = pd.DataFrame(columns=["BarIdx",
                                   "OpenTime",
                                   "CloseTime",
                                   "OpenPrice",
                                   "ClosePrice",
                                   "MinPrice",
                                   "MaxPrice",
                                   "Quantity"])

    for bar_idx, subdf in df.groupby(by="BarIdx"):
        dict_sub = {"BarIdx": bar_idx,
                    "OpenTime": subdf["Timestamp"].iloc[0],
                    "CloseTime": subdf["Timestamp"].iloc[-1],
                    "OpenPrice": subdf["Price"].iloc[0],
                    "ClosePrice": subdf["Price"].iloc[-1],
                    "MinPrice": subdf["Price"].min(),
                    "MaxPrice": subdf["Price"].max(),
                    "Quantity": subdf["Quantity"].sum()}
        df_out = df_out.append(dict_sub, ignore_index=True)

    return df_out


def main(year):
    """
    main function for iteration, to be called by joblib for parallel.
    will calculate daily data for 2016 - 2021, and hourly data for 2019 - 2021.
    :param year: int, year to be calculated
    :return: None, but sampled time bars will be stored as ./Data/yyyy_daily.csv and yyyy_hourly.csv
    """

    # Add year info to parameters
    start_date = "%i-01-01" % year
    end_date = "%i-12-31" % year
    bucket_name = "us-equity-taq-%i" % year
    df_calander = get_date_list(start_date, end_date)

    # prepare output
    df_timebar = pd.DataFrame(columns=["Date",
                                   "MJD",
                                   "BarIdx",
                                   "OpenTime",
                                   "CloseTime",
                                   "OpenPrice",
                                   "ClosePrice",
                                   "MinPrice",
                                   "MaxPrice",
                                   "Quantity"])

    # Loop among days using df_calander
    for i in range(len(df_calander)):
        Date = df_calander["Date"].iloc[i]  # yyyymmdd
        print(Date)

        # get data from algoseek
        key_name = "%s/A/AAPL.csv.gz" % Date
        df_raw = get_data_algoseek(key=key_name, bucket=bucket_name)

        # process timebar and append to output df
        df_timebar_day = process_timebar(df_raw,
                                         barsize=timedelta(minutes=15),
                                         early_close=df_calander["EarlyClose"].iloc[i])
        df_timebar_day["Date"] = Date
        df_timebar_day["MJD"] = df_calander["MJD"].iloc[i]
        df_timebar = df_timebar.append(df_timebar_day, ignore_index=True)

    # save result
    print("Saving Data for year %i..." % year)
    with open(os.path.join(RootPath, "Data", "TimeBar_%i.csv" % year), "w") as f:
        f.write(df_timebar.to_csv(index=False))

    return None


if __name__ == "__main__":
    import time
    from joblib import Parallel, delayed
    pd.options.mode.chained_assignment = None

    tic = time.time()
    Parallel(n_jobs=6)(delayed(main)(year) for year in np.arange(6)+2016)
    #a = main(2019)
    print("Calculation finished!")
    print("Total time consumption is %i sec" % (time.time() - tic))

    # Also generate a calendar reference for all dates
    df_calendar = get_date_list(start_date="2016-01-01", end_date="2021-12-31")
    df_calendar.reset_index(drop=True, inplace=True)
    df_calendar.reset_index(drop=False, inplace=True)
    df_calendar.rename(columns={"index": "TradeDateIdx"}, inplace=True)
    with open(os.path.join(RootPath, "Data", "CalendarData", "TradeDateInfo.csv"), "w") as f:
        f.write(df_calendar.to_csv(index=False))