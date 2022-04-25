# Check performance of models using AUC metric, also record model prediction to calculate ROI
import numpy as np
import pandas as pd
import os
from datetime import date, timedelta
import matplotlib.pyplot as plt
import time
from sklearn.metrics import roc_auc_score, confusion_matrix, ConfusionMatrixDisplay, log_loss, classification_report
import lightgbm as lgb
from scipy import stats
from GetPath import RootPath


def merge_feature_table(X_columns=None):
    """
    Function to read and merge per-engineered features and get column names to be used as features. Possible NaN
    values are also filtered out in the process.
    :param X_columns: None or list of str where each item should be a look-back period like '2h' or 1'k'
                      If None, all features will be used, otherwise only use features from specified periods.
                      Note each period is associated with two features, namely the normalized slope and average
                      turnover rate.
    :return: df_out, pandas.DataFrame, merged from data table of pre-engineered features
             X_columns, list, contains column names that will be used as feature in the decision-tree model.
    """
    df_hour = pd.read_csv(os.path.join(RootPath, "Data", "Features_Hour.csv"))
    df_day = pd.read_csv(os.path.join(RootPath, "Data", "Features_Day.csv"))

    # remove redundant columns in df_day and merge into df_hour
    df_day = df_day.drop(columns=["Date", "OpenTime", "CloseTime", "ClosePrice", "Quantity"])
    df_out = pd.merge(left=df_hour, right=df_day, on=["MJD", "BarIdx"], how="inner")

    # read in week features, remove redundant columns, and merge
    df_week = pd.read_csv(os.path.join(RootPath, "Data", "Features_Week.csv"))
    df_week = df_week.drop(columns=["Date", "OpenTime", "CloseTime", "ClosePrice", "Quantity"])
    df_out = pd.merge(left=df_out, right=df_week, on=["MJD", "BarIdx"], how="inner")

    # Filter out possible NaN rows and get X_columns
    for col in df_out.columns:
        df_out = df_out.loc[df_out[col].notna()]
    df_out.reset_index(drop=True, inplace=True)
    if X_columns is None:
        X_columns = df_out.columns[10:]
    else:
        X_columns = ["Slope_" + item for item in X_columns] + ["TurnOver_" + item for item in X_columns]

    # convert "Date" into datime64 dtype, add "TimeStamp" Column
    df_out["Date"] = pd.to_datetime(df_out["Date"].astype(str))
    df_out["TimeStamp"] = pd.to_datetime(df_out["Date"].astype(str) + " " + df_out["CloseTime"])

    # Also, read in the calendar data and get the index of trade/market day (not calendar day)
    # This helps to trace and select training data.
    df_calendar = pd.read_csv(os.path.join(RootPath, "Data", "CalendarData", "TradeDateInfo.csv"))
    df_calendar.drop(columns=["Date", "EarlyClose"], inplace=True)
    df_out = pd.merge(left=df_out, right=df_calendar, on="MJD", how="left")

    return df_out, X_columns


class ModelWalker():
    """
    Class to train and evaluate model with given trade frequency by walking through historical data.
    Requires un-labeled DataFrame from merge_feature_table(), and a trade_frequency to initialize.
    The trade_frequency should be a string consist of a number and a unit, such as "2h", "1d", or "4w".
    An optional input is X_columns, that is the column name to use as features, usually the output of
    merge_feature_table()

    Upon initialized, the class will add a "Label" column to the data frame which is the Y_column to predict.

    When walking through the historical data, the class would:
    1) Iterate among each month between 2017-01 and 2021-12;
    2) Take data from the target month as testing_data, and one-year (plus trading period) of data before the
       target month as training_data. If training and testing data are available, proceed, otherwise go to next
       month.
    3) Train a lgbm model with user specific parameters from the training dataset.
    4) Get the AUC metric from the trained model, as well as the predicted labels from testing dataset. Also get
       the AUC metric from a classifier generating random label, for comparison of the effectiveness.
    5) After the iteration, returns a) an array of AUC from the lgbm model among all valid months; b) an array of
       AUC from the random classifier among the same months; and c) a DataFrame containing the raw data and the
       predicted label for further analysis.
    """
    def __init__(self, df_unlabeled, trade_frequency="1d", X_columns=None):
        self.df = df_unlabeled
        self.trade_freq = trade_frequency

        # convert trade_frequency/lookback period into index unit
        # each hour has 4 bars, each trade day has 26 bars, each week has 5 trade days
        unit_to_idx = {"h": 4, "d": 26, "w": 26 * 5}
        self.trade_idx = int(trade_frequency[0:-1]) * unit_to_idx[trade_frequency[-1]]
        self.trade_day = int(np.max([1, np.ceil(self.trade_idx/26)]))

        # Add "Label" Column to self.df, this is the Y column to predict
        # For each row of self.df (info of a time bar), label will be 1 in ClosePrice is higher after trade_Frequency
        # Otherwise the value will be 0. This would fit into a binary classification task.
        self.df["Label"] = np.nan
        self.df["Label"].loc[self.df["ClosePrice"].shift(-1 * self.trade_idx) > self.df["ClosePrice"]] = 1
        self.df["Label"].loc[self.df["ClosePrice"].shift(-1 * self.trade_idx) < self.df["ClosePrice"]] = 0
        self.df = self.df.loc[self.df["Label"].notna()]

        # Don't forget the columns
        if X_columns is None:
            self.X_columns = self.df.columns[10:-2]
        else:
            self.X_columns = X_columns
        self.Y_column = "Label"

        # Lastly, create Result directory for output
        if not os.path.exists(os.path.join(RootPath, "Result")):
            os.mkdir(os.path.join(RootPath, "Result"))
        if not os.path.exists(os.path.join(RootPath, "Result", "Label")):
            os.mkdir(os.path.join(RootPath, "Result", "Label"))
        if not os.path.exists(os.path.join(RootPath, "Result", "Performance")):
            os.mkdir(os.path.join(RootPath, "Result", "Performance"))

    def get_data_df(self, year=2017, month=1):
        """
        Acquire training and testing data from given year and month. There must be at least of one year of training
        data, otherwise None will be returned.
        :param year: int, 2017 - 2021. Note we have data from 2016, but there is no way for any month in that year
                     that comes with one year of training data
        :param month: int, 1 - 12
        :return: df_train, df_test, both pandas.DataFrame, slices from self.df
        """

        test_start_date = pd.to_datetime("%i%02i01" % (year, month))
        test_end_date = test_start_date + pd.tseries.offsets.MonthEnd(1)
        if test_start_date < self.df["Date"].iloc[0]:
            return None
        df_test = self.df.loc[(self.df["Date"] >= test_start_date) & (self.df["Date"] <= test_end_date)]

        # Now get time range of training dataset
        # Note there is a gap of the size of trade_freq between the testing period and training data available.
        # That is, when we train the model at the start of testing period, the label within the gap cannot be
        # generated yet and hence they are not available for training.
        # To identify the end of training period:
        #    1. get TradeDateIdx of first day in testing period
        #    2. offset that with index with trade freq, to get the TradeDateIdx of the last day of training period
        #    3. get the calendar date of that TradeDateIdx
        #    4. look back for 365d from that date to get training data set.
        train_end_tradeidx = df_test["TradeDateIdx"].iloc[0] - self.trade_day
        if train_end_tradeidx < self.df["TradeDateIdx"].iloc[0]:
            return None
        train_end_date = self.df["Date"].loc[self.df["TradeDateIdx"] == train_end_tradeidx].iloc[0]
        train_start_date = train_end_date - pd.to_timedelta("365d")
        if train_start_date < self.df["Date"].iloc[0]:
            return None

        df_train = self.df.loc[(self.df["Date"] >= train_start_date) & (self.df["Date"] <= train_end_date)]

        return df_train, df_test

    def model_single_month(self, params, year=2017, month=1):
        """
        Acquire data from a specific month, train the model, and get AUC and predicted label.
        If not sufficient training data, a None will be returned.
        :param params: dict, parameters of the lgbm model.
        :param year: int, 2017 - 2021, to be passed to self.get_data_df()
        :param month: int, 1 - 12, to be passed to self.get_data_df()
        :return: dict_performance: dictionary, for Moneth, AUC_Model, AUC_Random, Log_Loss, Sensitivity,
                 and Specificity
                 df_month: pandas DataFrame, df contains basic info of each bar plus the predicted label
                 Or, if no data, None will be returned
        """

        data2use = self.get_data_df(year=year, month=month)
        if data2use is None:
            return None

        # set up model and do the training
        df_train = data2use[0]
        df_test = data2use[1]
        lgb_train = lgb.Dataset(df_train[self.X_columns], df_train[self.Y_column])
        lgb_test = lgb.Dataset(df_test[self.X_columns], df_test[self.Y_column])
        gbm = lgb.train(params, lgb_train, num_boost_round=1000, valid_sets=None, verbose_eval=-1)

        # Get dict_performance
        # Sometimes, when trade frequency is low (e.g. 4w), the real label are all identical and
        # np.nan will be returned for the two AUC and log_loss
        dict_performance = {"Month": "%i-%02i" % (year, month)}

        y_pred = gbm.predict(df_test[self.X_columns], num_iteration=gbm.best_iteration)
        y_model_pred = gbm.predict(df_train[self.X_columns], num_iteration=gbm.best_iteration)
        y_real = df_test["Label"].to_numpy()
        # get log_loss before rounding
        dict_performance["Log_Loss_Train"] = log_loss(df_train[self.Y_column].to_numpy(), y_model_pred)
        if len(np.unique(y_real)) == 1:
            dict_performance["Log_Loss_Test"] = np.nan
        else:
            dict_performance["Log_Loss_Test"] = log_loss(y_real, y_pred)


        # Other metric requires rounding
        y_model_pred = y_model_pred.round(0)
        dict_performance["AUC_Model_Train"] = roc_auc_score(df_train[self.Y_column].to_numpy(), y_model_pred)
        y_pred = y_pred.round(0)
        performance_report = classification_report(y_real, y_pred,
                                                   labels=[0, 1],
                                                   output_dict=True,
                                                   zero_division=0)  # surpass warns
        dict_performance["Sensitivity"] = performance_report["1"]["recall"]
        dict_performance["Specificity"] = performance_report["0"]["recall"]
        if len(np.unique(y_real)) == 1:
            dict_performance["AUC_Model_Test"] = np.nan
            dict_performance["AUC_Random"] = np.nan
        else:
            dict_performance["AUC_Model_Test"] = roc_auc_score(y_real, y_pred)

            # Get AUC of a random classifier.
            y_pred_random = np.random.rand(len(y_pred))
            y_pred_random = y_pred_random.round(0)
            dict_performance["AUC_Random"] = roc_auc_score(y_real, y_pred_random)

        df_month = df_test[df_train.columns[0:10].to_list() + ["Label"]]
        df_month["Label_Predicted"] = y_pred

        return dict_performance, df_month

    def walk(self, params):
        """
        Main method to be called that walk through the data. Iterate among all months, gather training
        result of that month, and aggregate results for output.
        :param params: dictionary, training parameters to be used by lgbm
        :return: df_performance, pandas DataFrame, model performance on
        """
        df_performance = pd.DataFrame(columns=["Month",
                                               "AUC_Model_Train", "AUC_Model_Test", "AUC_Random",
                                               "Log_Loss_Train", "Log_Loss_Test",
                                               "Sensitivity", "Specificity"])
        df_walk = pd.DataFrame(columns=self.df.columns[0:10].to_list() + ["Label", "Label_Predicted"])
        for year in np.arange(6) + 2016:
            for month in np.arange(12) + 1:
                print("=" * 30)
                print("%i - %02i, freq = %s" % (year, month, self.trade_freq))
                print("=" * 30)
                result_month = self.model_single_month(params, year=year, month=month)
                # result_month = None or (dict_performance, df_month)
                if result_month is not None:
                    df_performance = df_performance.append(result_month[0], ignore_index=True)
                    df_walk = df_walk.append(result_month[1], ignore_index=True)
                else:
                    print("No data, skipped...")

        return df_performance, df_walk


if __name__ == "__main__":
    params = {
            'boosting_type': 'gbdt',
            'objective': 'binary',
            'metric': 'auc',
            'num_leaves': 5,
            'learning_rate': 0.05,
            'feature_fraction': 0.8,
            'bagging_fraction': 0.8,
            'bagging_freq': 1,
            'num_iterations': 40,
        }



    df_unlabeled, X_columns = merge_feature_table()
    df_performance_summary = pd.DataFrame(columns=["TradeFrequency", "n_test",
                                                   "Log_Loss_Train", "Log_Loss_Train_err",
                                                   "Log_Loss_Test", "Log_Loss_Test_err",
                                                   "AUC_Model_Train", "AUC_Model_Train_err",
                                                   "AUC_Model_Test", "AUC_Model_Test_err",
                                                   "AUC_Random", "AUC_Random_err",
                                                   "p-value"])

    for freq in ["1h", "2h", "4h", "1d", "2d", "3d", "1w", "2w", "4w"]:
        tester = ModelWalker(df_unlabeled, trade_frequency=freq, X_columns=X_columns)
        df_performance, df_walk = tester.walk(params)

        # Write result of single frequency
        with open(os.path.join(RootPath, "Result", "Label", "%s_Predicted_Labels.csv" % freq), "w") as f:
            f.write(df_walk.to_csv(index=False))
        with open(os.path.join(RootPath, "Result", "Performance", "%s_Performance.csv" % freq), "w") as f:
            f.write(df_performance.to_csv(index=False))

        # Appending overall performance
        auc_model = df_performance["AUC_Model_Test"].to_numpy()
        auc_random = df_performance["AUC_Random"].to_numpy()
        t_result = stats.ttest_ind(auc_model[~np.isnan(auc_model)], auc_random[~np.isnan(auc_random)])
        result_dict = {"TradeFrequency": freq,
                       "n_test": len(df_performance),
                       "Log_Loss_Train": np.nanmean(df_performance["Log_Loss_Train"].to_numpy()),
                       "Log_Loss_Train_err": np.nanstd(df_performance["Log_Loss_Train"].to_numpy()),
                       "Log_Loss_Test": np.nanmean(df_performance["Log_Loss_Test"].to_numpy()),
                       "Log_Loss_Test_err": np.nanstd(df_performance["Log_Loss_Test"].to_numpy()),
                       "AUC_Model_Train": np.nanmean(df_performance["AUC_Model_Train"].to_numpy()),
                       "AUC_Model_Train_err": np.nanstd(df_performance["AUC_Model_Train"].to_numpy()),
                       "AUC_Model_Test": np.nanmean(df_performance["AUC_Model_Test"].to_numpy()),
                       "AUC_Model_Test_err": np.nanstd(df_performance["AUC_Model_Test"].to_numpy()),
                       "AUC_Random": np.nanmean(df_performance["AUC_Random"].to_numpy()),
                       "AUC_Random_err": np.nanstd(df_performance["AUC_Random"].to_numpy()),
                       "p-value": t_result.pvalue}
        df_performance_summary = df_performance_summary.append(result_dict, ignore_index=True)

        print("Trading Frequency = %s, n_months = %i" % (freq, len(df_performance)))
        print("Log_Loss of Model is %.3f pm %.3f" % (result_dict["Log_Loss_Test"], result_dict["Log_Loss_Test_err"]))
        print("AUC of Model is %.3f pm %.3f" % (result_dict["AUC_Model_Test"], result_dict["AUC_Model_Test_err"]))
        print("AUC of Random Guesser is %.3f pm %.3f" % (result_dict["AUC_Random"], result_dict["AUC_Random_err"]))
        print("p-value is %f" % t_result.pvalue)

    with open(os.path.join(RootPath, "Result", "Performance_Summary.csv"), "w") as f:
        f.write(df_performance_summary.to_csv(index=False))
