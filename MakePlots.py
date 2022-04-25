import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from matplotlib import gridspec
from datetime import date, timedelta
from GetPath import RootPath

#RootPath = "/Users/haoyufan/ValidereAssignment_HF/"
if not os.path.exists(os.path.join(RootPath, "Figures")):
    os.mkdir(os.path.join(RootPath, "Figures"))

# Plot 1, APPL v.s. SP_500
# Get data of Apple
df_aapl = pd.read_csv(os.path.join(RootPath, "Data", "TimeBar_2016.csv"))
for year in np.arange(5) + 2017:
    df_add = pd.read_csv(os.path.join(RootPath, "Data", "TimeBar_%i.csv" % year))
    df_aapl = df_aapl.append(df_add, ignore_index=True)
df_aapl = df_aapl[["Date", "MJD", "ClosePrice"]]

df_plot = pd.DataFrame(columns=df_aapl.columns.to_list())
for mjd, subdf in df_aapl.groupby(by="MJD"):
    df_plot = df_plot.append(subdf.iloc[-1])
df_plot["Date"] = pd.to_datetime(df_plot["Date"].astype(str))
df_plot["MJD"] = df_plot["MJD"].astype(int)
df_plot.rename(columns={"ClosePrice": "AAPL"}, inplace=True)

# Get data of SP_500
df_sp500 = pd.read_csv(os.path.join(RootPath, "Data", "SP_500.csv"))
df_sp500 = df_sp500[["Date", "close"]]
df_sp500["Date"] = pd.to_datetime(df_sp500["Date"])
df_sp500.rename(columns={"close": "SP_500"}, inplace=True)

# merge and plot
df_plot = pd.merge(left=df_plot, right=df_sp500, on="Date", how="inner")
# get index to use
x_tick = []
x_tick_label = []
for year in np.arange(6) + 2016:
    date_start = pd.to_datetime("%i0101" % year)
    date_end = pd.to_datetime("%i1231" % year)
    x_tick.append(df_plot["MJD"].loc[(df_plot["Date"] >= date_start) & (df_plot["Date"] <= date_end)].mean())
    x_tick_label.append(str(year))

fig = plt.figure(figsize=[8, 4], dpi=200)
plt.gcf().subplots_adjust(hspace=0)
spec = gridspec.GridSpec(ncols=1, nrows=2,
                         height_ratios=[4, 4])

# Top panel for AAPL, no
ax0 = fig.add_subplot(spec[0])
plt.plot(df_plot["MJD"], df_plot["AAPL"])
plt.ylabel("AAPL Stock Price")
plt.xticks(x_tick, x_tick_label)
plt.grid()

# Bottom panel for SP_500
ax1 = fig.add_subplot(spec[1])
plt.plot(df_plot["MJD"], df_plot["SP_500"])
plt.xticks(x_tick, x_tick_label)
plt.grid()
plt.ylabel("S&P 500 index")
plt.xlabel('Time')

#plt.tight_layout()
plt.savefig(os.path.join(RootPath, "Figures", "AAPL_all_data.png"))
plt.close()




# violin plots on ARR
import seaborn as sns
fig = plt.figure(figsize=[8, 7], dpi=200)
plt.gcf().subplots_adjust(hspace=0)
spec = gridspec.GridSpec(ncols=1, nrows=3,
                         height_ratios=[4, 4, 4])

for i, period in enumerate(["26w", "1y", "3y"]):
    df2plot = pd.DataFrame(columns=["freq", "ARR_%s" % period])
    ax = fig.add_subplot(spec[i])
    for freq in ["1h", "2h", "4h", "1d", "2d", "3d", "1w", "2w", "4w"]:
        df = pd.read_csv(os.path.join(RootPath, "Result", "ROI", "%s_ROI.csv" % freq))
        df["freq"] = freq
        df = df[["freq", "ARR_%s" % period]]
        df2plot = df2plot.append(df, ignore_index=True)

    ax = sns.violinplot(x="freq", y="ARR_%s" % period, data=df2plot, ax=ax)
    ax.grid()

    if i == 0:
        plt.ylabel("6 months")

    if i == 1:
        plt.yticks([0, 100, 200], [0, 100, 200])
        plt.ylabel("1 year")

    if i == 2:
        plt.ylabel("3 years")
        plt.xlabel("Trading Frequency")

plt.savefig(os.path.join(RootPath, "Figures", "violin.png"))
plt.close()
