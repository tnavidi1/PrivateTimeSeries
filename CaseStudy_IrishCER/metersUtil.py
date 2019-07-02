"""
This script collect basic feature calculations
"""
# import scipy
import numpy as np
import pandas as pd

import datetime as dt
from datetime import datetime

import matplotlib.pyplot as plt


# process the meters by iteration -- TODO implement it in parallel later
#
def iterateMeter(id, cache):
    meter_pid_df = cache[cache["Meter_ID"] == id]

    # if meter id is empty
    if meter_pid_df.empty :
        print("the meter id: {} is empty".format(id))
        return None

    print("start to parse meter id: {} ...".format(id))
    start_day_delta = min(meter_pid_df["Day"])
    days_duration = max(meter_pid_df["Day"]) - start_day_delta
    fix_epoch_ref = datetime.strptime('2009-01-01', '%Y-%m-%d')
    start_day = fix_epoch_ref + dt.timedelta(days=start_day_delta)
    start_day_str = start_day.strftime('%Y-%m-%d')
    end_day = start_day + dt.timedelta(days=days_duration)
    end_day_str = end_day.strftime('%Y-%m-%d')
    #
    meter_pid_ts = pd.date_range(start_day_str, end_day_str, freq="30min")
    #
    meter_trim_df = meter_pid_df[meter_pid_df["Day"] < max(meter_pid_df["Day"])]
    #
    if len(meter_pid_ts) <= meter_trim_df.shape[0]:
        meter_pid_kw_ts = pd.Series(
            meter_trim_df[["Elec_KW"]].values[0:(len(meter_pid_ts))].reshape(len(meter_pid_ts), ),
            index=meter_pid_ts)
    # display(meter_trim_df[["Elec_KW"]].values[0:(len(meter_pid_ts))].reshape(1, len(meter_pid_ts)))
    else:
        meter_pid_ts = pd.date_range(start_day_str, periods=meter_trim_df.shape[0], freq="30min")
        meter_pid_kw_ts = pd.Series(
            meter_trim_df[["Elec_KW"]].values[0:(len(meter_pid_ts))].reshape(len(meter_pid_ts), ),
            index=meter_pid_ts)

    return meter_pid_kw_ts


## ==================================================

def load_static_attr(attr_name,
                     filepath="../../Irish_CER_data_formated/Survey_data_CSV_format/Smart_meters_Residential_pre-trial_survey_data.csv",
                     device=None):

    # income ""
    # floor area "ID|floor|61031"
    # -- (... Smart_meters_Residential_post-trial_survey_data.csv",  encoding = "ISO-8859-1")
    # appliance ""
    # pre_survey_res_df = pd.read_csv(filepath, low_memory=False, encoding="ISO-8859-1")
    # print(pre_survey_res_df)
    # attr_related_table = pre_survey_res_df.iloc[:, pre_survey_res_df.columns.str.contains(attr_name)]
    res_table = None
    pre_survey_res_df = pd.read_csv(filepath, low_memory=False, encoding="ISO-8859-1")
    attr_related_table = pre_survey_res_df.iloc[:, pre_survey_res_df.columns.str.contains(attr_name)]

    if attr_name.find('income') > -1 :
        res_table = _parse_income_tab(attr_related_table, pre_survey_res_df)
        # print(res_table)
    elif attr_name.find('floor') > -1 :
        res_table = _parse_sqft_tab(attr_related_table, pre_survey_res_df)
        # print(res_table)
    elif attr_name.find('49001|49002') > -1: #attr_name.find('appliance') > -1:
        res_table = _parse_appliance_tab(attr_related_table, pre_survey_res_df, device=device)

    else:
        raise NotImplementedError("not implemented the keyword")
    return res_table

def _load_static_meterids(filepath="../../Irish_CER_data_formated/Survey_data_CSV_format/Smart_meters_Residential_pre-trial_survey_data.csv"):
    """
    this loading id method is a dum
    :param filepath:
    :return:
    """
    pre_survey_res_df = pd.read_csv(filepath, low_memory=False, encoding="ISO-8859-1")
    meter_ids = pre_survey_res_df[["ID"]]
    return meter_ids


def _parse_income_tab(related_table, full_df):
    """
    internal function to retrieve attribute columns
    :param related_table:
    :param full_df:
    :return:
    """

    income_level_column = related_table.iloc[:, 3:5].apply(lambda x: x[1] if np.isnan(x[0]) else x[0], axis=1)
    # print(income_related_table)
    meter_income_df = full_df[["ID"]].join(pd.DataFrame(income_level_column, columns=["income_level"])).dropna()
    return meter_income_df

def _parse_sqft_tab(related_table, full_df):
    """

    :param related_table:
    :param full_df:
    :return:
    """
    # print(related_table)
    df_floor_filtered = related_table[(related_table.iloc[:, 0] < 999999) & (related_table.iloc[:, 0] > 1) ]
    # print(df_floor_filter.hist())
    # plt.show()

    sqft_df = full_df[["ID"]].join(df_floor_filtered).dropna()
    sqft_df.rename({"ID": "ID", sqft_df.columns[1]: "floor_area"}, axis=1, inplace=True)
    return sqft_df

def _parse_appliance_tab(related_table, full_df, device="Washing machine"):
    # print(related_table)
    # print(related_table.columns)
    """

    :param related_table:
    :param full_df:
    :param device: "Washing machine", "Tumble dryer", "Dishwasher", "Electric shower",
                    "hot tank"
                   "Electric cooker", "Electric heater", "Stand alone freezer", "water pump", "Immersion"
    :return:
    """
    # print(related_table.columns)
    df_filtered_device = related_table.iloc[:, related_table.columns.str.contains(device)]
    # print(df_filtered_device)
    device_df = full_df[["ID"]].join(df_filtered_device.iloc[:,0]).dropna()
    device_df.rename({"ID": "ID", device_df.columns[1]: device}, axis=1, inplace=True)
    # print(device_df)
    return device_df


### =========== 1. Consumption features ========== ###

# calculating P_tot_mean_week
def calWeekTotalMean(pid_ts):
    weekTotP =pid_ts.groupby([pid_ts.index.year, pid_ts.index.week]).sum()
    return weekTotP[weekTotP > 5].mean()

# unit testing
# display(calWeekTotalMean(pid_ts_02))

def calWeekdayTotalMean(pid_ts):
    pid_ts_ = pid_ts[pid_ts.index.weekday < 5]
    weekdayTotP = pid_ts_.groupby([pid_ts_.index.year, pid_ts_.index.week]).sum()
    return weekdayTotP[weekdayTotP > 5].mean()

# unit-testing
# display(calWeekdayTotalMean(pid_ts_02))

#
def calWeekendTotalMean(pid_ts):
    pid_ts_ = pid_ts[pid_ts.index.weekday > 4]
    weekdayTotP_ = pid_ts_.groupby([pid_ts_.index.year, pid_ts_.index.week]).sum()
    return weekdayTotP_[weekdayTotP_ > 2].mean()

# unit-testing
# display(calWeekendTotalMean(pid_ts_02))

def calDayMean(pid_ts):
    # between 6am - 10pm
    pid_ts_ = pid_ts[(pid_ts.index.hour >= 6) & (pid_ts.index.hour < 22)]
    dayTotP_ = pid_ts_.groupby([pid_ts_.index.year, pid_ts_.index.week, pid_ts_.index.weekday]).sum()
    return dayTotP_[dayTotP_ > 0.5].mean()

# unit-testing
# display(calDayMean(pid_ts_02))

def calEveningMean(pid_ts):
    # between 6pm - 10pm
    pid_ts_ = pid_ts[(pid_ts.index.hour >= 18 ) & (pid_ts.index.hour < 22)]
    nightTotP_ = pid_ts_.groupby([pid_ts_.index.year, pid_ts_.index.week, pid_ts_.index.weekday]).sum()
    return nightTotP_[nightTotP_ > 0.5].mean()

# unit-testing
# calEveningMean(pid_ts_02)


def calMorningMean(pid_ts):
    # between 6 a.m. to 10 a.m.
    pid_ts_ = pid_ts[(pid_ts.index.hour >= 6 ) & (pid_ts.index.hour < 10)]
    nightTotP_ = pid_ts_.groupby([pid_ts_.index.year, pid_ts_.index.week, pid_ts_.index.weekday]).sum()
    return nightTotP_[nightTotP_ > 0.5].mean()

# unit-testing
# calMorningMean(pid_ts_02)


def calNoonMean(pid_ts):
    # between 10 a.m. to 2 p.m.
    pid_ts_ = pid_ts[(pid_ts.index.hour >= 10 ) & (pid_ts.index.hour < 14)]
    nightTotP_ = pid_ts_.groupby([pid_ts_.index.year, pid_ts_.index.week, pid_ts_.index.weekday]).sum()
    return nightTotP_[nightTotP_ > 0.5].mean()



def calNightMean(pid_ts):
    # between 1 a.m. to 5 a.m
    pid_ts_ = pid_ts[(pid_ts.index.hour >= 1 ) & (pid_ts.index.hour < 5)]
    nightTotP_ = pid_ts_.groupby([pid_ts_.index.year, pid_ts_.index.week, pid_ts_.index.weekday]).sum()
    return nightTotP_[nightTotP_ > 0.5].mean()

# unit-testing
# display(calNightMean(pid_ts_02) )

def calWeekMax(pid_ts):
    weekMaxP_ = pid_ts.groupby([pid_ts.index.year, pid_ts.index.week]).max()
    return weekMaxP_[weekMaxP_ > 0.5 ].mean()

# unit-testing
# display(calWeekMax(pid_ts_02))

def calWeekMin(pid_ts):
    weekMinP_ = pid_ts.groupby([pid_ts.index.year, pid_ts.index.week]).min()
    return weekMinP_.mean()


## constructing features ##
### ----- 2. ratios

# 2.1 Mean P_bar over maximum P_bar

def r_MeanP_over_MaxP(pid_ts):
    r_ = calWeekTotalMean(pid_ts)/calWeekMax(pid_ts)
    return r_

# display(r_MeanP_over_MaxP(pid_ts_02))

def r_MinP_over_MeanP(pid_ts):
    r_ = calWeekMin(pid_ts)/calWeekTotalMean(pid_ts)
    return r_

# display(r_MinP_over_MeanP(pid_ts_02))


def r_Morning_over_Noon(pid_ts):
    r_ = calMorningMean(pid_ts)/calNoonMean(pid_ts)
    return r_

# display(r_Morning_over_Noon(pid_ts_02))

def r_Evening_over_Noon(pid_ts):
    r_ = calEveningMean(pid_ts)/calNoonMean(pid_ts)
    return r_

# display(r_Evening_over_Noon(pid_ts_02) )

def r_Noon_over_Day(pid_ts):
    r_ = calNoonMean(pid_ts)/calDayMean(pid_ts)
    return r_

# display(r_Noon_over_Day(pid_ts_02) )

def r_Night_over_Day(pid_ts):
    r_ = calNightMean(pid_ts)/calDayMean(pid_ts)
    return r_

# display(r_Night_over_Day(pid_ts_02) )

def r_Weekday_over_Weekend(pid_ts):
    r_ = calWeekdayTotalMean(pid_ts)/calWeekendTotalMean(pid_ts)
    return r_

### ----- 3. Temporal properties -------

# 3.1 Proportion of time with P > 0.5 kW
def propT_above(pid_ts, thres=0.5):
    pid_ts_ = pid_ts[pid_ts.index.weekday < 5]
    return (pid_ts_[pid_ts_ > thres].count()*1.0/float(len(pid_ts_)) )


def s_var(pid_ts):
    return(pid_ts[pid_ts.index.weekday < 5].var())

# display(s_var(pid_ts_02))

# 4.2 sum_t (|P_t - P_{t-1}|)
def s_diff(pid_ts):
    return(abs(pid_ts[pid_ts.index.weekday < 5].diff()[1:-1]).sum())

# display(s_diff(pid_ts_02))

# 4.3 cross-correlation
# cross correlation of subsequent days
def s_crosscorr(pid_ts, lag=48):
    pid_ts_ = pid_ts[pid_ts.index.weekday < 5]
    x_corr_ = pid_ts_.corr(pid_ts_.shift(lag))
    return x_corr_

# display(s_crosscorr(pid_ts_02))

# 4.4 number of counts of P_t - P_{t-1} > 0.2
# this feature shows the number of dramatic deviations on power
def s_num_peaks(pid_ts, deviation_thres=0.2):
    pid_ts_ = pid_ts[pid_ts.index.weekday < 5].diff()[1:-1]
    numCnts_ = pid_ts_[pid_ts_ > deviation_thres].count()
    return numCnts_


## feature construction
## ----- 5. Principal components -----
### PCA - select first 10 p.c.
## the PCA select function returns
##   loadingVectors (SVD) and explained var ratio
# --
# additional notes:  X = USV'; X project to orth space V (each column in V is an eigenvector)
#                   XV = USV'V = US
# --
def PCA_select(pid_ts, k=10, daily_resolution=48):
    pid_ts_ = pid_ts[pid_ts < 5]
    num_sampled_days = int(np.floor(len(pid_ts_) / daily_resolution))
    n_length_ = int(daily_resolution * num_sampled_days)
    X_ = pid_ts_.values[0:n_length_].reshape((num_sampled_days, daily_resolution))

    from sklearn.decomposition import PCA
    pca_ = PCA(n_components=k)
    pca_.fit(X_)

    return pca_.components_, pca_.explained_variance_ratio_
