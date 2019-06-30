"""
This script collect basic feature calculations 
"""
# import scipy
import numpy as np



### ---- 1. Consumption features ----- ###

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
def PCA_select(pid_ts, k=10):
    pid_ts_ = pid_ts[pid_ts < 5]
    num_sampled_days = int(np.floor(len(pid_ts_) / 48))
    n_length_ = int(48 * num_sampled_days)
    X_ = pid_ts_.values[0:n_length_].reshape((num_sampled_days, 48))

    from sklearn.decomposition import PCA
    pca_ = PCA(n_components=k)
    pca_.fit(X_)
    return pca_.components_, pca_.explained_variance_ratio_