# parse the data

import pandas as pd
import numpy as np
import os
import re
import matplotlib.pyplot as plt
import metersUtil

# "ISO-8859-1"


############
# separate #
############
# print(np.sort(reformatDF["Meter_ID"].unique()))
# print(reformatDF)

# counts_DF_file1 = reformatDF[["Meter_ID", "Elec_KW"]].groupby("Meter_ID", as_index=False).count()
# IDs = counts_DF_file1[counts_DF_file1["Elec_KW"] > 9000]["Meter_ID"]
# print(IDs)

# print(reformatDF.loc[reformatDF["Meter_ID"] == '1001'] )

# pid_ts = metersUtil.iterateMeter(1002, reformatDF)

# metersUtil.load_static_attr('income')
# metersUtil.load_static_attr('floor')

# "Washing machine", "Tumble dryer", "Dishwasher", "Electric shower"
#                    "Electric cooker", "Electric heater", "Stand alone freezer", "water pump", "Immersion"



# device_names = ["Washing machine", "Tumble dryer", "Dishwasher", "Electric shower", "hot tank",
#                 "Electric cooker", "Electric heater", "Stand alone freezer", "water pump", "Immersion"]
#
# metersUtil.load_static_attr('49001|49002', device=device_names[4])
# print(metersUtil._load_static_meterids())




df_income_id_label = metersUtil.load_static_attr('income')


def parse_train_test(dir_root="../../Irish_CER_data_formated",
                     filename="reformated_File1.txt", nsample_per_mid=150,
                     id_label_df=df_income_id_label.iloc[0:1000,:]):
    """

    :param dir_root:
    :param filename:
    :param nsample_per_mid:
    :return:
    """
    reformatDF = pd.read_csv(os.path.join(dir_root, filename), index_col=None,
                             encoding="iso-8859-1", sep=',',
                             low_memory=False, error_bad_lines=False, nrows=None)  # 24465540
    # Index(['Meter_ID', 'Year', 'Day', 'Hour', 'Minute', 'Elec_KW']
    # print(reformatDF.columns)
    # print(reformatDF['Meter_ID'].dtype)

    def f_parse(x):
        sx = str(x)
        if len(sx) > 4 or len(sx) < 4:
            return np.NaN
        elif len(sx) == 4:

            if sx.find('\x9f') > -1 or sx.find('\x94') > -1:
                return np.NaN
            else:
                return np.int(sx)

    mids = np.unique(reformatDF["Meter_ID"].values)
    new_mids=np.vectorize(f_parse)(mids)
    subfile_mid_pool = new_mids[~np.isnan(new_mids)].astype(np.int64)
    print(id_label_df["ID"])
    # print(type(id_label_df["ID"]))
    attr_id = min((id_label_df["ID"]).astype(np.int64))
    if max(subfile_mid_pool) < attr_id:
        raise NotImplementedError("Out range!!")
    # min_mid = max(min((id_label_df["ID"]).astype(np.int64)), min(subfile_mid_pool))
    # max_mid = min(max((id_label_df["ID"].astype(np.int64))), max(subfile_mid_pool))
    # print(min_mid, max_mid)
    # raise NotImplementedError
    i = 0
    batch_size = nsample_per_mid
    X = None
    Y = None
    for row in id_label_df.itertuples():
        i += 1
        # print(getattr(row, "ID"), getattr(row, "income_level"))
        mid = str(int(getattr(row, "ID")))
        income = int(getattr(row, "income_level"))
        mid_ts = metersUtil.iterateMeter(mid, reformatDF)
        # ==========================

        mid_ts_df = pd.DataFrame({'Date': mid_ts.index.date, 'Hr': mid_ts.index.hour,
                                  'Min': mid_ts.index.minute,
                                  'Step': (np.round(mid_ts.index.hour.astype(float) + mid_ts.index.minute.astype(float)/60.0, 1) * 2).astype(int),
                                  'Elec_KW': mid_ts.values})
        # reshape to one-day format
        daily_KW_matrix = mid_ts_df.pivot(index='Date', columns='Step', values='Elec_KW').dropna()
        nrows = daily_KW_matrix.shape[0]
        np.random.seed(1)
        nsamples = np.random.choice(nrows, size=min(batch_size, nrows-1), replace=False)
        df_mid_ts_attr_sub = pd.DataFrame({'Date': daily_KW_matrix.index[nsamples],
                                       'Income': np.repeat(income, min(batch_size, nrows-1))})

        daily_KW_matrix_sub=(daily_KW_matrix.iloc[nsamples, :].reset_index(level=['Date']))

        mid_batch_samples = pd.merge(daily_KW_matrix_sub, df_mid_ts_attr_sub, on='Date', how='inner')
        # cols: date , 1, 2..., 48(power), 49(label)
        if X is None and Y is None:
            X = mid_batch_samples.iloc[:, 1:49].to_numpy()
            Y = mid_batch_samples.iloc[:, 49:50].to_numpy()
        else:
            X = np.vstack((X, mid_batch_samples.iloc[:, 1:49].to_numpy() ))
            Y = np.vstack((Y, mid_batch_samples.iloc[:, 49:50].to_numpy() ) )


        if i > 2:
            print(X.shape)
            print(Y.shape)
            break


parse_train_test()

# print('\x9få\x0c\x94'.find('\x9f'))
# print( ('\x9få\x0c\x94'.find(')))
# print('1232'[0])
# print(re.match("(\x9f)|(\x0c)|(\x94)", '\x9få\x0c\x94') ) # .find('\x9f\x94'))
# print(re.match('\\x[0-9][a-z]', '\x9få\x0c\x94') ) # .find('\x9f\x94'))