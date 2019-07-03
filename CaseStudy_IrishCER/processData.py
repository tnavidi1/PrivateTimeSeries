# parse the data

import pandas as pd
import numpy as np
import os
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
# print(pid_ts)

# metersUtil.load_static_attr('income')
# metersUtil.load_static_attr('floor')

# "Washing machine", "Tumble dryer", "Dishwasher", "Electric shower"
#                    "Electric cooker", "Electric heater", "Stand alone freezer", "water pump", "Immersion"



# device_names = ["Washing machine", "Tumble dryer", "Dishwasher", "Electric shower", "hot tank",
#                 "Electric cooker", "Electric heater", "Stand alone freezer", "water pump", "Immersion"]
#
# metersUtil.load_static_attr('49001|49002', device=device_names[4])
# print(metersUtil._load_static_meterids())







def parse_train_test(dir_root="../../Irish_CER_data_formated",
                     filename="reformated_File1.txt", nsample_per_mid=150):
    """

    :param dir_root:
    :param filename:
    :param nsample_per_mid:
    :return:
    """
    reformatDF = pd.read_csv(os.path.join(dir_root, filename), index_col=None,
                             encoding="iso-8859-1", sep=',',
                             low_memory=False, error_bad_lines=False, nrows=None)  # 24465540

    df_income_id_label = metersUtil.load_static_attr('income')

    i = 0
    batch_size = nsample_per_mid
    X = None
    Y = None
    for row in df_income_id_label.itertuples():
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


        # if i > 2:
        #     print(X)
        #     print(Y)
        #     break


parse_train_test()