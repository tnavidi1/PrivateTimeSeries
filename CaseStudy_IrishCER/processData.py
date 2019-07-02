# parse the data

import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import metersUtil

# "ISO-8859-1"

reformatDF = pd.read_csv("../../Irish_CER_data_formated/reformated_File1.txt",  index_col=None,
                         encoding="iso-8859-1", sep=',',
                         low_memory=False, error_bad_lines=False, nrows=None) #24465540

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



df_income_id_label = metersUtil.load_static_attr('income')

i=0
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

    daily_KW_matrix = mid_ts_df.pivot(index='Date', columns='Step', values='Elec_KW').dropna()
    print(daily_KW_matrix)
    print("="*20)
    print(daily_KW_matrix.iloc[1:10,:])
    # print(pd.melt(mid_ts_df, id_vars=["Date"], value_vars=["Elec_KW"]))
    if i > 2:
        break


