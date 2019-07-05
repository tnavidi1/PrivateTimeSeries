# parse the data

import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import metersUtil
import torch
from torch.utils.data import DataLoader, TensorDataset

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


def load_cached_files(dir_root='../Data_IrishCER', attr='income', filename='File2', n=50):
    if not os.path.isdir(dir_root):
        print(os.listdir(dir_root))
        raise FileNotFoundError("=== folder not exist ===")

    data = np.load(os.path.join(dir_root, '{:s}/n{:d}perID_{:s}.npz'.format(attr, n, filename)))
    return data['X'], data['Y']
    # plt.hist(data['Y'].flatten())
    # plt.show()


# Y_arr = None
# for fn in ['file1', 'file2']:
#     X, Y = load_cached_files(filename=fn)
#     if Y_arr is None:
#         Y_arr = Y
#     else:
#         Y_arr = np.vstack((Y_arr, Y))
#
# plt.hist(Y_arr.flatten())
# plt.show()

# raise NotImplementedError

df_income_id_label = metersUtil.load_static_attr('income')


def parse_X_Y(dir_root="../../Irish_CER_data_formated",
                     filename="reformated_File1.txt", nsample_per_mid=50,
                     id_label_df=df_income_id_label ): # .iloc[0:1000,:]):
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

    # internal function to parse the err string
    def f_parse(x):
        if isinstance(x, str):
            sx = x
        else:
            sx = str(x)

        if len(sx) > 4 or len(sx) < 4:
            return np.NaN

        elif len(sx) == 4:

            if sx.find('\x9f') > -1 or sx.find('\x94') > -1:
                return np.NaN
            else:
                # print(sx)
                return np.float(sx)

    # print(reformatDF["Meter_ID"].unique())
    # mids = np.unique(reformatDF["Meter_ID"].values)
    mids = reformatDF["Meter_ID"].dropna().unique()
    new_mids=np.vectorize(f_parse)(mids)
    # raise NotImplementedError
    subfile_mid_pool = new_mids[~np.isnan(new_mids)].astype(np.int64)

    # print(id_label_df["ID"])
    # print(subfile_mid_pool)
    max_subfile_mid = max(subfile_mid_pool)
    min_subfile_mid = min(subfile_mid_pool)

    print("file {:s} has meter id: minID={:d}, maxID={:d}".format(filename, min_subfile_mid, max_subfile_mid))


    attr_min_mid = min((id_label_df["ID"]).astype(np.int64))
    max_iter_idx = id_label_df[id_label_df["ID"] <= max_subfile_mid].index[-1]
    min_iter_idx = id_label_df[id_label_df["ID"] >= min_subfile_mid].index[0]

    max_iter = max_iter_idx - min_iter_idx
    if max_subfile_mid < attr_min_mid:
        raise NotImplementedError("Out range!!")

    # min_mid = max(min((id_label_df["ID"]).astype(np.int64)), min(subfile_mid_pool))
    # max_mid = min(max((id_label_df["ID"].astype(np.int64))), max(subfile_mid_pool))
    # print(min_mid, max_mid)

    i = 0
    batch_size = nsample_per_mid
    X = None
    Y = None
    for row in id_label_df.iloc[min_iter_idx:max_iter_idx,:].itertuples():
        i += 1
        # print(getattr(row, "ID"), getattr(row, "income_level"))
        mid = str(int(getattr(row, "ID")))
        income = int(getattr(row, "income_level"))
        mid_ts = metersUtil.iterateMeter(mid, reformatDF)
        # ==========================
        if mid_ts is None:
            print("skip meter id: {}".format(mid))
            continue

        if i > (max_iter):
            print("==" * 40)
            print("Finish all entries")
            print("==" * 40)
            break

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

        if i % 10 == 0:
            print("==" * 30)
            print("== {} meters, X size :{}, Y size : {}".format(i, X.shape, Y.shape))
            print("==" * 30)

    return X, Y


def get_train_test_split(dir_root='../'):

    # X, Y = parse_X_Y()
    X_all = np.array([]).reshape(0, 48)
    Y_all = np.array([]).reshape(0, 1)

    for fn in ['File1', 'File2', 'File3', 'File4']:
        X, Y = load_cached_files(dir_root=dir_root, attr='income', filename=fn, n=50)
        # print(X.shape)
        X_all = np.concatenate((X_all, X), axis=0)
        Y_all = np.concatenate((Y_all, Y), axis=0)

    print(X_all.shape, Y_all.shape)
    print("==" * 20)



    n_tt = int(X.shape[0] * 0.8)

    X_train, Y_train = X_all[:n_tt, :], Y_all[:n_tt, :]
    X_test, Y_test   = X_all[n_tt:, :], Y_all[n_tt:, :]

    arrays = {'X_train': torch.Tensor(X_train), 'Y_train': torch.Tensor(Y_train),
              'X_test': torch.Tensor(X_test), 'Y_test': torch.Tensor(Y_test)}

    return arrays


def get_loaders_tt(arrays_dict, bsz):
    train_loader = DataLoader(TensorDataset(
        arrays_dict['X_train'], arrays_dict['Y_train']), shuffle=False, batch_size=bsz)
    test_loader  = DataLoader(TensorDataset(
        arrays_dict['X_test'], arrays_dict['Y_test']), shuffle=False, batch_size=bsz)
    return {'train': train_loader, 'test': test_loader}

def get_train_hold_split(tensors_dict, th_frac, save_folder):
    X_train = tensors_dict['X_train']
    Y_train = tensors_dict['Y_train']

    np.random.seed(1)
    inds = np.random.permutation(X_train.size(0))

    if os.path.exists(os.path.join(save_folder, 'th_split_permutation')):
        inds = load_np_inds(os.path.join(save_folder, 'th_split_permutation'))
    else:
        with open(os.path.join(save_folder, 'th_split_permutation'), 'wb') as f:
            np.save(f, inds)

    train_inds = torch.LongTensor(inds[ :int(X_train.size(0) * th_frac)])
    hold_inds = torch.LongTensor(inds[int(X_train.size(0) * th_frac):])

    X_train2, X_hold2 = X_train[train_inds, :], X_train[hold_inds, :]
    Y_train2, Y_hold2 = Y_train[train_inds, :], Y_train[hold_inds, :]

    tensors_task = {'X_train': X_train2, 'Y_train': Y_train2,
            'X_hold': X_hold2, 'Y_hold': Y_hold2,
            'X_test': tensors_dict['X_test'].clone(),
            'Y_test': tensors_dict['Y_test'].clone()}
    return tensors_task


def get_loaders_tth(arrays_dict, bsz):
    train_loader = DataLoader(TensorDataset(
        arrays_dict['X_train'], arrays_dict['Y_train']), shuffle=False, batch_size=bsz)
    test_loader  = DataLoader(TensorDataset(
        arrays_dict['X_test'], arrays_dict['Y_test']), shuffle=False, batch_size=bsz)
    hold_loader  = DataLoader(TensorDataset(
        arrays_dict['X_hold'], arrays_dict['Y_hold']), shuffle=False, batch_size=bsz)
    return {'train': train_loader, 'test': test_loader, 'hold': hold_loader}


def load_np_inds(file):
    return np.load(file)


# ============================================== #
# == run the following for generating npz data== #
# ============================================== #

# n_ = 50
#
# for fname in ["File4", "File5"]:
#     X, Y = parse_X_Y(filename="reformated_{:s}.txt".format(fname), nsample_per_mid=n_, id_label_df=df_income_id_label) # .iloc[0:1000,:]
#     np.savez_compressed('../Data_IrishCER/income/n{:d}perID_{:s}.npz'.format(n_, fname), X=X, Y=Y)
#     print("==== save the file: {:s} ====".format(fname))






