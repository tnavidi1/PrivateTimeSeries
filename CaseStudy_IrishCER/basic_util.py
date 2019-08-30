import torch

def convert_onehot(y_label, alphabet_size=6):
    y_ = y_label.long()
    one_hot = torch.FloatTensor(y_.size(0), alphabet_size).zero_()
    y_oh = one_hot.scatter_(1, (y_.data), 1) # if y is from 1 to 6
    return y_oh

def convert_binary_label(y_label, median=4):
    y_ = y_label.squeeze()
    y_.apply_(lambda x: 1 if x >=median else 0)
    return y_.long()
