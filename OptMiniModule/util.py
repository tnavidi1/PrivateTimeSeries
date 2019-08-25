import torch
import numpy as np


def to_np(t):
    """
    expect to receive tensor
    :param t:
    :return:
    """
    if t is None:
        return None
    elif t.nelement() == 0:
        return np.array([])
    else:
        return t.cpu().numpy()



def convert_binary_label(y_label, median=4):
    if y_label.shape[1] == 1:
        y_ = y_label.squeeze()
        y_.apply_(lambda x: 1 if x >=median else 0)
        return y_.long()
    else:
        raise NotImplementedError("reshape y label: current size is {} ".format(y_label.shape))



def construct_G_batt_raw(T=24):
    """
    :@description

    We construct a G matrix in the senario of Battery charging without demand
    G = [[I 0 0] [-I 0 0 ] [0 I 0] [0 -I 0] [0 0 I] [0 0 -I]]

    :param T: time horizon
    :return: G matrix which is a torch tensor
    """

    G_1 = torch.cat([torch.eye(T), -torch.eye(T),
                     torch.zeros((T,T)), torch.zeros((T,T)),
                     torch.zeros((T,T)), torch.zeros((T,T))], dim=0)  # 1st block-column
    G_2 = torch.cat([torch.zeros((T,T)), torch.zeros((T,T)),
                     torch.eye(T), -torch.eye(T),
                     torch.zeros((T,T)), torch.zeros((T,T))], dim=0)  # 2nd block-column
    G_3 = torch.cat([torch.zeros((T,T)), torch.zeros((T,T)),
                     torch.zeros((T,T)), torch.zeros((T,T)),
                     torch.eye(T), -torch.eye(T)], dim=0)     # 3rd block-column
    G = torch.cat([G_1, G_2, G_3], dim=1)

    return G


def construct_h_batt_raw(T=24, c_i=1, c_o=1, batt_B=2):
    """
    We construct a h vector that comprise charging-in capacity $c_{in}$, discharging capacity $c_{out}$
    :param T:
    :param c_i:
    :param c_o:
    :param batt_B:
    :return:
    """
    h = torch.cat([torch.ones(T)*c_i, torch.zeros(T), \
               torch.ones(T)*c_o, torch.zeros(T), \
                torch.ones(T)*batt_B, torch.zeros(T)], dim=0)

    return h.unsqueeze(1)


def construct_A_batt_raw(T=24, eta=0.9):
    """

    :param T:
    :return:
    """
    A1 = torch.cat([torch.zeros(2*T), torch.cat([torch.ones(1), torch.zeros(T-1)], dim=0)], dim=0)
    A1 = A1.unsqueeze(0)
    I_ = torch.eye(T-1) # (T-1) x (T-1)
    A2 = torch.cat( [torch.cat([I_*eta, torch.zeros((T-1,1))], dim=1),
                     torch.cat([-I_, torch.zeros((T-1, 1))], dim=1),
                     torch.cat([I_, torch.zeros((T-1,1))], dim=1)-torch.cat([torch.zeros((T-1,1)), I_], dim=1)], dim=1)
    A = torch.cat([A1, A2], dim=0)
    return A


def construct_b_batt_raw(T=24, batt_init=1):
    b = torch.cat([torch.tensor([batt_init]),
                   torch.zeros(T-1)], dim=0)
    return b.unsqueeze(1)



def construct_Q_batt_raw(T=24, beta1=0.5, beta2=0.5, gamma=0.1):
    vec = torch.cat([torch.ones(T)*beta1, torch.ones(T)*beta2, torch.ones(T)*gamma], dim=0)
    Q=torch.diag(vec)
    return Q


def construct_q_batt_raw(T=24, price=None, batt_B=1, gamma=0.5, alpha=0.2):

    if price is None:
        torch.manual_seed(2)
        price = torch.rand((T, 1))

    if price.shape == (T, ):
        price=price.unsqueeze(1)

    if price.shape == (1, T):
        price = price.reshape(T, 1)

    q = torch.cat([price, -price, -2*gamma*alpha*batt_B*torch.ones((T,1))], dim=0)
    return q, price



def sample_gaussian(m, v):
    """
    Element-wise application reparameterization trick to sample from Gaussian

    Args:
        m: tensor: (batch, ...): Mean
        v: tensor: (batch, ...): Variance

    Return:
        z: tensor: (batch, ...): Samples
    """
    ################################################################################
    # TODO: Modify/complete the code here
    # Sample z
    ################################################################################

    eps_ = torch.randn_like(m)
    b_covs = eps_ * torch.sqrt(v)
    z = m + b_covs

    ################################################################################
    # End of code modification
    ################################################################################
    return z
