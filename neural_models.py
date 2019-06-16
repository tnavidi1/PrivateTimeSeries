import torch

from torch import nn
from torch.nn import functional as F

import os


class ClassifierLatent(nn.Module):
    def __init__(self, z_dim=10, y_dim=0):
        super(ClassifierLatent, self).__init__()
        self.z_dim = z_dim
        self.y_dim = y_dim
        self.net = nn.Sequential(
            nn.Linear(z_dim, 15),
            nn.ELU(),
            nn.Linear(15, y_dim)
            # nn.Linear(z_dim, y_dim),
        )

    def forward(self, x):
        o = self.net(x)
        return o


class LinearFilter(nn.Module):
    def __init__(self, input_dim = 10, y_dim=0, output_dim=10):
        super(LinearFilter, self).__init__()
        self.input_dim = input_dim
        self.y_dim = y_dim
        self.output_dim = output_dim
        self.fc = nn.Linear(self.input_dim + self.y_dim, self.output_dim)

    def forward(self, x, y=None):
        """

        :param x: input the feature vectors
        :param y: the default y input would be one-hot encoding version of discrete labels
        :return:
        """
        xy = x if y is None else torch.cat([x, y], dim=1)
        o = self.fc(xy)
        return o


class Generator(nn.Module):
    def __init__(self, nn='v1', name='g_filter', z_dim=24, y_priv_dim=2, device=None):

        super().__init__()
        self.name = name
        self.z_dim = z_dim
        self.y_priv_dim = y_priv_dim

        self.device = device

        # create a linear filter
        self.filter = LinearFilter(self.z_dim, self.y_priv_dim, output_dim=self.z_dim)  # setting noise dim is same as latent dim

        # Set prior as fixed parameter attached to module
        self.z_prior_m = torch.nn.Parameter(torch.zeros(1), requires_grad=False)
        self.z_prior_v = torch.nn.Parameter(torch.ones(1), requires_grad=False)
        self.z_prior = (self.z_prior_m, self.z_prior_v)

    def forward(self, x, y=None):
        batch_size = x.size()[0] # batch size is the first dimension

        z_noise = self.sample_z(batch=batch_size)
        z_noise = z_noise / z_noise.norm(2, dim=1).unsqueeze(1).repeat(1, self.z_dim)

        x_proc_noise = self.filter(z_noise, y)

        x_noise = x + x_proc_noise

        return x_noise

    def sample_z(self, batch):
        return sample_gaussian(self.z_prior[0].expand(batch, self.z_dim),
                                self.z_prior[1].expand(batch, self.z_dim))


def sample_gaussian(m, v):
    """
    Element-wise application reparameterization trick to sample from Gaussian

    Args:
        m: tensor: (batch, ...): Mean
        v: tensor: (batch, ...): Variance

    Return:
        z: tensor: (batch, ...): Samples
    """

    eps_ = torch.randn_like(m)
    b_covs = eps_ * torch.sqrt(v)
    z = m + b_covs

    return z


def save_model_by_desc(model, name, desc, global_step, root='f-div'):

    save_dir = os.path.join('checkpoints', root)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    save_dir = os.path.join(save_dir, name)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    save_dir = os.path.join(save_dir, desc)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    file_path = os.path.join(save_dir, 'model-{:05d}.pt'.format(global_step))
    state = model.state_dict()
    torch.save(state, file_path)
    print('Saved to {}'.format(file_path))



if __name__ == '__main__':
    # load static parameters
    T = 24

    d_priv = ClassifierLatent(z_dim=24, y_dim=2)

    generator = Generator(z_dim=10, y_priv_dim=10, device=None)
