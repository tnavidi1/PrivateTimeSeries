import numpy as np

import torch


class Optimizer:

    def __init__(self, T, price_e, u_max, Q_max, Q0):
        self.F_inv = None
        self.gamma = None
        self.mu = None
        self.T = T
        self.price_e = price_e
        self.u_max = u_max
        self.Q_max = Q_max
        self.Q0 = Q0

    def getFinv(self):
        price_e = self.price_e
        T = self.T
        u_max = self.u_max
        Q_max = self.Q_max

        diff_mat = np.zeros((T, 1))
        diff_mat = np.hstack((-np.eye(T), diff_mat))
        diff_mat[:, 1:] += np.eye(T)

        q0_mat = np.zeros((T + 1, 1))
        q0_mat[0] = 1
        # print(q0_mat)

        # Calculate approximate charging capacity constants
        mu = np.max(price_e) * 2 / Q_max
        gamma = np.max(price_e) * 2 / u_max
        self.gamma = gamma
        self.mu = mu
        # print(gamma.shape)

        F = np.block([
            [2 * gamma * np.eye(T), np.zeros((T, T + 1)), np.eye(T), np.zeros((T, 1))],
            [np.zeros((T + 1, T)), 2 * mu * np.eye(T + 1), -diff_mat.T, -q0_mat],
            [np.eye(T), -diff_mat, np.zeros((T, T + 1))],
            [np.zeros((1, T)), -q0_mat.T, np.zeros((1, T + 1))]
        ])

        # print('F', F.shape)
        # print(np.linalg.matrix_rank(F))
        # np.savetxt('f_mat.csv', F)

        F_inv = np.linalg.inv(F)
        self.F_inv = F_inv

        return F_inv, gamma, mu

    def u_star_of_d(self, d):

        batch_size, T = d.shape

        # pytorch version
        price_e = self.price_e
        gamma = self.gamma
        T = self.T
        Q0 = self.Q0
        F_inv = torch.FloatTensor(self.F_inv)

        d = d.transpose(0,1)
        d_bar = torch.mean(d, dim=0, keepdim=True)

        # print('shape of d', d.shape)
        # print(d_bar.shape)

        y = torch.cat((torch.FloatTensor(-price_e.T - 2 * gamma) * (d - d_bar), torch.zeros(T + 1, batch_size),
                       torch.zeros(T, batch_size), torch.FloatTensor(np.tile(-Q0, (1,batch_size)))), dim=0)

        # print(torch.FloatTensor(np.tile(-Q0, (1,batch_size))).shape)
        # print('shape of y', y.shape)

        # print('y', y.shape)

        ans = F_inv @ y

        u_star = ans[0: T, :]
        Q_star = ans[T: 2 * T + 1, :]

        # print('shape of ans', u_star.shape)

        return u_star, Q_star

    def costU(self, u, Q, d):
        # pytorch version
        price_e = torch.FloatTensor(self.price_e)
        mu = self.mu
        gamma = self.gamma
        T = self.T

        d = d.transpose(0,1)

        # print('u shape', u.shape)
        # print('d shape', d.shape)

        d_bar = torch.mean(d, dim=0, keepdim=True)

        cost = price_e @ u + torch.FloatTensor(mu * np.ones(T + 1)) @ Q ** 2 + \
               torch.FloatTensor(gamma * np.ones(T)) @ (u + d - d_bar) ** 2

        # print('cost shape', cost.shape)
        cost = cost.mean()

        # cost = torch.mm(price_e, u) + torch.mm(torch.FloatTensor(mu * np.ones(T + 1)), Q ** 2) + \
        #       torch.mm(torch.FloatTensor(gamma * np.ones(T)), (u + d - d_bar) ** 2)

        # print('cost shape', cost.shape)

        return cost

    def u_star_of_d_np(self, d):
        # numpy version
        price_e = self.price_e
        gamma = self.gamma
        T = self.T
        Q0 = self.Q0
        F_inv = self.F_inv

        y = np.vstack((-price_e.T - 2 * gamma * (d - np.mean(d)), np.zeros((T + 1, 1)), np.zeros((T, 1)), -Q0))

        # print('y', y.shape)

        ans = F_inv @ y

        # print(ans)
        # print('ans', ans.shape)

        u_star = ans[0: T, :]
        Q_star = ans[T: 2 * T + 1, :]

        return u_star, Q_star

    def costU_np(self, u, Q, d):
        # numpy version
        price_e = self.price_e
        mu = self.mu
        gamma = self.gamma
        T = self.T
        cost = np.dot(price_e, u) + np.dot(mu * np.ones(T + 1), Q ** 2) + np.dot(gamma * np.ones(T),
                                                                                 (u + d - np.mean(d)) ** 2)

        return cost
