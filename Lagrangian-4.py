import numpy as np
import matplotlib.pyplot as plt


def getFinv(T, price_e, u_max, Q_max):
    diff_mat = np.zeros((T, 1))
    diff_mat = np.hstack((-np.eye(T), diff_mat))
    diff_mat[:, 1:] += np.eye(T)

    q0_mat = np.zeros((T + 1, 1))
    q0_mat[0] = 1
    # print(q0_mat)

    # Calculate approximate charging capacity constants
    mu = np.max(price_e) * 2 / Q_max
    gamma = np.max(price_e) * 2 / u_max
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

    return F_inv, gamma, mu

def u_star_of_d(d, price_e, gamma, T, Q0, F_inv):
    y = np.vstack((-price_e.T - 2 * gamma * (d - np.mean(d)), np.zeros((T + 1, 1)), np.zeros((T, 1)), -Q0))

    # print('y', y.shape)

    ans = F_inv @ y

    # print(ans)
    # print('ans', ans.shape)

    u_star = ans[0: T, :]
    Q_star = ans[T: 2 * T + 1, :]

    return u_star, Q_star

def costU(u, Q, price_e, mu, gamma, d, T):
    cost = np.dot(price_e, u) + np.dot(mu*np.ones(T+1), Q**2) + np.dot(gamma*np.ones(T), (u+d-np.mean(d))**2)

    return cost


if __name__ == '__main__':
    # define constants of problem
    price_e = np.hstack((.202 * np.ones((1, 12)), .463 * np.ones((1, 6)), .202 * np.ones((1, 6))))
    T = 24  # number of hours in optimization horizon
    # print('price', price_e.shape)

    u_min = -4 * 20  # multiply by 25 to represent 50% of homes with storage
    Q_min = 0.1
    Q_max = 12 * 20
    u_max = -u_min
    Q0 = 0  # about 50% capacity

    # Get lagrangian matrix inverse
    F_inv, gamma, mu = getFinv(T, price_e, u_max, Q_max)

    netDemandFull = np.loadtxt('netDemandFull.csv')
    netDemand = 1000 * netDemandFull[76, :]  # convert to KW Data is 50% solar
    d = netDemand[0:T]
    # print(d)
    d = d.reshape((T, 1))

    filters = np.arange(50)
    costs = []
    for filt in filters:
        d_hat = d + filt * np.random.normal(0, 1, (T ,1))

        # find optimal u and Q using d
        u_star, Q_star = u_star_of_d(d_hat, price_e, gamma, T, Q0, F_inv)

        cost = costU(u_star, Q_star, price_e, mu, gamma, d, T)
        # print(cost.shape)
        costs.append(cost.flatten())

    plt.figure()
    plt.plot(costs)
    plt.show()

    """
    u_real = u_star
    # u_real = u_star
    Q_real = Q_star - np.min(Q_star)

    # print('d', d)
    # print('u star', u_star)
    # print('Q', Q_star)

    plt.figure()
    plt.plot(u_real)
    plt.plot(Q_real)
    plt.legend(('u', 'Q'))

    plt.figure()
    plt.plot(d.reshape((T, 1)))
    plt.plot(d.reshape((T, 1))+u_real)
    plt.legend(('demand before battery', 'demand with battery'))

    plt.show()
    """