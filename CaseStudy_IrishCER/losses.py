"""
@Description: This scripts contains loss functions
"""


# params = dict(c_i=0.99, c_o=0.98, eta_eff=0.95, T=2, B=1.5, beta1=0.6, beta2=0.4, gamma=0.5, alpha=0.2)

def objective_loss(ctrl, price, params):
    T = params['T']
    x_in = ctrl[:, :T].float()
    x_out = ctrl[:, T:2*T].float()
    x_s = ctrl[:, 2*T:].float()
    costs = (
        (x_in - x_out) * price) + \
         (params['gamma'] * (x_s - params['B']*params['alpha'])**2) + \
         (params['beta1'] * x_in**2) + \
        (params['beta2'] * x_out**2)
    return costs.mean(dim=0)


def rmse_loss(Y_pred, Y_actual):
    return ((Y_pred - Y_actual)**2).mean(dim=0).sqrt().data



