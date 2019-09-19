import torch
import cvxpy as cp
import time
import numpy as np
import pandas as pd
from enum import Enum
import sys
sys.path.append('..')

import OptMiniModule.util as optMini_util
import OptMiniModule.cvx_runpass as optMini_cvx
import basic_util as bUtil


import seaborn as sns
import matplotlib.pyplot as plt

sns.set('paper', style="whitegrid", font_scale=1.8, rc={"lines.linewidth": 2.5}, )

class SolverTest(Enum):
    GUROBI = 1
    MOSEK = 2
    FilterModule = 3


HORIZON = 48
params = dict(c_i=0.99, c_o=0.98, eta_eff=0.95, T=48, B=1.5, beta1=0.6, beta2=0.4, beta3=0.5, alpha=0.2)

def prof_cvx_instance(D, nBatch, param_set, p=None, solver=cp.GUROBI, cuda=False):

    batch_size = D.shape[0]
    Q, q, G, h, A, b, T, price = bUtil._form_QP_params(param_set, p)
    G_append = torch.cat([-torch.eye(T), torch.eye(T), torch.zeros((T, T))], dim=1)
    G = torch.cat([G, G_append], dim=0)

    Gs = [optMini_util.to_np(G) for i in range(batch_size)]
    hs = [optMini_util.to_np(torch.cat([h, d.view(T, 1)], dim=0)) for d in D]  # demand d is from data input
    Qs = [optMini_util.to_np(Q) for i in range(batch_size)]
    qs = [optMini_util.to_np(q) for i in range(batch_size)]
    As = [optMini_util.to_np(A) for i in range(batch_size)]
    bs = [optMini_util.to_np(b) for i in range(batch_size)]

    start = time.perf_counter()
    res = optMini_cvx.cvx_transform_solve_batch(Qs, qs, Gs, hs, As, bs, cp_sol=solver, n_jobs=10)
    end = time.perf_counter()
    t_ = end - start
    print("[%s] - [bs=%d] Compute solution and set up derivative: %.2f s." % (solver, nBatch, t_))
    return t_, solver

def prof_diffcp_instance(D, nBatch, param_set, p=None, solver='diffcp', debug=False):

    # def construct_QP_battery_w_D_conic_batch(param_set=None, D=None, p=None, debug=False):
    batch_size = D.shape[0]  # bs == batch size

    Q, q, G, h, A, b, T, price = bUtil._form_QP_params(param_set, p)
    G_append = torch.cat([-torch.eye(T), torch.eye(T), torch.zeros((T, T))], dim=1)
    G = torch.cat([G, G_append], dim=0)

    Gs = [optMini_util.to_np(G) for i in range(batch_size)]
    hs = [optMini_util.to_np(torch.cat([h, d.view(T, 1)], dim=0)) for d in D]  # demand d is from data input
    Qs = [optMini_util.to_np(Q) for i in range(batch_size)]
    qs = [optMini_util.to_np(q) for i in range(batch_size)]
    As = [optMini_util.to_np(A) for i in range(batch_size)]
    bs = [optMini_util.to_np(b) for i in range(batch_size)]


    # note : the following method solves the conic form of convex program
    start = time.perf_counter()
    x_sols_batch, y_sols_batch, s_sols_batch, Ds_batch, DTs_batch, As_batch, bs_batch, cs_batch = optMini_cvx.conic_transform_solve_batch(
        Qs, qs, Gs, hs, As, bs, n_jobs=10)
    end = time.perf_counter()
    t_ = end - start
    print("[%s] - [bs=%d] Compute solution and set up derivative: %.2f s." % (solver, nBatch, t_))
    return t_, solver



# nTrial=10
# res=None
# for i in range(nTrial):
#     for bsz in [2, 32, 64, 128]:
#         # bsz = 128
#         D_input = torch.rand(bsz, HORIZON)
#         price = torch.rand((HORIZON, 1))
#         t1, s1 = prof_cvx_instance(D_input, bsz, params, p=price, solver=cp.GUROBI)
#         t2, s2 = prof_cvx_instance(D_input, bsz, params, p=price, solver=cp.MOSEK)
#         t3, s3 = prof_diffcp_instance(D_input, bsz, params, p=price, solver='FilterModule')
#         r1 = np.array([[t1, SolverTest.GUROBI.value, bsz, i]])
#         r2 = np.array([[t2, SolverTest.MOSEK.value, bsz, i]])
#         r3 = np.array([[t3, SolverTest.FilterModule.value, bsz, i]])
#         block_res = np.concatenate((r1, r2, r3), axis=0)
#
#         res = block_res if res is None else np.concatenate((res, block_res), axis=0)
#
#     # if i % 2 == 0:
#     #     print(res)
#
# df=pd.DataFrame(res, columns=['time', 'solver', 'bsz', 'i'])

# df.to_csv('profiling_solver_speed.csv')
df = pd.read_csv('profiling_solver_speed.csv')

df['solver'] = df.solver.astype(int)
df['bsz'] = df.bsz.astype(int)
df['i'] = df.i.astype(int)

# plt.figure()
f, ax = plt.subplots(figsize=(7,5.0))
g=sns.catplot(x="bsz", y="time", hue="solver", kind="bar", data=df, ax=ax);

# new_labels = [SolverTest.GUROBI.name, SolverTest.MOSEK.name, SolverTest.FilterModule.name]
new_labels = [SolverTest.GUROBI.name, SolverTest.MOSEK.name, "Ours Batched"]

L=ax.legend()
print(L.get_texts())
# for t, l in zip(ax._legend.texts, new_labels):
#     t.set_text(l)
for t, l in zip(L.get_texts(), new_labels):
    t.set_text(l)
# ax.legend(new_labels)
ax.set_yscale('log')
ax.set_ylabel('Time (s)')
ax.set_xlabel('Batch Size')
# plt.tight_layout()
f.savefig('batch_compare_solver.png')