import numpy as np
import pandas as pd

import matplotlib
gui_env = ['TKAgg','GTKAgg','Qt4Agg','WXAgg']
for gui in gui_env:
    try:
        print("testing", gui)
        matplotlib.use(gui,warn=False, force=True)
        from matplotlib import pyplot as plt
        break
    except:
        continue
print("Using:",matplotlib.get_backend())

import seaborn as sns
sns.set('paper', style="whitegrid", font_scale=1.5, rc={"lines.linewidth": 2.5}, )


if __name__ == '__main__':

    # load results
    data = np.load('cost_vs_distortion.npz')
    faccs_u = data['faccs_u']
    faccs_p = data['faccs_p']
    faccs_p_r = data['faccs_p_r']

    epses = (np.arange(20) + 1) * 10.

    print(np.array(faccs_p).shape)

    u_max = np.max(np.array(faccs_u), axis=0)
    u_min = np.min(np.array(faccs_u), axis=0)
    u_ave = np.mean(np.array(faccs_u), axis=0)
    p_max = np.max(np.array(faccs_p), axis=0)
    p_min = np.min(np.array(faccs_p), axis=0)
    p_ave = np.mean(np.array(faccs_p), axis=0)
    pr_max = np.max(np.array(faccs_p_r), axis=0)
    pr_min = np.min(np.array(faccs_p_r), axis=0)
    pr_ave = np.mean(np.array(faccs_p_r), axis=0)

    datag = np.load('gaussMech.npz')
    costs = datag['fcosts']
    epses = (np.arange(20) + 1.) * 2. * np.sqrt(24)

    n = np.array(costs).shape[0]

    plt.figure()
    sns.lineplot(np.tile(epses, n), np.array(costs).flatten())
    plt.show()

    """
    plt.figure()
    plt.plot(epses, u_max)
    plt.plot(epses, u_min)
    plt.plot(epses, u_ave)
    plt.figure()
    plt.plot(epses, p_max)
    plt.plot(epses, p_min)
    plt.plot(epses, p_ave)
    plt.figure()
    #plt.plot(epses, pr_max)
    #plt.plot(epses, pr_min)
    #plt.plot(epses, pr_ave)
    plt.errorbar(epses, pr_ave, yerr=[pr_min - pr_ave, pr_ave - pr_max])
    

    plt.figure()
    sns.lineplot(np.tile(epses, 10), np.array(faccs_u).flatten())
    ax2 = plt.twinx()
    sns.lineplot(np.tile(epses, 10), np.array(faccs_p).flatten(), ax=ax2)
    plt.figure()
    sns.lineplot(np.tile(epses, 10), np.array(faccs_p_r).flatten())
    plt.show()


    
    long_dict = {'acc' : list(np.concatenate((acc_priv, acc_util, acc_util2))), 'eps' : list(np.concatenate((eps_l, eps_l, eps_l))), 'label' : value_n*eps_n*['priv']+value_n*eps_n*['util']+value_n*eps_n*['util2']}
    long_df = pd.DataFrame.from_dict(long_dict)

    plt.figure(figsize=(10, 7))

    sns.lineplot(x="eps", y="acc", hue='label', data=long_df)
    # sns.lineplot(x="eps", y="acc", hue='label', style='filter', data=long_df, legend=False)
    # sns.lineplot(x="eps", y="MI", data=long_df, legend=False)

    # plt.legend(["Priv (g-filter)", "Priv (gaussian-noise)", "Util (g-filter)", "Util (gaussian-noise)"], fontsize=20, loc=3)
    plt.legend(["Priv (g-filter)"], fontsize=20, loc=3)
    plt.tick_params(labelsize=20)
    plt.ylabel("accuracy", fontsize=22)
    # plt.ylabel("MI", fontsize=22)
    plt.xlabel(r"$b$", fontsize=22)
    plt.tight_layout()
    plt.savefig("fig/acc_mnist_plot_%s.png" % time.strftime("%Y-%m-%d_%H_%M_%S", time.gmtime()))

    plt.show()
    """