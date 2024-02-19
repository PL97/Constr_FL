import numpy as np
import matplotlib.pyplot as plt


if __name__ == "__main__":
    # dataset_name = "monks-1"
    dataset_name = "adult"
    random_idx = 1
    workspace = "new_files_parallel_new"
    fig, axs = plt.subplots(nrows=1, ncols=3, sharey=True, figsize=(15, 4))
    ax2s = []
    for tmpi, n in enumerate([5, 10, 20]):
        constr_val = 0.1
        data = np.load(f"{workspace}/obj_{dataset_name}_{n}_{random_idx}.npy")
        consts = np.load(f"{workspace}/constr_{dataset_name}_{n}_{random_idx}.npy")
        print(consts.shape)
        print(consts[0, :])
        # data = data[1:, :]
        # consts = consts[1:, :]

        import matplotlib.pyplot as plt
         
        ax = axs[tmpi]
        # plt.rcParams.update({'font.size': 15})   
        # for i in range(consts.shape[1]):
        #     ax.plot(consts[:, i], label=f'client-{i+1}')
        # sorted_idx = np.argsort(consts[1, :])
        # num_infeas = np.sum(consts[1, :]>constr_val)
        # print(sorted_idx)
        # sorted_consts = np.zeros_like(consts)
        # for idx, i in enumerate(sorted_idx):
        #     sorted_consts[:, idx] = consts[:, i]
        # ax.set_yscale('log')
        x = np.arange(0, consts[:, 0].shape[0], 1)
        y_est = np.mean(consts[:, :-2], 1)
        y_err_top = np.max(consts[:, :-2], 1)
        y_err_low = np.min(consts[:, :-2], 1)
        ax.fill_between(x, y_err_top, y_err_low, alpha=0.2)
        ax.plot(x, y_est, label="local mean")
        ax.plot(x, consts[:, -1], label="global", linestyle="dashdot", color="tab:blue", alpha=1)
        
        ax.axhline(constr_val, linestyle='--', color='tab:blue', alpha=0.5)
        
        
        # for i in range(num_infeas):
        #     # ax.plot(sorted_consts[:, i], label=f'client-{i+1}')
        #     ax.plot(, sorted_consts[:, i])
        ax.set_xticks(np.arange(0, consts[:, 0].shape[0]+1, 1))
        
        
        # ax.plot([constr_val], '--', c='black', label='constraint (<=0.2)')
        # ax.plot(np.arange(1, sorted_consts[:, i].shape[0]+1, 1), [constr_val]*sorted_consts.shape[0], '--', c='black')
        ax.set_title(f"{dataset_name}-b (n={n})", fontsize=15)

           
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.set_xlabel('iterations', fontsize=15)
        # ax.set_ylabel('iterations', color="tab:blue", fontsize=15)
        ax.tick_params(axis='y', labelcolor="tab:blue")
        
        
        ax2 = ax.twinx()
        # ax2.set_yscale('log')
        ax2.set_ylabel('objective value', color="brown", fontsize=15)
        y_est = np.mean(data, 1)
        y_err = np.max(data, 1) - y_est
        ax2.fill_between(x, y_est - y_err, y_est + y_err, alpha=0.2, color='brown')
        ax2.plot(x, y_est, color='brown', label="objective value")
        ax2.tick_params(axis='y', labelcolor="brown")
        ax2.spines['top'].set_visible(False)
        
        ax.tick_params(axis='x')
        ax.set_xlabel('iterations', fontsize=15)
        ax2s.append(ax2)
        # ax2.set_ylim([0, 1000])
        
        ax.legend(loc="upper right")
        # ax2.legend(loc="upper right")
        
        if tmpi < 2:
            ax2.get_yaxis().set_visible(False)
    # axs[0].get_shared_y_axes().join(*ax2s)
    axs[0].set_ylabel("loss disparity", fontsize=15, color="tab:blue")
    plt.savefig(f"{workspace}/{dataset_name}_constrs.png", bbox_inches = 'tight')