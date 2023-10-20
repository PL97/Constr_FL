import numpy as np
import matplotlib.pyplot as plt


if __name__ == "__main__":
    # dataset_name = "monks-1"
    # dataset_name = "adult"
    
    fig, axs = plt.subplots(nrows=1, ncols=3, sharey=False, figsize=(15, 4))
    ax2s = []
    n = 5
    for tmpi, dataset_name in enumerate(["breast-cancer-wisc", "adult", "monks-1"]):
        constr_val = 0.2
        data = np.load(f"new_files/obj_{dataset_name}_{n}.npy")
        consts = np.load(f"new_files/constr_{dataset_name}_{n}.npy")

        import matplotlib.pyplot as plt
         
        ax = axs[tmpi]
        
        x = np.arange(0, consts[:, 0].shape[0], 1)
        y_est = np.mean(consts, 1)
        y_err_top = np.max(consts, 1)
        y_err_low = np.min(consts, 1)
        ax.fill_between(x, y_err_top, y_err_low, alpha=0.2)
        ax.plot(x, y_est, label="local mean")
        # ax.plot(x, consts[:, -1], linestyle='--', label="global", color="tab:blue")
        
        
        # for i in range(num_infeas):
        #     # ax.plot(sorted_consts[:, i], label=f'client-{i+1}')
        #     ax.plot(, sorted_consts[:, i])
        if tmpi == 1:
            ax.set_xticks(np.arange(0, consts[:, 0].shape[0]+1, 1))
        elif tmpi == 0:
            ax.set_xticks(np.arange(0, consts[:, 0].shape[0]+1, 3))
        else:
            ax.set_xticks(np.arange(0, consts[:, 0].shape[0]+1, 5))
        
        
        # ax.plot([constr_val], '--', c='black', label='constraint (<=0.2)')
        # ax.plot(np.arange(1, consts[:, 0].shape[0]+1,), [constr_val]*consts.shape[0], '*', c='tab:blue', alpha=0.5)
        ax.axhline(constr_val, linestyle="--",c='tab:blue', alpha=0.8)
        ax.set_title(f"{dataset_name} (n={n})", fontsize=15)

           
        ax.spines['top'].set_visible(False)
        # ax.spines['right'].set_visible(False)
        ax.set_xlabel('iterations', fontsize=15)
        # ax.set_ylabel('iterations', color="tab:blue", fontsize=15)
        ax.tick_params(axis='y', labelcolor="tab:blue")
        
        
        
        
        ax2 = ax.twinx()
        
        y_est = np.mean(data, 1)
        y_err = np.max(data, 1) - y_est
        ax2.fill_between(x, y_est - y_err, y_est + y_err, alpha=0.2, color='brown')
        ax2.plot(x, y_est, color='brown', label="loss for class 1")
        
        ax2.tick_params(axis='y', labelcolor="brown")
        ax2.spines['top'].set_visible(False)
        
        ax.tick_params(axis='x')
        ax.set_xlabel('iterations', fontsize=15)
        ax2s.append(ax2)
        
        # ax.legend(loc="center right")
        # ax2.legend(loc="upper right")
        
        if tmpi == 0:
            ax.set_ylabel("loss for class 1", fontsize=15, color="tab:blue")        
        if tmpi == 2:
            ax2.set_ylabel('loss for class 0', color="brown", fontsize=15)
        #     ax2.get_yaxis().set_visible(False)
    # axs[0].get_shared_y_axes().join(*ax2s)
    fig.tight_layout()
    plt.savefig(f"new_files/{dataset_name}_constrs.png", bbox_inches = 'tight')