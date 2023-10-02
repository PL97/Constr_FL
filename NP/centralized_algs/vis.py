import numpy as np
import matplotlib.pyplot as plt


if __name__ == "__main__":
    dataset_name = "monks-1"
    n = 20
    constr_val = 0.2
    data = np.load(f"files/obj_{dataset_name}_{n}.npy")
    consts = np.load(f"files/constr_{dataset_name}_{n}.npy")
    print(data.shape, consts.shape)
    consts[consts<constr_val] = constr_val
    plt.plot(np.log(data))
    plt.savefig("obj.png")
    plt.close()
    fig, ax = plt.subplots()
    # for i in range(consts.shape[1]):
    #     ax.plot(consts[:, i], label=f'client-{i+1}')
    sorted_idx = np.argsort(consts[0, :])[::-1]
    num_infeas = np.sum(consts[0, :]>constr_val)
    print(sorted_idx)
    sorted_consts = np.zeros_like(consts)
    for idx, i in enumerate(sorted_idx):
        sorted_consts[:, idx] = consts[:, i]
    for i in range(num_infeas):
        # ax.plot(sorted_consts[:, i], label=f'client-{i+1}')
        ax.plot(sorted_consts[:, i])
    
    # ax.plot([constr_val], '--', c='black', label='constraint (<=0.2)')
    ax.plot([constr_val]*sorted_consts.shape[0], '--', c='black')
    ax.set_title(f"{dataset_name}")
    # ax.plot(np.max(consts, 1), label="max")


    # ax2 = ax.twinx()
    # ax.plot(data, label="obj", color='r')
    
    
    
    
    
    # ax.plot(np.mean(consts, 1), label="mean")
    # ax.plot([constr_val]*(consts.shape[0]), '--', c='black', label='constraint (<=0.2)')
    
    # ax.legend()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_xlabel("iterations")
    ax.set_ylabel("feasibility violation")
    plt.savefig("constrs.png")