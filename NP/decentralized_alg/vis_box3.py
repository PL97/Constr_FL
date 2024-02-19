import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch


if __name__ == "__main__":
    # dataset_name = "monks-1"
    # dataset_name = "breast-cancer-wisc"
    for dataset_name in ['adult-a', 'monks-1', 'breast-cancer-wisc']:
        random_idx = 0
        workspace = "new_files"
        fig, axs = plt.subplots(nrows=1, ncols=1, sharey=False, figsize=(10, 5))
        ax2s = []
        f_data, u_data = [], []
        f_consts, u_consts = [], []
        for tmpi, n in enumerate([5, 10, 20]):
            # constr_val = 0.1
            # tmp_data = np.load(f"{workspace}/obj_{dataset_name}_{n}_{random_idx}.npy")
            # tmp_consts = np.load(f"{workspace}/constr_{dataset_name}_{n}_{random_idx}.npy")

            import json
            tmp_my_dict = open(f"{workspace}/{dataset_name}_repeat_10_{n}.json")
            tmp_my_dict = json.load(tmp_my_dict)
            tmp_data = tmp_my_dict[str(n)][str(0)]['u_obj']
            tmp_consts = tmp_my_dict[str(n)][str(0)]['u_constr']

            u_data.append(tmp_data)
            u_consts.append(tmp_consts)
            
            tmp_data = tmp_my_dict[str(n)][str(0)]['f_obj']
            tmp_consts = tmp_my_dict[str(n)][str(0)]['f_constr']
            

            f_data.append(tmp_data)
            f_consts.append(tmp_consts)
            
            

        import matplotlib.pyplot as plt
            
        labels = ["5", "10", "20"]
        ax = axs
        bplot1 = ax.boxplot(f_consts,
                vert=True,  # vertical box alignment
                patch_artist=True,  # fill with color
                showfliers=False,
                positions=np.array(np.arange(len(f_data)))*2.0+0.35)
        
        # for element in ['whiskers', 'fliers', 'means', 'medians', 'caps']:
        
        plt.setp(bplot1["means"], color="black")
        plt.setp(bplot1["caps"], color="black")
        plt.setp(bplot1["medians"], color="black")
        plt.setp(bplot1["fliers"], color="black")
        plt.setp(bplot1["whiskers"], color="black")
        # plt.setp(bplot1["boxes"], color="tab:blue")
        for patch in bplot1['boxes']:
            patch.set_facecolor("tab:blue")

        
        
        bplot1_1 = ax.boxplot(f_data,
                vert=True,  # vertical box alignment
                patch_artist=False,  # fill with color
                showfliers=False,
                positions=np.array(np.arange(len(f_data)))*2.0+0.35,
                labels = ["unconstrained loss 1"]*len(f_data))
        
        plt.setp(bplot1_1["means"], color="tab:blue")
        plt.setp(bplot1_1["caps"], color="tab:blue")
        plt.setp(bplot1_1["medians"], color="tab:blue")
        plt.setp(bplot1_1["fliers"], color="tab:blue")
        plt.setp(bplot1_1["whiskers"], color="tab:blue")
        plt.setp(bplot1_1["boxes"], color="tab:blue")
        
        
        bplot2_2 = ax.boxplot(u_consts,
                vert=True,  # vertical box alignment
                patch_artist=True,  # fill with color
                showfliers=False,
                positions=np.array(np.arange(len(f_data)))*2.0-0.35)
        
        plt.setp(bplot2_2["means"], color="black")
        plt.setp(bplot2_2["caps"], color="black")
        plt.setp(bplot2_2["medians"], color="black")
        plt.setp(bplot2_2["fliers"], color="black")
        plt.setp(bplot2_2["whiskers"], color="black")
        # plt.setp(bplot2_2["boxes"], color="brown")
        for patch in bplot2_2['boxes']:
            patch.set_facecolor("brown")
        
        
        
        
        bplot2 = ax.boxplot(u_data,
                    vert=True,  # vertical box alignment
                    patch_artist=False,  # fill with color
                    showfliers=False,
                    positions=np.array(np.arange(len(f_data)))*2.0-0.35)
        
        plt.setp(bplot2["means"], color="brown")
        plt.setp(bplot2["caps"], color="brown")
        plt.setp(bplot2["medians"], color="brown")
        plt.setp(bplot2["fliers"], color="brown")
        plt.setp(bplot2["whiskers"], color="brown")
        plt.setp(bplot2["boxes"], color="brown")
        
        
        # ax.legend([bplot1["boxes"][0], bplot1_1["boxes"][0], bplot2["boxes"][0], bplot2_2["boxes"][0]], ['A', 'B', 'c', 'd'], loc='best')
        legend_elements = [Patch(facecolor='brown', edgecolor='black', label='unconstrained model: class 1'),
                            Patch(facecolor='white', edgecolor='brown', label='unconstrained model: class 0'),
                            
                            Patch(facecolor='tab:blue', edgecolor='black', label='constrained model: class 1'),
                            Patch(facecolor='white', edgecolor='tab:blue', label='constrained model: class 0')
                            
                            ]

        # Create the figure
        ax.legend(handles=legend_elements, loc='best')
        # for line in leg1.get_lines():
        #     line.set_linewidth(6.0)
        ax.axhline(0.2, linestyle="--",c='black', alpha=0.8)


        
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        
        # ax.set_ylabel('iterations', color="tab:blue", fontsize=15)
        ax.tick_params(axis='y')
        ax.set_xlabel('number of clients', fontsize=15)
        
        
        # by iterating over all properties of the box plot
        def define_box_properties(plot_name, color_code, label, tmp_ax=None):
            # for k, v in plot_name.items():
                # plt.setp(plot_name.get(k), color=color_code)
            for patch in plot_name['boxes']:
                patch.set_facecolor(color_code)
                
            # use plot function to draw a small line to name the legend.
            tmp_ax.plot([], c=color_code, label=label)
            # tmp_ax.legend(loc="best")
            
        
        
        ax.set_xticks(np.arange(0, len(labels) * 2, 2), labels)
        ax.set_ylabel("loss", fontsize=15)
        fig.suptitle(f'{dataset_name}', fontsize=15)
        
        plt.savefig(f"{dataset_name}_box.png", bbox_inches = 'tight')