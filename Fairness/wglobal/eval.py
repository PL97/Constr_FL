
import json
from prettytable import PrettyTable
import numpy as np


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)
    
# adult, #["breast-cancer-wisc", "monks-1", "adult"]
dataset = "adult"
workspace = "new_files_parallel_new"

x = PrettyTable()
x.field_names = ["n", "obj_ours", "obj_cen", "diff", "feas(ours) mean", "feas(ours) max", "feas(cen) mean", "feas(cen) max"]


from collections import defaultdict

for n in [1, 5, 10, 20]:
    my_dict = defaultdict(lambda: {})
    for xx in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]:
        tmp_my_dict = open(f"{workspace}/{dataset}_repeat_{n}_{xx}.json")
        tmp_my_dict = json.load(tmp_my_dict)
        my_dict[str(n)][str(xx)] = tmp_my_dict[str(n)][str(xx)]
        


    tmp_federa_obj, tmp_central_obj, tmp_unconstr_obj, tmp_central_cnstr, tmp_federa_cnstr, tmp_unconstr_cnstr = [], [], [], [], [], []
    fed_obj_mean, fed_obj_max, fed_constr_mean, fed_constr_max, cen_obj_mean, cen_obj_max, cen_constr_mean, cen_constr_max = [], [], [], [], [], [], [], []
    for k, v in my_dict[str(n)].items():
        tmp_central_obj.extend(v['c_obj'])
        tmp_federa_obj.extend(v['f_obj'])
        tmp_unconstr_obj.extend(v['u_obj'])
        
        tmp_central_cnstr.extend(v['c_constr'])
        tmp_federa_cnstr.extend(v['f_constr'])
        tmp_unconstr_cnstr.extend(v['u_constr'])
        
        ## stats mean and max
        fed_obj_mean.append(np.mean(v['f_obj']))
        fed_obj_max.append(np.max(v['f_obj']))
        cen_obj_mean.append(np.mean(v['c_obj']))
        cen_obj_max.append(np.max(v['c_obj']))
    
        fed_constr_mean.append(np.mean(v['f_constr']))
        fed_constr_max.append(np.max(v['f_constr']))
        cen_constr_mean.append(np.mean(v['c_constr']))
        cen_constr_max.append(np.max(v['c_constr']))
    
    diff = [abs(x-y)/abs(x) for x, y in zip(tmp_central_obj, tmp_federa_obj)]
    print(tmp_federa_obj)
        # x.field_names = ["n", "obj_ours", "obj_cen", "obj_uncnstr", "feas(ours) mean", "feas(ours) max", "feas(cen) mean", "feas(cen) max", "feas(uncnstr) mean", "feas(uncnstr) max"]
    x.add_row([n, "{:.2f}({:.2e})".format(np.mean(tmp_federa_obj), np.std(tmp_federa_obj)), \
                    "{:.2f}({:.2e})".format(np.mean(tmp_central_obj), np.std(tmp_central_obj)), \
                    "{:.2e}({:.2e})".format(np.mean(diff), np.std(diff)), \
                    # "{:.2f}({:.2e})".format(np.mean(tmp_federa_cnstr), np.std(tmp_federa_cnstr)), \
                    # "{:.2f}".format(np.amax(tmp_federa_cnstr)), \
                    # "{:.2f}({:.2e})".format(np.mean(tmp_central_cnstr), np.std(tmp_central_cnstr)), \
                    # "{:.2f}".format(np.amax(tmp_central_cnstr))])
                    
                    "{:.2f}({:.2e})".format(np.mean(fed_constr_mean), np.std(fed_constr_mean)), \
                    "{:.2f}({:.2e})".format(np.mean(fed_constr_max), np.std(fed_constr_max)), \
                    "{:.2f}({:.2e})".format(np.mean(cen_constr_mean), np.std(cen_constr_mean)), \
                    "{:.2f}({:.2e})".format(np.mean(cen_constr_max), np.std(cen_constr_max))])
print(x)