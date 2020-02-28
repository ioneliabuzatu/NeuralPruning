from models import *
from utils.utils import *
import torch
import numpy as np
from copy import deepcopy
from test import evaluate
from terminaltables import AsciiTable
import time
from utils.prune_utils import *


class opt():
    model_def = "/content/weight-pruning/config/yolov3-custom.cfg"
    data_config = "/content/weight-pruning/config/custom.data"
    model = '/content/weight-pruning/yolov3.weights'


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = Darknet(opt.model_def).to(device)

try:
    model.load_state_dict(torch.load(opt.model))
except:
    model.load_darknet_weights(opt.model)

data_config = parse_data_config(opt.data_config)
valid_path = data_config["test"]
class_names = load_classes(data_config["names"])

eval_model = lambda model: evaluate(model, path=valid_path, iou_thres=0.0005, conf_thres=0.01,
                                    nms_thres=0.0005, img_size=416, batch_size=1)
obtain_num_parameters = lambda model: sum([param.nelement() for param in model.parameters()])

origin_model_metric = eval_model(model)
origin_nparameters = obtain_num_parameters(model)

CBL_idx, Conv_idx, prune_idx = parse_module_defs(model.module_defs)


def weight_prune_layerwise(model, CBLs, pruning_layers, which_layer_id=None, threshold=None):
    all_masks = []
    num_filters = []
    thresholds_layerwise = [88 for i in range(len(CBL_idx))]
    # thresholds_layerwise[prune_single_layer_id] = pruning_percentile

    for layer, pruning_id in zip(CBL_idx, thresholds_layerwise):
        module = model.module_list[layer][1]
        all_layer_wise_weights = []

        # magnitude based threshold layer wise
        if layer in pruning_layers:
            for param in module.parameters():
                all_layer_wise_weights += list(param.cpu().data.abs().numpy().flatten())
            # threshold = np.percentile(np.array(all_layer_wise_weights), pruning_threshold)
            threshold = np.percentile(np.array(all_layer_wise_weights), pruning_id)

            # generate mask
            mask = module.weight.data.abs().ge(threshold).float()
            remain = int(mask.sum())
            mask = np.array(mask.cpu())

        else:  # skip shortcuts
            mask = np.ones(module.weight.data.shape)
            remain = mask.shape[0]

        all_masks.append(mask.copy())
        num_filters.append(remain)

    return num_filters, all_masks


num_filters, filters_mask = weight_prune_layerwise(model, CBL_idx, prune_idx)

CBLidx2mask = {idx: mask for idx, mask in zip(CBL_idx, filters_mask)}

pruned_model = prune_model_keep_size(model, prune_idx, CBL_idx, CBLidx2mask)

eval_model(pruned_model)

# %%
compact_module_defs = deepcopy(model.module_defs)
for idx, num in zip(CBL_idx, num_filters):
    assert compact_module_defs[idx]['type'] == 'convolutional'
    compact_module_defs[idx]['filters'] = str(num)

# %%
compact_model = Darknet([model.hyperparams.copy()] + compact_module_defs).to(device)
compact_nparameters = obtain_num_parameters(compact_model)

init_weights_from_loose_model(compact_model, pruned_model, CBL_idx, Conv_idx, CBLidx2mask)

# %%
random_input = torch.rand((1, 3, model.img_size, model.img_size)).to(device)


def obtain_avg_forward_time(input, model, repeat=200):
    model.eval()
    start = time.time()
    with torch.no_grad():
        for i in range(repeat):
            output = model(input)
    avg_infer_time = (time.time() - start) / repeat

    return avg_infer_time, output


pruned_forward_time, pruned_output = obtain_avg_forward_time(random_input, pruned_model)
compact_forward_time, compact_output = obtain_avg_forward_time(random_input, compact_model)

diff = (pruned_output - compact_output).abs().gt(0.001).sum().item()
if diff > 0:
    print('Something wrong with the pruned model!')

compact_model_metric = eval_model(compact_model)

metric_table = [
    ["Metric", "Before", "After"],
    ["mAP", f'{origin_model_metric[2].mean():.6f}', f'{compact_model_metric[2].mean():.6f}'],
    ["Parameters", f"{origin_nparameters}", f"{compact_nparameters}"],
    ["Inference", f'{pruned_forward_time:.4f}', f'{compact_forward_time:.4f}']
]
print(AsciiTable(metric_table).table)

pruned_cfg_name = "/content/weight-pruning/config/pruned.cfg"
pruned_cfg_file = write_cfg(pruned_cfg_name, [model.hyperparams.copy()] + compact_module_defs)
print(f'Config file has been saved: {pruned_cfg_file}')
#
compact_model_name = "/content/weight-pruning/pruned.pth"
torch.save(compact_model.state_dict(), compact_model_name)
print(f'Compact model has been saved: {compact_model_name}')
