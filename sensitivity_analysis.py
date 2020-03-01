from config_analysis import *
from utils import prune_utils
from test import evaluate



class SENSITIVITY:
    def __init__(self, model, batch_norm_idx, conv_idx, to_prune_idx, testset_path, pruning_percentile, prune_single_layer):
        self.model = model
        self.all_masks = []
        self.num_filters = []
        self.batch_norm_idx = batch_norm_idx
        self.conv_idx = conv_idx
        self.to_prune_idx = to_prune_idx

        self.pruning_percentile = pruning_percentile
        self.prune_single_layer_id = prune_single_layer
        self.testset_path = testset_path

        self.modules2masks = None
        self.pruned_tmp = None
        self.pruned_modules_modules = None
        self.pruned_final = None

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def weight_prune_layerwise(self):  # , model, layers_ids, pruning_layers):
        if self.prune_single_layer_id != None:
            thresholds_layerwise = [0 for i in range(len(self.batch_norm_idx))]
            thresholds_layerwise[self.prune_single_layer_id] = self.pruning_percentile
        else:
            thresholds_layerwise = self.pruning_percentile
            assert isinstance(self.pruning_percentile, list)

        for layer, pruning_id in zip(self.batch_norm_idx, thresholds_layerwise):
            module = self.model.module_list[layer][1]
            all_layer_wise_weights = []

            # magnitude based threshold layer wise
            if layer in self.to_prune_idx:
                for param in module.parameters():
                    all_layer_wise_weights += list(param.cpu().data.abs().numpy().flatten())
                threshold = np.percentile(np.array(all_layer_wise_weights), pruning_id)

                # generate mask
                mask = module.weight.data.abs().ge(threshold).float()
                remain = int(mask.sum())
                mask = np.array(mask.cpu())

            else:  # skip shortcuts
                mask = np.ones(module.weight.data.shape)
                remain = mask.shape[0]

            self.all_masks.append(mask.copy())
            self.num_filters.append(remain)

    def build_pruned_model(self):

        self.modules2masks = {idx: mask for idx, mask in zip(self.batch_norm_idx, self.all_masks)}
        self.pruned_tmp = prune_utils.prune_model_keep_size(self.model, self.to_prune_idx, self.batch_norm_idx,
                                                            self.modules2masks)
        self.pruned_modules_modules = deepcopy(self.model.module_defs)

        # for making the config model file
        for id_module, new_filters in zip(self.batch_norm_idx, self.num_filters):
            assert self.pruned_modules_modules[id_module]['type'] == 'convolutional'
            self.pruned_modules_modules[id_module]['filters'] = str(new_filters)

        self.pruned_final = Darknet([self.model.hyperparams.copy()] + self.pruned_modules_modules).to(self.device)
        prune_utils.init_weights_from_loose_model(self.pruned_final,self.pruned_tmp, self.batch_norm_idx,
                                                  self.conv_idx,
                                                  self.modules2masks)

    def evaluate_model_(self, model):
        eval = evaluate(model, path=self.testset_path, iou_thres=0.5, conf_thres=0.001, nms_thres=0.5, img_size=416,
                        batch_size=1)
        return eval

    def get_tot_parameters_model(self, model):
        tot_params = sum([param.nelement() for param in model.parameters()])
        return tot_params

    def get_params_by_layer(self):
        params_table = [["Layer num", "Original params", "Pruned params"]]
        for layer_id, layer in enumerate(self.pruned_final.module_list):
            if isinstance(layer[0], torch.nn.Conv2d):
                tot_original_params = self.get_tot_parameters_model(self.model.module_list[layer_id])
                tot_pruned_params = self.get_tot_parameters_model(self.pruned_final.module_list[layer_id])

                params_table.append([f"{layer_id}", f"{tot_original_params}", f"{tot_pruned_params}"])
        # print(AsciiTable(params_table).table)
        return AsciiTable(params_table).table

    def save_pruned_model(self):
        name_cfg = "/content/weight-pruning/config/pruned.cfg"
        name_weights = "/content/weight-pruning/pruned.pth"
        assert name_cfg and name_cfg
        prune_utils.write_cfg(name_cfg, [self.model.hyperparams.copy()] + self.pruned_modules_modules)
        torch.save(self.pruned_tmp.state_dict(), name_weights)
        print("Config and Weights of the model have been saved")
