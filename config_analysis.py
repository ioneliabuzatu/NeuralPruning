import numpy as np
from copy import deepcopy
import torch
import pandas as pd
from terminaltables import AsciiTable

from models import *
from sensitivity_analysis import SENSITIVITY

from utils.utils import *
from utils.prune_utils import *


class opt():
  model_def = "/content/weight-pruning/config/yolov3-custom.cfg"
  data_config = "/content/weight-pruning/config/custom.data"
  model = '/content/weight-pruning/last_hand_checkpoint.pth'

# class opt():
#   model_def = "/Users/ioneliabuzatu/weight-pruning/config/yolov3-custom.cfg"
#   data_config = "/Users/ioneliabuzatu/weight-pruning/config/custom.data"
#   model = '/Users/ioneliabuzatu/PycharmProjects/all-prunings/yolov3.weights'


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_origin = Darknet(opt.model_def).to(device)

data_config = parse_data_config(opt.data_config)
testset_path = data_config["test"]
class_names = load_classes(data_config["names"])

try:
  model_origin.load_state_dict(torch.load(opt.model))
except:
  model_origin.load_darknet_weights(opt.model)

CBL_idx, Conv_idx, prune_idx= parse_module_defs(model_origin.module_defs)
pruning_percentiles_analysis = [5, 10, 20, 30, 40, 50, 60, 70, 80, 88]

final_pruning_percentiles = [20,88,30,88,88,5,88,10,88,88,10,88,20,88,88,88,88,88,30,88,30,88,50,88,30,88,
                             88,70,88,10,88,40,88,88,88,50,88,10,88,50,88,5,88,88,10,88,5,88,10,88,88,88,80,80,80,
                             70,70,88,88,50,50,50,50,50,88,88,40,50,50,40,40,30]

save_layerwise_accuracy = pd.DataFrame(np.nan, index=CBL_idx, columns=pruning_percentiles_analysis)