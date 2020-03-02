from utils.flops_and_params import get_model_complexity_info
from models import *
from utils.prune_utils import *

if __name__ == '__main__':
    model = '/Users/ioneliabuzatu/PycharmProjects/all-prunings/yolov3.weights'
    model_def = "/Users/ioneliabuzatu/weight-pruning/config/yolov3-custom.cfg"
    #

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_origin = Darknet(model_def).to(device)

    try:
        model_origin.load_state_dict(torch.load(model))
    except:
        model_origin.load_darknet_weights(model)

    # flops, params = get_model_complexity_info(model_origin, (3, 416,416), as_strings=True, print_per_layer_stat=True)
    output_origianl = get_model_complexity_info(model_origin, (3, 416, 416), as_strings=True, print_per_layer_stat=True)

    for original in output_origianl:
        out = f"{original.split('|')[0]} | {original.split('|')[1]}({original.split('|')[2]})"
        print(out)
