from config_analysis import *


def main(conv_bn_layers, pruning_percentiles, conv_only, to_prune_layers):
    print(f"Generating the pruned model with the percentiles: {pruning_percentiles}")
    sensitivity_start = SENSITIVITY(model=model_origin, conv_bn_layers=conv_bn_layers, Conv_idx=conv_only,
                                    prune_idx=to_prune_layers, testset_path=testset_path,
                                    pruning_percentile=final_pruning_percentiles, prune_single_layer=None)
    sensitivity_start.weight_prune_layerwise()
    sensitivity_start.build_pruned_model()
    print("Done pruning the model")

    eval_model = sensitivity_start.evaluate_model_(model_origin)
    eval_pruned = sensitivity_start.evaluate_model_(sensitivity_start.pruned_final)

    origin_nparams = sensitivity_start.get_tot_parameters_model(model_origin)
    pruned_nparams = sensitivity_start.get_tot_parameters_model(sensitivity_start.pruned_final)

    mAP_origin = f'{eval_model[2].mean():.6f}'
    precision_origin = f'{eval_model[0].mean():.6f}'
    recall_origin = f'{eval_model[1].mean():.6f}'
    f1_origin = f'{eval_model[3].mean():.6f}'

    mAP_pruned = f'{eval_pruned[2].mean():.6f}'
    precision_pruned = f'{eval_pruned[0].mean():.6f}'
    recall_pruned = f'{eval_pruned[1].mean():.6f}'
    f1_pruned = f'{eval_pruned[3].mean():.6f}'

    metric_table = [
        ["Metric", "Before", "After"],
        ["mAP", mAP_origin, mAP_pruned],
        ["Precision", precision_origin, precision_pruned],
        ["Recall", recall_origin, recall_pruned],
        ["F1", f1_origin, f1_pruned],
        ["Parameters", f"{origin_nparams}", f"{pruned_nparams}"]
    ]

    print(AsciiTable(metric_table).table)

    # pruned_model_accuracy = [f"{precision_pruned}|{recall_pruned}|{mAP_pruned}|{f1_pruned}"]
    # print(f"Pruned model accuracy:\n{pruned_model_accuracy}")


main(conv_bn_layers=CBL_idx, pruning_percentiles=pruning_percentiles_analysis)
