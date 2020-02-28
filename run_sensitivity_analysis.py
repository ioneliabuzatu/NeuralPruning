from config_analysis import *


def main(conv_bn_layers, pruning_percentiles, conv_only, to_prune_layers):
    for layer_i, layer in enumerate(conv_bn_layers):
      for prune_percentile in pruning_percentiles:
        print(f"Analysis percentile: {prune_percentile} on layer: {layer}")
        sensitivity_start = SENSITIVITY(model=model_origin,  batch_norm_idx=conv_bn_layers, conv_idx=conv_only, to_prune_idx=to_prune_layers, testset_path=testset_path, pruning_percentile=prune_percentile, prune_single_layer=layer_i)
        sensitivity_start.weight_prune_layerwise()
        sensitivity_start.build_pruned_model()

        # eval_model = sensitivity_start.evaluate_model_(model_origin)
        eval_pruned = sensitivity_start.evaluate_model_(sensitivity_start.pruned_final)

        # origin_nparams = sensitivity_start.get_tot_parameters_model(model_origin)
        pruned_nparams = sensitivity_start.get_tot_parameters_model(sensitivity_start.pruned_final)

        mAP_pruned = f'{eval_pruned[2].mean():.6f}'
        precision_pruned = f'{eval_pruned[0].mean():.6f}'
        recall_pruned = f'{eval_pruned[1].mean():.6f}'
        f1_pruned = f'{eval_pruned[3].mean():.6f}'

        save_layerwise_accuracy.loc[layer, prune_percentile] = f"{precision_pruned}|{recall_pruned}|{mAP_pruned}|{f1_pruned}"

      print(f"Done pruning layer {layer}")

    print("Saving analysis to pickle file")
    save_layerwise_accuracy.to_pickle("/content/weights-pruning/analysis.pkl")


main(conv_bn_layers=CBL_idx, pruning_percentiles=pruning_percentiles_analysis, conv_only=Conv_idx, to_prune_layers=prune_idx)
