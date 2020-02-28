from config_analysis import *


for layer_i, layer in enumerate(CBL_idx):
  for prune_percentile in pruning_percentiles:
    print(f"Analysis percentile: {prune_percentile} on layer: {layer}")
    sensitivity_start = SENSITIVITY(model, CBL_idx, Conv_idx, prune_idx, testset_path, class_names, prune_percentile, layer_i)
    sensitivity_start.weight_prune_layerwise()
    sensitivity_start.build_pruned_model()
    print("Done pruning the model")

    eval_model = sensitivity_start.evaluate_model_(model)
    eval_pruned = sensitivity_start.evaluate_model_(sensitivity_start.pruned_final)

    origin_nparams = sensitivity_start.get_tot_parameters_model(model)
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
    save_layerwise_accuracy.loc[layer, prune_percentile] = f"{precision_pruned}|{recall_pruned}|{mAP_pruned}|{f1_pruned}"

    # print(df)

print("Saving analysis to pickle file")
# save_layerwise_accuracy.to_pickle("/home/ionelia/final-pruning/analysis.pkl")
save_layerwise_accuracy.to_pickle("/content/weights-pruning/analysis.pkl")
