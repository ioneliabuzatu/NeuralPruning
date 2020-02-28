1. Sensitivity Analysis layer-wise
2. Estimating computational weight per layer (FLOPS)
3. Estimating thresholds per layer based on the above
4. Running the pruning process


plots thresholds...

table accuracy, params and flops

accuracy: 3 steps: 
    1) original weights trained on the hand dataset 
    2) pruned before finetuned
    3) pruned after finetuned

model architecture based on [PyTorch-YOLOv3](https://github.com/eriklindernoren/PyTorch-YOLOv3)