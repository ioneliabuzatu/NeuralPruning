# Goal
1. Sensitivity Analysis layer-wise
2. Estimating computational weight per layer (FLOPS)
3. Estimating thresholds per layer based on the above
4. Running the pruning process

model architecture based on [PyTorch-YOLOv3](https://github.com/eriklindernoren/PyTorch-YOLOv3)


# What I did:

Trained the model with the pretrained yolov3.weights on the [hand dataset](link here) for 10 epochs
Run the sensitivity analysis per layer with the following pruning percentiles: [5, 10, 20, 30, 40, 50, 60, 70, 80, 88]
(above 88 yields unresolved bugs)

Final pruning with the best pick in the plots for mAP for each of the above percentiles

Fine tuned the pruned model for tot epochs


# Figures 
plots thresholds...

table accuracy, params and flops

accuracy 3 steps: 
    1) original weights trained on the hand dataset 
    2) pruned before finetuned
    3) pruned after finetuned


flops original VS pruned



after pretrained on the hand dataset
total loss: 0.78 -> last_hand_checkpoint.pth

test on 103 images:
Detecting objects: 100% 103/103 [00:24<00:00,  4.67it/s]
Computing AP: 100% 1/1 [00:00<00:00, 30.91it/s]
Average Precisions:
+ Class '0' (hand) - AP: 0.7141667580393108
mAP: 0.7141667580393108 