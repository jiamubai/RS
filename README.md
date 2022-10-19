# RS
Representer Sketch inference and application

### Result:

Pretrained Resnet50 + MLP achieves about 86% accuracy
Pretrained Resnet50 + RS achieves 43% accuracy on test set (overfitting)

For RS, tried both "linear" and "avg" aggregation, linear yields very high train acc (97%) while avg yields low train acc (around 60%), but both have low test acc
To address overfitting, following techniques were also implemented:

1. Adding weight decay: weight decay somehow degrade Representer Sketch performance 
2. Reduce model size: original RS model size is R8L4000, downsized model size is R7L800, both yield similar patterns (serious overfit)
3. Dropping out weights: overfitting still exists (see eatly_stop_result.txt)
