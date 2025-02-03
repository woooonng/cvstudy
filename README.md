# Applying ResNet and ViT to Low-Resolution Inputs
This study aims to apply ResNet and ViT to a low-Resolution dataset with a resolution of 32x32, consisting of 10 classes. A total of four models were used, including ResNet50, ResNet50 pretrained on ImageNet-1k, ViT-S/16 and ViT-S/16 pretrained on ImageNet-1k. In addition, ViT-S/3(42x42) which takes 42x42 inputs was analyzed.

----
## EDA
The data used in this study has a resolution of 32x32 and consists of 10 classes. The statistics of the train and test sets are presented below.
<div align="center">
  <img src="https://github.com/user-attachments/assets/d3b087ab-1ca5-4386-ab1c-edc58bcbfc4e" width="700">
</div><br>


The table shows the counts/ratio of each class in the train and test sets, as described in the graph above
| Model |class 0   |class 1   |class 2   |class 3   |class 4   |class 5   |class 6   |class 7 |class 8 |class 9 | Total |
| ------| -----    | ---------| -----    | ------   | -------- |--------  |--------  |--------|--------|--------| ------|
| Train | 5000/0.24| 3871/0.19| 2997/0.15| 2320/0.11| 1796/0.09| 1391/0.07| 1077/0.05|834/0.04|645/0.03|500/0.02| 20431 |
| Test  | 1000/0.1 | 1000/0.1 | 1000/0.1 | 1000/0.1 | 1000/0.1 | 1000/0.1 | 1000/0.1 |1000/0.1|1000/0.1|1000/0.1| 10000 |

----
## Models
All the models used in this study were based on the architectures described in the papers below.
- [Deep Residual Learning for Image Recognition](https://arxiv.org/abs/1512.03385)
- [An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale](https://arxiv.org/abs/2010.11929)

#### ResNet50 & ResNet50 pretrained on ImageNet-1k
To adapt to low-resolution dataset, the initial "conv1" in original model was replaced with a 2D convolutional layer that processes 3-channel inputs and outputs 64 channels in both models. The classifier was also modified to generate 10-dimension output. In the pyramid structure of the pretrained model, the feature maps from all layers were average pooled, concatenated, and then used as inputs to the classifier.

#### ViT-S/16 & ViT-S/16 pretrained on ImageNet-1k
The images were resized to 224x224 using bilinear interpolation to make the low-resolution data compatible with the both models. The output dimension of the classifier was set to 10, similar to the ResNet case. To examine the impact of the image size, the experiments were conducted with ViT-S/3(42x42) which takes the inputs with 42x42 resolution. The details are provided below.

| Model          | Patch size |Embedding dimension| Heads | Blocks | MLP dimension |
| -------------- | ---------- | ----------------- | ----- | ------ | ------------- |
| ViT-S/16       | 16x16      | 384               | 6     | 12     | 1536          |
| ViT-S/3(42x42) | 3x3        | 216               | 6     | 12     | 432           |
---
## Experiments
### Setting
The original train dataset was split into a train and validation set at a 9:1 ratio to evaluate generalization to the test set. All experiments were conducted using only the train and validation sets. After the experiments, the best model from each of the four categories was selected and tested on the test set. The criterion for choosing the best model was top-1 accuracy(%). Lastly, to examine the impact of class imbalance on generalization, models trained based on the F1 score were compared with those trained based on accuracy. The The learning rate schedules used for ResNet were multi-step lr, warm restarts, plateau and cosine decay, corresponding to MultiStepLR, ReduceLROnPlateau, CosineAnnealingWarmRestarts, CosineAnnealingLR in PyTorch. For ViT, only warm restarts and cosine decay were applied. Since the augmentations applied to ResNet and ViT differed despite having the same names, details are presented in the "Results" section. The additional hyperparameters in the experiments are provided as a table below.

| Category | Batch size |Optimizer| learning rate | Betas       | Epochs/Steps  |
| -------- | ---------- | --------| -----         | ------      | ------------- |
| ResNet   | 256        | AdamW   | 0.001         | (0.9, 0.999)| 200/14400     |
| ViT      | 128        | AdamW   | 0.0001        | (0.9, 0.999)| 100/14400     |


### Results
The "soft crop augmentation" technique was used here, and the related paper is listed below.
- [Soft Augmentation for Image Classification](https://arxiv.org/abs/2211.04625)

**1. Evaluation of ResNet50 & ResNet50 pretrained on ImageNet-1k**

The augmentations applied to ResNet models included None, Standard and Softcrop. **None** refers to simply normalize the images, while in **Standard** setting,. the images were first normalized, then padded, and randomly cropped to 32x32. Random horizontal and vertical flip was also applied. In **Softcrop**, normalization and random hoizontal flip were first performed, followed by soft crop augmentation, as proposed in the referenced paper. To put it simply, soft augmentation involved random cropping and label smoothing. Only the None type of augmentation was used for evaluation. In the experiments with ResNet50 pretrained on ImageNet-1k, fine-tuning, linear evaluation, and experiments on the pyramid structure were conducted.

ResNet50

| Augmentation          | Scheduler     | Top-1 Acc(%) |
| --------------------- | ------------- | ------------ |
| None                  | plateau       | 72.5         |
| None                  | multi-step lr | 74.4         |
| None                  | warm restarts | 78.3         |
| Standard              | warm restarts | 85.3         |
| Softcrop              | warm restarts | 91.9         |

ResNet50 pretrained on ImageNet-1k
| Type           | Augmentation          | Scheduler     |Pyramid| Top-1 Acc(%) |
| -------------- | --------------------- | ------------- |-------| ------------ |
|linear evaluaion| None                  | cosine decay  |x      | 73.9         |
|linear evaluaion| None                  | cosine decay  |x      | 77.9         |
|fine tuning     | Standard              | warm restarts |x      | 91.4         |
|fine tuning     | Standard              | cosine decay  |o      | 92.0         |
|fine tuning     | Softcrop              | cosine decay  |o      | 93.8         |
<br>

**2. Evaluaion of ViT-S/16 & ViT-S/16 pretrained on ImageNet-1k**

The augmentations applied to ResNet models included Standard, Standard_crop and Softcrop. **Standard** refers to resize the images to 224x224 and normalize them and **Standard_crop** augmentation resized the images to 300x300 and cropped them to 224x224. In **Softcrop**, the images were resized to 224x224 and soft crop augmentation was applied, just like in the ResNet case. In the case of ViT-S/3(16x16), the resized size was 42x42. Only the standard type of augmentation was used for evaluation. Only fine-tuning was performed in the experiments with the pretrained ViT-S/16.

ViT-S/16 & ViT-S/3(42x42)

|Model         | Augmentation  | Scheduler     | Top-1 Acc(%) |
|---           | --------------| ------------- | ------------ |
|ViT-S/16      | Standard      | cosine decay  | 66.6         |
|ViT-S/16      | Softcrop      | cosine decay  | 75.9         |
|ViT-S/3(42x42)| Standard      | cosine decay  | 59.8         |
|ViT-S/3(42x42)| Softcrop      | cosine decay  | 60.6         |

ViT-S/16 pretrained on ImageNet-1k
| Augmentation          | Scheduler     | Top-1 Acc(%) |
| --------------------- | ------------- | ------------ |
| Standard              | warm restarts | 95.1         |
| Standard_crop         | cosine decay  | 96.8         |
| Softcrop              | cosine decay  | 97.8         |
<br>

**3. Test of the best models**

Top-1 accuracy was calculated on the test set for the best models from each of the four categories. (p) refers to "pretrained" and -pyr refers to pyramid. None augmentation was applied for ResNet50, while Standard augmentation was used for ViT.

|Model           | Augmentation  | Scheduler     | Top-1 Acc(%) |
|---             | --------------| ------------- | ------------ |
|ResNet50        | Softcrop      | warm restarts | 88.6         |
|ResNet50(p)-pyr | Softcrop      | cosine decay  | 90.4         |
|ViT-S/16        | Softcrop      | cosine decay  | 66.7         |
|ViT-S/16(p)     | Softcrop      | cosine decay  | 96.5         |

**4.  Compare the accuracy-based models with the F1-score-based model**

Top-1 accuracy and F1-score were measured on the test set.

|Model           |Criterion      | F1-macro     | Top-1 Acc(%) |
|---             | -------------- | ------------- | ------------ |
|ResNet50(p)-pyr | top-1 acc      | 0.904         | 90.4         |
|ResNet50(p)-pyr | F1-macro       | 0.911         | 91.1         |
|ViT-S/16(p)     | top-1 acc      | 0.965         | 96.5         |
|ViT-S/16(p)     | F1-macro       | 0.958         | 95.8         |


## Conclusion
- Both warm start and cosine decay were effective in ResNet and ViT models, with no clear superiority between the two.
- Soft augmentation yielded the highest performance across all models, confirming the effectiveness of flexible cropping and label smoothing.
- For pretrained ResNet50, fine-tuning significantly outperformed other approaches, and the pyramid structure achieved the best performance, indicating that low-resolution inputs benefit classification.
- In ViT, interpolating small-resolution data to a higher resolution was effective.
- Despite class imbalance, when the number of samples in the minor class is sufficiently large (500 in this case), using accuracy as a metric does not significantly impact generalization.

Based on the results so far, ResNet50(p)-pyr based on the F1-score and ViT-S/16(p) based on accuracy can be considered the best models for ResNet and ViT, respectively.

---
## Appendix
Additional supplements are provided only for ResNet50(p)-pyr based on the F1-score and ViT-S/16(p) based on accuracy.

**1. Top-1 error rate per class**


<div align="center">
  <img src="https://github.com/user-attachments/assets/50810f1b-7d47-47fa-9f87-d50522a0d9a5" width="700">
</div><br>

**2. Confusion Matrix**

<div style="display: flex;">
    <img src="https://github.com/user-attachments/assets/f67432c8-c7eb-415f-8b74-d5860a2b8ea6" width="45%">
    <img src="https://github.com/user-attachments/assets/56c08545-cd79-42d2-81e5-5ce336e563c0" width="45%">
</div>



