# blood-cell-detection

## Method/Approach

### Model Selection

For this problem, I used ResNet50 pretrained with ImageNet. ResNet50 is a middle ground architecture, the model is deep enough to learn complex features (50 layers) but not too deep so no need to worry about overwhelming computation like ResNet-101 or ResNet-152. Also, it achieves a good balance between accuracy and efficiency, making it suitable for a wide range of tasks. Thus, I chose ResNet50 over other pretrained models.

### Data Preprocessing

The biggest challenge I had was that this was a multi-label classification problem. Previously, all of the assignments we got in class were to classify an image that represented one label. (An image is digit 0, an image is shoes, one image represents one label.) But for this question, in one image, we can have more than one label associated with it. (For example, one image can only have several RBCs, but another image can have RBCs, WBC, and Platelets, etc.) 

To address this issue, instead of having `[0, 1, 2]` as a label (which is used for multi-class classification), we used a one-hot encoded vector of size three. If an image has RBCs only, then the label becomes `[1, 0, 0]`, and if an image has all the blood cells, then it will become `[1, 1, 1]`.

### Model Architecture (Activation, Loss)

For the model architecture, I replaced the final classification layer. Sigmoid activation allows the model to independently predict probabilities for each label, enabling multiple-label outputs. Thus, instead of using a single output for each class with softmax, I used a separate sigmoid activation for each class independently. The sigmoid outputs a value between 0 and 1. A value of 1 means the class is present in the image, and 0 means the class is absent in the image.

For the loss function, I used binary cross-entropy since we are dealing with multiple labels per image, and the usage of sigmoid treats each class as a separate binary prediction task. So the model is evaluating whether an image has RBCs or not, has WBCs or not, and has Platelets or not. The final output is a vector of size 3, where each element indicates the presence (1) or absence (0) of the class.

### Metrics

For performance metrics, I calculated the precision, recall, and F1-score for each class (RBC, WBC, and Platelet) and the Hamming Loss. We opted for F1-score and Hamming Loss over accuracy due to the unique challenges of multi-label classification.

Accuracy is commonly used for single-label classification tasks, where each sample belongs to exactly one class. However, for multi-label problems, accuracy often requires a perfect match across all labels in a sample. This strict criterion fails to capture partial correctness when only some of the labels are predicted correctly. For example, if a model correctly predicts one label (e.g., WBC) but misses others (e.g., Platelets), the entire sample is considered incorrect in terms of accuracy, which can be misleading.

In contrast, Hamming Loss is better suited for multi-label tasks. It calculates the fraction of labels that are incorrectly predicted, averaging errors across all samples. This metric is more flexible, as it measures the model’s performance on individual labels rather than requiring an all-or-nothing match. For example, if the model incorrectly predicts a class (e.g., RBC) when it’s not present, Hamming Loss captures this error without invalidating the entire sample.

Moreover, Hamming Loss provides a granular view of the model’s performance by penalizing errors equally across all labels. This is particularly valuable in cases of class imbalance. In multi-label classification, some classes (e.g., Platelets) may be less frequent than others (e.g., RBC). Hamming Loss does not favor frequent classes, as it treats errors for rare and frequent labels equally.

On the other hand, metrics like precision, recall, and F1-score are sensitive to class imbalance. They may prioritize performance on frequent classes while under-representing rare ones. Since our dataset (images) are more likely to have blood cells, it will likely have RBCs, WBCs, and Platelets in images. Thus, it is possible that the model tends to predict that an image has Platelets even when there are no platelets (highly imbalanced & possibility of overfitting), and to check this situation, F1 score is needed.

## Experiment

Since there are so many plots for this section, all of the visuals have been placed in the Supplement section.

Before my experiment, I was curious if using dropout might help the model generalize better, leading to better performance. So, I used dropout with a 50% dropout rate. With that, I got:

- 96.45% F1 score for RBC
- 99.31% F1 score for WBC
- 70.59% F1 score for Platelets
- Hamming loss: 0.1416
- Confusion matrix: ![Confusion Matrix 1]([path_to_cm_1_image](https://github.com/saeahgo/blood-cell-detection/blob/main/cm_b.png))

We observe that the model classifies RBC and WBC pretty well, but struggles a bit with classifying Platelet.

For the next stage, I still used dropout, but reduced the dropout percentage from 50% to 10%. With that, I got:

- 95.71% F1 score for RBC
- 99.31% F1 score for WBC
- 75.56% F1 score for Platelets
- Hamming loss: 0.1324
- Confusion matrix: ![Confusion Matrix 2](path_to_cm_2_image)

Even though the F1 score for Platelet has some improvements (~5% increase), it doesn't show a huge improvement in terms of the confusion matrix. So, I searched why and found out that when the dataset is relatively small or simple, dropout might unnecessarily reduce the capacity of the model during training. Since our dataset only includes ~400 images, it is small, and I decided to remove the dropout layer.

So, I removed the dropout layer and ran the model again. With that, I got:

- 95.71% F1 score for RBC
- 99.31% F1 score for WBC
- 82% F1 score for Platelets
- Hamming loss: 0.0959
- Confusion matrix: ![Confusion Matrix 3](path_to_cm_3_image)

Without the dropout layer, the model improves significantly. Not only did the F1 score for Platelets improve, but also the confusion matrix. There were 18, 19 False Positive (FP) cases for Platelets (even though images didn't have Platelets, the model predicted we did). But now, without the dropout layer, the number of FP cases reduced to 9. Still some, but much better results.

I wanted to check if the 0.5 threshold for output (if the output is greater than or equal to 0.5, classify as 1, otherwise classify as 0) is ideal or if other numbers are better. So, I changed the threshold to 0.7 (70%). With that, I got:

- 94.89% F1 score for RBC
- 99.31% F1 score for WBC
- 71.05% F1 score for Platelet
- Hamming loss: 0.1370
- Confusion matrix: ![Confusion Matrix 4](path_to_cm_4_image)

Compared to the 0.5 threshold, the 0.7 threshold did not improve the model's performance.

I also tried a 0.3 threshold and got:

- 97.14% F1 score for RBC
- 99.31% F1 score for WBC
- 73.56% F1 score for Platelets
- Hamming loss: 0.1279
- Confusion matrix: ![Confusion Matrix 5](path_to_cm_5_image)

The 0.3 threshold did not show improvement compared to the 0.5 threshold, so the best model parameters are when using the 0.5 threshold for output.

## Conclusion

The model performs best when we only train the last layer with two outputs (without dropout layer), and use a 50% output threshold. I believe the model cannot classify Platelets correctly compared to RBCs and WBCs because there are mostly one or two Platelets in an image, and sometimes they are not visible, so we don't have a lot of samples to train. Thus, its F1 score is relatively lower than the others. 

But overall, the model still classifies each blood cell strongly with a 0.0959 Hamming loss. Below are the confusion matrix results and a comparison of predicted/true labels with the best parameter set (No dropout layer, 50% output threshold).
