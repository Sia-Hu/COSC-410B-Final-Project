# Intelligent Image-Based Waste Sorting System for Sustainable Resource Management
Code and dataset for Sia and Christine's final project for Spring 2025: COSC 410B Applied Machine Learning.
# Dataset
We used TrashNet collected by Gary Thung and Mindy Yang.

**Total Images:** 2,527

**Minimal Example.ipynb:**
This simple CNN architecture consisted of three convolutional layers, each followed by a max pooling layer. Starting with 32 filters in the first layer, the network increased the filter count to 64 and 128 in subsequent layers, using the ReLU activation function to introduce non-linearity. After the convolutional blocks, a flatten layer converted the feature maps into a one-dimensional vector, which was then passed through a fully connected dense layer. A dropout layer with a rate of 0.5 was included to mitigate overfitting by randomly deactivating some neurons during training. Finally, the output layer employed a softmax activation function with as many neurons as there are waste categories, enabling multi-class classification. The model was trained using the Adam optimizer with categorical cross-entropy as the loss function, and the training process ran for 30 epochs. While the training accuracy reached approximately 80%, the validation accuracy was only around 35% with a significantly higher validation loss. This discrepancy between training and validation performance suggests that the model was overfitting to the training data. Thus, we would implement data augmentation that could have helped improve generalization. 

**Mininal_Working_Example_(with_data_augmentation).ipynb:** 
This CNN architecture consists of three convolutional layers with 32, 64, and 128 filters, each followed by max pooling and ReLU activation, before flattening into a dense layer with dropout regularization (0.5 rate) to combat overfitting. The model was trained for 30 epochs using the Adam optimizer and categorical cross-entropy loss, with slight data augmentation (including rotations, shifts, zooms, and brightness adjustments) applied to the training images (resized to 128×128 pixels). The model achieved 68% training accuracy and 41% validation accuracy, indicating some overfitting despite the augmentation and dropout, although the performance gap has slightly improved compared to the previous model without augmentation. The performance gap suggests opportunities for improvement through more advanced architectures, additional regularization techniques, or dataset expansion. The implementation includes standard evaluation metrics, generating both a classification report and confusion matrix to analyze model performance across different trash categories, with the training set (2,021 images) and test set (506 images) carefully split to maintain class balance.

**ResNet.py:** 
This file implemented a Residual Network (ResNet) architecture, leveraging skip connections to allow gradients to flow more effectively through deep layers and mitigate the vanishing gradient problem. The architecture consisted of stacked residual blocks, each containing convolutional layers followed by batch normalization and ReLU activations, with shortcut connections bypassing one or more layers. The model began with an initial convolutional layer, followed by a series of residual blocks that progressively increased the number of filters. After the residual stack, a global average pooling layer reduced the feature maps before feeding them into a fully connected dense layer and an output softmax layer for multi-class classification. The model was trained using the Adam optimizer with categorical cross-entropy loss over 200 epochs. Comment out stages in the file to produce reduced-depth variants.

**ResNetHinge.py:** 
The final dense layer was modified to activated by linear function to align with hinge loss requirements.

**TransResNet.py:** 
Applying transfer learning principles, it incorporated pretrained weights from ImageNet and is finetuned for 30 epoches on the current dataset.
