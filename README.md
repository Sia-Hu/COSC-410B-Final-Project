# Intelligent Image-Based Waste Sorting System for Sustainable Resource Management
Code and dataset for Sia and Christine's final project for Spring 2025: COSC 410B Applied Machine Learning.
# Dataset
We used TrashNet collected by Gary Thung and Mindy Yang.

**Total Images:** 2,527

**Minimal Example.ipynb:**
This simple CNN architecture consisted of three convolutional layers, each followed by a max pooling layer. Starting with 32 filters in the first layer, the network increased the filter count to 64 and 128 in subsequent layers, using the ReLU activation function to introduce non-linearity. After the convolutional blocks, a flatten layer converted the feature maps into a one-dimensional vector, which was then passed through a fully connected dense layer. A dropout layer with a rate of 0.5 was included to mitigate overfitting by randomly deactivating some neurons during training. Finally, the output layer employed a softmax activation function with as many neurons as there are waste categories, enabling multi-class classification. The model was trained using the Adam optimizer with categorical cross-entropy as the loss function, and the training process ran for 30 epochs. While the training accuracy reached approximately 80%, the validation accuracy was only around 35% with a significantly higher validation loss. This discrepancy between training and validation performance suggests that the model was overfitting to the training data. Thus, we would implement data augmentation that could have helped improve generalization. 
