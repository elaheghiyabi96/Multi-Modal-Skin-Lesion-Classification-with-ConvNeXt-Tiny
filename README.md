# Multi-Modal-Skin-Lesion-Classification-with-ConvNeXt-Tiny
A multi-modal deep learning framework for skin lesion classification on the HAM10000 dataset, combining dermoscopic images and clinical metadata using a ConvNeXt-Tiny backbone to achieve robust performance under class imbalance.
In this work, we developed a multi-modal deep learning model for skin lesion classification using the HAM10000 dataset, which contains 10,015 dermoscopic images across 7 diagnostic categories (akiec, bcc, bkl, df, mel, nv, vasc) along with clinically relevant metadata such as age, sex, and lesion localization. The dataset was obtained from Kaggle:
👉 https://www.kaggle.com/datasets/kmader/skin-cancer-mnist-ham10000

Data Preparation

The metadata were preprocessed by filling missing age values with the mean, applying one-hot encoding to categorical variables, and standardizing the age feature. Image paths were reconstructed from the dataset structure, and labels were encoded using a sorted class mapping for consistency. The data were split using a stratified 80/20 train–validation split to preserve class distribution. To address the strong class imbalance in HAM10000, balanced class weights were computed and applied during training.

Multi-Modal Architecture

The proposed model follows a dual-branch architecture:

Image branch:
A pretrained ConvNeXt-Tiny backbone (ImageNet weights, classification head removed) is used to extract visual features from dermoscopic images resized to 224×224. ConvNeXt-Tiny provides a strong balance between performance and computational efficiency. Global Average Pooling and dropout are applied to reduce overfitting.

Metadata branch:
Clinical metadata (≈17 features) are processed through a lightweight MLP consisting of two fully connected layers (32 and 16 units with ReLU activations).

The image and metadata features are concatenated and passed through a shared dense layer before final classification using a 7-class softmax output.

Training Strategy

The ConvNeXt-Tiny backbone was kept frozen, and only the classification and metadata fusion layers were trained using the Adam optimizer (learning rate = 1e-3). Early stopping based on validation loss was employed to prevent overfitting. A tf.data pipeline ensured efficient data loading, shuffling, batching, and prefetching.

Final Results

The model achieved strong and stable performance on the validation set:

Validation Accuracy: 73.24%

Weighted F1-score: ≈ 0.75

Macro F1-score: ≈ 0.55

The model demonstrated particularly good performance on clinically important and minority classes:

akiec: F1 ≈ 0.57

bcc: F1 ≈ 0.54

mel: F1 ≈ 0.47

vasc: Recall ≈ 0.86

These results indicate that ConvNeXt-Tiny, when combined with structured clinical metadata, can serve as an effective and computationally efficient backbone for multi-modal skin lesion classification. The approach highlights the benefit of integrating visual and non-visual information in medical image analysis tasks
