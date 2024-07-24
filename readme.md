# Recyclable and Household Waste Image Classification Project

## Overview

This project aims to develop a robust and accurate waste classification model using deep learning techniques. The model is designed to classify various types of waste, aiding in efficient waste management and recycling efforts.

## Key Achievements

1. **Data Preparation and Analysis**: Ensured a consistent distribution of classes with minor deviations in specific categories.
2. **Model Selection**: Chose EfficientNet_b3 for its balance between performance and computational efficiency.
3. **Hyperparameter Optimization**: Utilized Optuna's Tree-structured Parzen Estimator (TPE) to optimize key parameters.
4. **Model Evaluation**: Achieved an accuracy of 93.54% on the test set, marking the best performance on Kaggle at the time of posting.
5. **Production Readiness**: Tested the model with unseen images to ensure practical application.

## Installation

To run this project, you need to have Python and the following libraries installed:

- torch
- torchvision
- numpy
- matplotlib
- scikit-learn
- optuna
- PIL (Pillow)

You can install these dependencies using pip:

```bash
pip install torch torchvision numpy matplotlib scikit-learn optuna pillow
```

## Usage

### 1. Hyperparameter Optimization

Utilize Optuna to optimize key hyperparameters such as learning rate, batch size, optimizer type, and weight decay to maximize validation accuracy.

### 2. Training with the Best Parameters

Train the EfficientNet_b3 model with the best hyperparameters obtained from the optimization step. Implement dropout and regularization techniques to enhance the model's generalization capabilities.

### 3. Fine-Tuning by Unfreezing Layers

Fine-tune the model by unfreezing the last few layers and retraining with a lower learning rate to improve performance.

### 4. Final Evaluation

Evaluate the trained models on the test set to determine their performance in terms of accuracy, loss, and F1 score. Compare the results of different models to select the best-performing one.

### 5. Production Readiness: Testing with Unseen Images

Test the final model with unseen images to ensure its practical application and production readiness. This step involves classifying new images and verifying the model's predictions.

## Model Performance

### Epochs and Metrics

- Epoch [1/50] - Training Accuracy: 0.3974, Validation Loss: 1.2495, Validation Accuracy: 0.7293, Validation F1 Score: 0.7190
- Epoch [2/50] - Training Accuracy: 0.7281, Validation Loss: 0.6304, Validation Accuracy: 0.8223, Validation F1 Score: 0.8180
- Epoch [3/50] - Training Accuracy: 0.8200, Validation Loss: 0.4789, Validation Accuracy: 0.8497, Validation F1 Score: 0.8474
- Epoch [4/50] - Training Accuracy: 0.8700, Validation Loss: 0.4163, Validation Accuracy: 0.8650, Validation F1 Score: 0.8617
- Epoch [5/50] - Training Accuracy: 0.8975, Validation Loss: 0.3794, Validation Accuracy: 0.8747, Validation F1 Score: 0.8735
- Epoch [6/50] - Training Accuracy: 0.9169, Validation Loss: 0.3668, Validation Accuracy: 0.8813, Validation F1 Score: 0.8806
- Epoch [7/50] - Training Accuracy: 0.9361, Validation Loss: 0.3803, Validation Accuracy: 0.8760, Validation F1 Score: 0.8740
- Epoch [8/50] - Training Accuracy: 0.9419, Validation Loss: 0.3895, Validation Accuracy: 0.8740, Validation F1 Score: 0.8718
- Epoch [9/50] - Training Accuracy: 0.9508, Validation Loss: 0.3901, Validation Accuracy: 0.8777, Validation F1 Score: 0.8766
- Epoch [10/50] - Training Accuracy: 0.9544, Validation Loss: 0.4094, Validation Accuracy: 0.8760, Validation F1 Score: 0.8748
- Epoch [11/50] - Training Accuracy: 0.9630, Validation Loss: 0.4049, Validation Accuracy: 0.8753, Validation F1 Score: 0.8741

### Early Stopping

Early stopping was triggered to prevent overfitting, ensuring the model's performance remains optimal on the validation set.

### Final Evaluation

The base EfficientNet B3 model achieved higher accuracy (93.54%) and F1 score (93.49%) compared to the model with one unfrozen block (88.61% and 88.49% respectively), indicating that unfreezing a block did not improve generalization.

## Analysis of the Results

### Impact of Learning Rate

The learning rate was not further decreased for the model with one unfrozen block, which likely affected its performance negatively. However, further training was not initiated, considering that the base model achieved impressive results, beating all the other solutions available on Kaggle at that time.

### Potential Improvements

1. **Learning Rate Adjustment**: Use a lower learning rate for the model with unfrozen layers to improve fine-tuning.
2. **Further Fine-Tuning**: Experiment with unfreezing more blocks or different combinations of layers, and consider gradual unfreezing.

## Conclusion

In this project, we successfully developed a high-performing waste classification model using EfficientNet_b3. The model was fine-tuned and optimized using advanced techniques, achieving impressive results on the test set. The final model is ready for deployment, demonstrating its practical application in real-world scenarios.

## License

This project is licensed under the Apache 2.0 License. See the [LICENSE](LICENSE) file for details.
The model directory is available upon request.
---

Feel free to contribute to this project by opening issues or submitting pull requests. Your feedback and contributions are highly appreciated!
