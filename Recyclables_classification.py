#!/usr/bin/env python
# coding: utf-8

# # Recyclable and Household Waste Image Classification ♻️

# # 1. Problem Definition

# ## 1.1 Project Goal
# The primary goal of this project is to develop a robust and accurate model for classifying various types of waste materials. This classification will aid in the efficient sorting and recycling of waste, thereby contributing to environmental sustainability.

# ## 1.2 Problem Statement
# The problem is to classify images of waste materials into predefined categories such as plastic, paper and cardboard, glass, metal, organic waste, and textiles. The challenge lies in accurately identifying and categorizing these materials from images, which can vary significantly in appearance due to different lighting conditions, angles, and levels of contamination.

# ## 1.3 Impact of the Solution
# An effective solution to this problem will have a significant positive impact on waste management and recycling processes. By automating the classification of waste materials, the solution can:
# - Improve the efficiency and accuracy of waste sorting.
# - Reduce the amount of recyclable materials ending up in landfills.
# - Support environmental sustainability efforts by promoting recycling.
# - Potentially reduce the costs associated with waste management.

# ---

# # 2. Data Collection

# ## 2.1 Source of Data
# The dataset is sourced from ALISTAIR KING on Kaggle - Recyclable and Household Waste Classification, which can be found at the following URL: https://www.kaggle.com/datasets/alistairking/recyclable-and-household-waste-classification/data
# 

# ## 2.2 Composition of the Dataset
# The images are stored in the PNG format and covers the following waste categories and items:
# - **Plastic**: water bottles, soda bottles, detergent bottles, shopping bags, trash bags, food containers, disposable cutlery, straws, cup lids.
# - **Paper and Cardboard**: newspaper, office paper, magazines, cardboard boxes, cardboard packaging.
# - **Glass**: beverage bottles, food jars, cosmetic containers.
# - **Metal**: aluminum soda cans, aluminum food cans, steel food cans, aerosol cans.
# - **Organic Waste**: food waste (fruit peels, vegetable scraps), eggshells, coffee grounds, tea bags.
# - **Textiles**: clothing, shoes.
# 

# ## 2.3 Data Organization
# The dataset is organized into a hierarchical folder structure as follows:
# - **images/**
#   - **Plastic water bottles/**
#     - **default/**
#       - image1.png
#       - image2.png
#       - ...
#     - **real_world/**
#       - image1.png
#       - image2.png
#       - ...
#   - **Plastic soda bottles/**
#     - **default/**
#       - image1.png
#       - image2.png
#       - ...
#     - **real_world/**
#       - image1.png
#       - image2.png
#       - ...
#   - **Plastic detergent bottles/**
#     - **default/**
#       - image1.png
#       - image2.png
#       - ...
#     - **real_world/**
#       - image1.png
#       - image2.png
#       - ...
#   - (Similar structure for other categories)
# 
# Needless to say, there is not a seperate file for the labels, as they can be derived from the structure. 
# 

# ---

# # 3. Data Exploration and Visualization

# In[ ]:


import os, re, random, time
import pickle
import pandas as pd
import matplotlib.pyplot as plt
import altair as alt
from PIL import Image
from sklearn.metrics import accuracy_score, f1_score

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torchvision import models, datasets, transforms
from torchvision.transforms import RandomErasing

import optuna
from optuna.trial import TrialState
import optuna.visualization as vis

drive.mount('/content/drive')

dataset_path = '/content/drive/My Drive/Recyclables Classification Dataset/images/images'

save_dir = '/content/drive/My Drive/Recyclables Classification Dataset/optuna_studies'


# ## 3.1 Loading the Dataset and Visualizing the Images

# In[49]:


import os
import random
from PIL import Image
import matplotlib.pyplot as plt

# Function to load and visualize a random patch of images from the directory
def visualize_random_images(folder_path, title, num_images=5):
    images = []
    file_names = [f for f in os.listdir(folder_path) if f.endswith('.png')]
    random_files = random.sample(file_names, min(num_images, len(file_names)))

    for file_name in random_files:
        img_path = os.path.join(folder_path, file_name)
        img = Image.open(img_path)
        images.append(img)

    # Plot the images
    fig, axes = plt.subplots(1, len(images), figsize=(20, 5))
    fig.suptitle(title)
    for ax, img in zip(axes, images):
        ax.imshow(img)
        ax.axis('off')
    plt.show()

# Function to traverse the dataset and visualize random images
def visualize_dataset(dataset_path, num_images=5):
    categories = [category for category in os.listdir(dataset_path) if os.path.isdir(os.path.join(dataset_path, category))]
    selected_categories = random.sample(categories, min(3, len(categories)))

    for category in selected_categories:
        category_path = os.path.join(dataset_path, category)
        subfolder = random.choice(['default', 'real_world'])
        subfolder_path = os.path.join(category_path, subfolder)
        if os.path.isdir(subfolder_path):
            title = f"{category} - {subfolder}"
            visualize_random_images(subfolder_path, title, num_images)

visualize_dataset(dataset_path, num_images=5)


# ## 3.2 Class Distribution

# In[42]:


# Count images in each category and subfolder
def count_images(dataset_path):
    class_counts = {}
    for category in os.listdir(dataset_path):
        category_path = os.path.join(dataset_path, category)
        if os.path.isdir(category_path):
            class_counts[category] = {'default': 0, 'real_world': 0}
            for subfolder in ['default', 'real_world']:
                subfolder_path = os.path.join(category_path, subfolder)
                if os.path.isdir(subfolder_path):
                    num_images = len([f for f in os.listdir(subfolder_path) if f.endswith('.png')])
                    class_counts[category][subfolder] = num_images
    return class_counts

# Visualize the class distribution
def visualize_class_distribution(class_counts):
    categories = list(class_counts.keys())
    default_counts = [class_counts[category]['default'] for category in categories]
    real_world_counts = [class_counts[category]['real_world'] for category in categories]

    x = range(len(categories))

    fig, ax = plt.subplots(figsize=(12, 8))
    ax.bar(x, default_counts, width=0.4, label='Default', align='center')
    ax.bar(x, real_world_counts, width=0.4, label='Real World', align='edge')

    ax.set_xlabel('Categories')
    ax.set_ylabel('Number of Images')
    ax.set_title('Class Distribution of the Dataset')
    ax.set_xticks(x)
    ax.set_xticklabels(categories, rotation=90)
    ax.legend()

    plt.tight_layout()
    plt.show()

class_counts = count_images(dataset_path)
visualize_class_distribution(class_counts)


# ### Analysis of Class Distribution
# The dataset shows that most items have both 'default' and 'real_world' values set at 250, indicating consistency. Only two items, Glass Food Jars (249) and Glass Beverage Bottles (252), have slight differences from the default. This suggests that the default values are generally accurate and reflect real-world data well. The small deviations in these two items might need a closer look to understand why they differ. Overall, the data is consistent and covers a wide range of items like cans, bottles, paper, plastics, and organic waste.

# ---

# # 4. Data Preprocessing

# In[ ]:


import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, datasets, transforms
from torch.utils.data import DataLoader, random_split
from torchvision.transforms import RandomErasing

# Define transformations for training, validation, and test sets
train_transforms = transforms.Compose([
    transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),  # Randomly crop and resize to 224x224
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(degrees=15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # Normalize with ImageNet mean and std
    RandomErasing(p=0.5, scale=(0.02, 0.33), ratio=(0.3, 3.3), value='random')  # Randomly erase a rectangle region with 50% probability
])

val_test_transforms = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Load the dataset
dataset = datasets.ImageFolder(root=dataset_path, transform=train_transforms)

# Split the dataset into training, validation, and test sets
train_size = int(0.7 * len(dataset))
val_size = int(0.2 * len(dataset))
test_size = len(dataset) - train_size - val_size

train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])

# Apply validation/test transformations to the validation and test sets
val_dataset.dataset.transform = val_test_transforms
test_dataset.dataset.transform = val_test_transforms

# Create data loaders
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=2)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=2)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=2)


# ## Explanation of the Pre-processing Steps
# ### Training Set Transformations
# The training images are **augmented** and standardized through a series of transformations: random cropping and resizing to **224x224** pixels, horizontal flipping, rotation, color jittering, conversion to tensors, **normalization** using ImageNet statistics, and random erasing of a rectangle region to enhance robustness.
# 
# ### Validation and Test Set Transformations
# Validation and test images are **resized** to **256** pixels on the shorter side, center-cropped to **224x224** pixels, converted to tensors, and **normalized** using ImageNet mean and standard deviation values.
# 
# ### Dataset Loading and Splitting
# The dataset is loaded from directory, then split into training (70%), validation (20%), and test (10%).
# 
# ### Data Loaders
# Data loaders are created for each dataset split, with a batch size of 32. The training loader shuffles data and uses 2 worker threads (compatible with the free Colab environment), while the validation and test loaders do not shuffle the data.
# 
# 

# ---

# # 5. Model Selection

# ## 5.1 Selecting the Models and defining the Training function

# In[ ]:


def train_and_evaluate(model, train_loader, val_loader, criterion, optimizer, device, num_epochs=10):
    accuracies = []
    f1_scores = []

    for epoch in range(num_epochs):
        # Training loop
        model.train()
        correct_train = 0
        total_train = 0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            # Zero the parameter gradients
            optimizer.zero_grad()

            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)

            # Backward pass and optimize
            loss.backward()
            optimizer.step()

            # Calculate training accuracy
            _, predicted = torch.max(outputs, 1)
            total_train += labels.size(0)
            correct_train += (predicted == labels).sum().item()

        train_accuracy = correct_train / total_train

        # Evaluation on the validation set
        model.eval()
        val_loss = 0.0
        correct_val = 0
        total_val = 0
        all_labels = []
        all_preds = []

        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)

                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item()

                _, predicted = torch.max(outputs, 1)
                total_val += labels.size(0)
                correct_val += (predicted == labels).sum().item()

                all_labels.extend(labels.cpu().numpy())
                all_preds.extend(predicted.cpu().numpy())

        val_loss /= len(val_loader)
        val_accuracy = correct_val / total_val
        val_f1 = f1_score(all_labels, all_preds, average='weighted')

        accuracies.append(val_accuracy)
        f1_scores.append(val_f1)

        print(f'Epoch [{epoch+1}/{num_epochs}] - Training Accuracy: {train_accuracy:.4f}, Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy:.4f}, Validation F1 Score: {val_f1:.4f}')

    return accuracies, f1_scores

# Define the loss function
criterion = nn.CrossEntropyLoss()

# Define the number of classes
num_classes = len(dataset.classes)

# Define the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# List of models to train
model_configs = [
    ("ResNet34", models.resnet34, "fc"),
    ("ResNeXt50", models.resnext50_32x4d, "fc"),
    ("EfficientNetV2_s", models.efficientnet_v2_s, "classifier[1]"),
    ("EfficientNetV2_m", models.efficientnet_v2_m, "classifier[1]"),
    ("ConvNeXt_tiny", models.convnext_tiny, "classifier[2]"),
    ("EfficientNet_b3", models.efficientnet_b3, "classifier[1]"),
    ("EfficientNet_b5", models.efficientnet_b5, "classifier[1]")
]

# Dictionary to store results
results = {}

# Train and evaluate each model
for model_name, model_func, classifier_attr in model_configs:
    model = model_func(pretrained=True)
    classifier_layer = eval(f"model.{classifier_attr}")
    classifier_layer = nn.Linear(classifier_layer.in_features, num_classes)
    exec(f"model.{classifier_attr} = classifier_layer")
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Calculate the number of parameters
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"Training and evaluating {model_name}")
    start_time = time.time()
    accuracies, f1_scores = train_and_evaluate(model, train_loader, val_loader, criterion, optimizer, device, num_epochs=10)
    end_time = time.time()
    training_time = end_time - start_time

    results[model_name] = {
        "accuracies": accuracies,
        "f1_scores": f1_scores,
        "training_time": training_time,
        "num_params": num_params
    }

    print(f"{model_name} - Training Time: {training_time:.2f} seconds, Number of Parameters: {num_params}")

# Plot the results
epochs = range(1, 11)

plt.figure(figsize=(12, 6))

for model_name, metrics in results.items():
    fig, ax1 = plt.subplots()

    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Accuracy', color='tab:blue')
    ax1.plot(epochs, metrics["accuracies"], label=f'{model_name} Accuracy', color='tab:blue')
    ax1.tick_params(axis='y', labelcolor='tab:blue')

    ax2 = ax1.twinx()
    ax2.set_ylabel('F1 Score', color='tab:red')
    ax2.plot(epochs, metrics["f1_scores"], label=f'{model_name} F1 Score', color='tab:red')
    ax2.tick_params(axis='y', labelcolor='tab:red')

    fig.tight_layout()
    plt.title(f'{model_name} - Validation Accuracy and F1 Score')
    fig.legend(loc='upper left', bbox_to_anchor=(0.1,0.9))
    plt.show()

# Display training time and number of parameters
for model_name, metrics in results.items():
    print(f"{model_name} - Training Time: {metrics['training_time']:.2f} seconds, Number of Parameters: {metrics['num_params']}")


# ### Brief Explanation
# 
# The code trains and evaluates models, tracking their performance over each epoch. Key steps include:
# 
# 1. **Function `train_and_evaluate`**: Trains a model and evaluates it on validation data, recording accuracy and F1 scores.
# 2. **Model Configurations**: Defines a list of models to test, each with specific configurations.
# 3. **Training Loop**: Iterates over each model, trains it, evaluates it, and records metrics.
# 4. **Results Visualization**: Plots accuracy and F1 scores for each model and prints training time and parameter count.
# 
# ### Model Selection Reasoning
# 
# 1. **ResNet34**: is a classic image classification model known for its residual connections, which help mitigate the vanishing gradient problem, making it easier to train deeper networks. It provides a good balance between depth and computational efficiency, making it a solid baseline for comparison.
# 
# 2. **ResNeXt50 (32x4d)**: introduces a cardinality dimension (number of parallel paths) to the ResNet architecture, which allows for improved performance with fewer parameters. It is designed to achieve higher accuracy by increasing the model's capacity without significantly increasing computational cost.
# 
# 3. **EfficientNetV2_s**: designed to optimize both accuracy and efficiency by systematically scaling depth, width, and resolution. The 's' variant is a smaller version that offers faster training times and good performance.
# 
# 4. **EfficientNetV2_m**: is a medium-sized variant of the EfficientNetV2 family. It helps in understanding the trade-offs between model size, accuracy, and computational requirements.
# 
# 5. **ConvNeXt_tiny**: is a modernized CNN designed to be competitive with Vision Transformers (ViTs) while retaining the simplicity and efficiency of traditional CNNs. The 'tiny' variant is a lightweight model that provides insights into the performance of modern CNN architectures.
# 
# 6. **EfficientNet_b3**: is part of the original EfficientNet family, EfficientNet_b3 is known for its strong performance on image classification tasks with fewer parameters and FLOPs compared to traditional models.
# 
# 7. **EfficientNet_b5**: is a larger variant of the EfficientNet family, it provides insights into how increasing the model size impacts performance and computational requirements.

# ## 5.2 Training and Visualizing the Results

# In[ ]:


print(results)


# In[ ]:


import altair as alt
import pandas as pd

alt.renderers.enable('colab')


results = {
    'ResNet34': {'accuracies': [0.5213333333333333, 0.642, 0.6706666666666666, 0.7353333333333333, 0.7426666666666667, 0.7486666666666667, 0.7483333333333333, 0.7693333333333333, 0.755, 0.763], 'f1_scores': [0.5081295033529909, 0.631034418209872, 0.675525849660733, 0.7294248834904548, 0.7461875395986077, 0.742482268501885, 0.7468793306416618, 0.7689829459947185, 0.751870157636376, 0.763699358130046], 'training_time': 2810.7038106918335, 'num_params': 21300062},
    'ResNeXt50': {'accuracies': [0.5723333333333334, 0.6693333333333333, 0.7226666666666667, 0.7616666666666667, 0.7223333333333334, 0.7153333333333334, 0.7683333333333333, 0.7613333333333333, 0.748, 0.7673333333333333], 'f1_scores': [0.5629588813266339, 0.6580377909059748, 0.7122910453352991, 0.763345304486588, 0.7093089421662162, 0.719731684315688, 0.7684183675718466, 0.7582708254245388, 0.7499190438486797, 0.7656365780858606], 'training_time': 1620.0616106987, 'num_params': 23041374},
    'EfficientNetV2_s': {'accuracies': [0.753, 0.7746666666666666, 0.8036666666666666, 0.7786666666666666, 0.8203333333333334, 0.825, 0.8083333333333333, 0.838, 0.8233333333333334, 0.8183333333333334], 'f1_scores': [0.7506080839230885, 0.7766669739899547, 0.8022736579305763, 0.7752619196722441, 0.819052489478621, 0.8257148511217407, 0.8033361043247143, 0.8372094019496465, 0.8231842443006527, 0.8171476424007671], 'training_time': 1392.4612758159637, 'num_params': 20215918},
    'ConvNeXt_tiny': {'accuracies': [0.7553333333333333, 0.7543333333333333, 0.78, 0.7636666666666667, 0.7896666666666666, 0.802, 0.8056666666666666, 0.8166666666666667, 0.7763333333333333, 0.8043333333333333], 'f1_scores': [0.7483682606547979, 0.7410241796239666, 0.7733772371961729, 0.756018024119291, 0.7879340362242587, 0.7970192969009205, 0.8023133843587184, 0.8193266015160069, 0.7835362877257446, 0.8067698673197111], 'training_time': 2863.425936937332, 'num_params': 27843198},
    'EfficientNetV2_m': {'accuracies': [0.683, 0.759, 0.6263333333333333, 0.7753333333333333, 0.7826666666666666, 0.8046666666666666, 0.8183333333333334, 0.7963333333333333, 0.8033333333333333, 0.808], 'f1_scores': [0.670277520946631, 0.7572790837692275, 0.6413093514798639, 0.7689944440135893, 0.7795427358232019, 0.803577344399984, 0.8144420735087179, 0.7950387940895071, 0.8038970880079731, 0.8075271151286207], 'training_time': 2304.7926330566406, 'num_params': 52896786},
    'EfficientNet_b3': {'accuracies': [0.8243333333333334, 0.8326666666666667, 0.8226666666666667, 0.8323333333333334, 0.84, 0.8536666666666667, 0.8426666666666667, 0.859, 0.8506666666666667, 0.848], 'f1_scores': [0.8183829616918221, 0.8303248456348703, 0.8131747612176526, 0.8269887697061891, 0.8368019992733163, 0.8520778277018118, 0.8399803435908745, 0.8571353650825546, 0.8497957927398513, 0.8472812052152597], 'training_time': 1274.0337572097778, 'num_params': 10742342},
    'EfficientNet_b5': {'accuracies': [0.711, 0.757, 0.7863333333333333, 0.8103333333333333, 0.8103333333333333, 0.8076666666666666, 0.8056666666666666, 0.8096666666666666, 0.8266666666666667, 0.8236666666666667], 'f1_scores': [0.7057630213068112, 0.7624198603488529, 0.7842713314987089, 0.8077602841104768, 0.809159649294959, 0.8063334891852098, 0.798851832525373, 0.8100004334255643, 0.8270489294513554, 0.82185869892906], 'training_time': 2203.1013729572296, 'num_params': 28402254}
}

for model_name, metrics in results.items():
    metrics['training_time'] /= 60

data = []
for model_name, metrics in results.items():
    data.append({
        'Model': model_name,
        'Training Time (minutes)': metrics['training_time'],
        'F1 Score': metrics['f1_scores'][-1],
        'Accuracy': metrics['accuracies'][-1],  # Assuming you have accuracy data
        'Number of Parameters': metrics['num_params']
    })

df = pd.DataFrame(data)

df['Training Time (minutes)'] = pd.to_numeric(df['Training Time (minutes)'])

# Create the scatter plot using Altair
scatter_plot = alt.Chart(df).mark_circle().encode(
    x=alt.X('Training Time (minutes):Q', title='Training Time (minutes)', scale=alt.Scale(domain=[10, df['Training Time (minutes)'].max()])),
    y=alt.Y('F1 Score:Q', title='F1 Score', scale=alt.Scale(domain=[0.7, 1.0])),
    size=alt.Size('Number of Parameters:Q', title='Number of Parameters', scale=alt.Scale(type='log', range=[1000, 10000])),  # Increase the range for bigger dots
    color=alt.Color('Model:N', legend=alt.Legend(title='Model Name')),
    tooltip=[
        alt.Tooltip('Model:N', title='Model'),
        alt.Tooltip('Training Time (minutes):Q', title='Training Time (minutes)', format='.2f'),
        alt.Tooltip('F1 Score:Q', title='F1 Score', format='.2f'),
        alt.Tooltip('Accuracy:Q', title='Accuracy', format='.2f'),
        alt.Tooltip('Number of Parameters:Q', title='Number of Parameters', format=',')
    ]
).properties(
    title='Model Performance: F1 Score vs Training Time',
    width=800,
    height=600
)

# Display the plot
scatter_plot


# ![Models Performance](https://i.imgur.com/ZpHKEqM.png)
# 

# ### Analysis of Results
# 
# #### ResNet34
# ResNet34 shows moderate performance with a validation accuracy of **76.3%** and an F1 score of **76.37%**. The training time is relatively high at approximately **46.85 minutes**, and it has **21.3 million** parameters. This indicates that while it performs decently, it may not be the most efficient model in terms of training time.
# 
# #### ResNeXt50 (32x4d)
# ResNeXt50 performs slightly better than ResNet34, with a validation accuracy of **76.73%** and an F1 score of **76.56%**. It has a significantly lower training time of approximately **27 minutes** and **23.04 million** parameters, making it a more efficient choice compared to ResNet34.
# 
# #### EfficientNetV2_s
# EfficientNetV2_s demonstrates strong performance with a validation accuracy of **81.83%** and an F1 score of **81.71%**. It has a relatively low training time of approximately **23.21 minutes** and **20.22 million** parameters, making it both efficient and effective.
# 
# #### EfficientNetV2_m
# EfficientNetV2_m shows good performance with a validation accuracy of **80.8%** and an F1 score of **80.75%**. However, it has a higher number of parameters (**52.9 million**) and a longer training time (approximately **38.41 minutes**), making it less efficient compared to EfficientNetV2_s.
# 
# #### ConvNeXt_tiny
# ConvNeXt_tiny achieves good performance with a validation accuracy of **80.43%** and an F1 score of **80.68%**. However, it has a higher training time of approximately **47.72 minutes** and **27.84 million** parameters, indicating a higher computational cost compared to EfficientNetV2_s.
# 
# 
# #### EfficientNet_b3
# EfficientNet_b3 stands out with the highest validation accuracy of **84.8%** and an F1 score of **84.73%**. It also has the lowest training time of approximately **21.23 minutes** and the fewest parameters (**10.74 million**), making it the most efficient and effective model in this comparison.
# 
# #### EfficientNet_b5
# EfficientNet_b5 performs well with a validation accuracy of **82.37%** and an F1 score of **82.19%**. However, it has a higher training time of approximately **36.72 minutes** and more parameters (**28.4 million**) compared to EfficientNet_b3, making it less efficient.
# 
# ### Conclusion
# **EfficientNet_b3** is the most efficient (😄) and effective model in this comparison, achieving the highest accuracy and F1 score with the lowest training time and fewest parameters. Therefore, selecting EfficientNet_b3 for further optimization is an obvious choice.
# 
# #### Note:
# The plot has tooltips enabled (hover over each circle to see its values), but it is not natively supported in Jupyter Notebooks, hence only the raw picture is attached.

# ---

# # 6. Hyperparamter Tuning of the EfficientNet_b3 model

# ## 6.1 Defining and Running a Bayesian Hyperparamter Optimizer (TPE)

# In[ ]:


# Define the objective function for Optuna
def objective(trial):
    # Suggest hyperparameters
    lr = trial.suggest_float('lr', 1e-6, 1e-2, log=True)
    batch_size = trial.suggest_categorical('batch_size', [16, 24, 32])
    optimizer_name = trial.suggest_categorical('optimizer', ['SGD', 'Adam', 'AdamW'])
    weight_decay = trial.suggest_float('weight_decay', 1e-6, 1e-2, log=True)

    # Update data loaders with the new batch size
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2)

    # Initialize the model
    model = models.efficientnet_b3(weights=models.EfficientNet_B3_Weights.DEFAULT)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    model = model.to(device)

    # Get the optimizer
    optimizer = get_optimizer(optimizer_name, model.parameters(), lr, weight_decay)

    # Train and evaluate the model
    print(f"Training EfficientNet_b3 with lr={lr}, batch_size={batch_size}, optimizer={optimizer_name}, weight_decay={weight_decay}")
    start_time = time.time()
    accuracies, f1_scores = train_and_evaluate(model, train_loader, val_loader, criterion, optimizer, device, num_epochs=9)
    end_time = time.time()
    training_time = end_time - start_time

    # Return the best validation accuracy
    return max(accuracies)

# Function to get the optimizer
def get_optimizer(optimizer_name, model_params, lr, weight_decay):
    if optimizer_name == 'Adam':
        return optim.Adam(model_params, lr=lr, weight_decay=weight_decay)
    elif optimizer_name == 'SGD':
        return optim.SGD(model_params, lr=lr, weight_decay=weight_decay)
    elif optimizer_name == 'RMSprop':
        return optim.RMSprop(model_params, lr=lr, weight_decay=weight_decay)
    elif optimizer_name == 'AdamW':
        return optim.AdamW(model_params, lr=lr, weight_decay=weight_decay)
    else:
        raise ValueError(f"Unknown optimizer: {optimizer_name}")

# Create a study object and optimize the objective function
study = optuna.create_study(direction='maximize', storage=f'sqlite:///{save_dir}/example_study.db', study_name='example_study', load_if_exists=True)
study.optimize(objective, n_trials=100)

# Save the study results to a file
with open(os.path.join(save_dir, "study.pkl"), "wb") as f:
    pickle.dump(study, f)

# Load the study results from a file
with open(os.path.join(save_dir, "study.pkl"), "rb") as f:
    study = pickle.load(f)

# Display the best trial
print("Best trial:")
trial = study.best_trial
print(f"  Value: {trial.value}")
print("  Params: ")
for key, value in trial.params.items():
    print(f"    {key}: {value}")

# Function to resume the study
def resume_study(study_name, storage):
    study = optuna.load_study(study_name=study_name, storage=storage)
    study.optimize(objective, n_trials=100)
    return study

# Resume the study
study = resume_study("example_study", f'sqlite:///{save_dir}/example_study.db')


# ## Brief explanation:
# **Optuna's Tree-structured Parzen Estimator (TPE)** is used for the hyperparameter optimization process.
# 
# The `objective` function suggests hyperparameters (**learning rate, batch size, optimizer type, weight decay**), updates data loaders, initializes the model, selects the optimizer, and trains the model, returning the **maximum validation accuracy**.
# 
# The `get_optimizer` function selects the appropriate optimizer.
# 
# An `Optuna study` is created to **maximize validation accuracy**, storing results in a **SQLite database**, and optimized over **100 max trials**. The study results are saved and loaded using **pickle**, and the **best trial's details** are printed.
# 
# A `resume_study` function is provided to continue the optimization process in case it was interrupted.

# ## 6.2 Visualizing the Trials

# In[ ]:


# Load the study
study = optuna.load_study(
    study_name='example_study',
    storage=f'sqlite:///{save_dir}/example_study.db'
)

# Plot the optimization history
vis.plot_optimization_history(study).show()

# Plot the hyperparameter importance
vis.plot_param_importances(study).show()


# ![Optimization History](https://i.imgur.com/RFPTrNQ.png)
# 

# ![Hyperparameter Importance](https://i.imgur.com/19XrfKu.png)
# 

# ### Printing the Best Trials

# In[ ]:


all_trials = study.trials

# Filter out incomplete trials
completed_trials = [t for t in all_trials if t.state == TrialState.COMPLETE]

# Sort trials by their objective value in descending order
sorted_trials = sorted(completed_trials, key=lambda t: t.value, reverse=True)

# Extract the top 5 trials
top_n = 5
best_trials = sorted_trials[:top_n]

# Display the best trials
print(f"Top {top_n} trials:")
for i, trial in enumerate(best_trials):
    print(f"Trial {i+1}:")
    print(f"  Value: {trial.value}")
    print("  Params: ")
    for key, value in trial.params.items():
        print(f"    {key}: {value}")


# ---

# # 7. Applying the best Paramters and Fine Tuning

# ## 7.1 Training with the Best Paramters

# In[ ]:


# Define the model with dropout and regularization
model = models.efficientnet_b3(pretrained=True)
num_ftrs = model.classifier[1].in_features
model.classifier[1] = nn.Sequential(
    nn.Dropout(0.5),
    nn.Linear(num_ftrs, num_classes)
)
model = model.to(device)

# Define the optimizer with the best trial parameters
optimizer = optim.Adam(model.parameters(), lr=6.881098092316187e-05, weight_decay=4.6048934792804534e-05)

# Define the learning rate scheduler
scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=3, verbose=True)

# Early stopping parameters
early_stopping_patience = 5
best_val_loss = float('inf')
early_stopping_counter = 0

# Modified train_and_evaluate function with early stopping and learning rate reduction
def train_and_evaluate(model, train_loader, val_loader, criterion, optimizer, scheduler, device, num_epochs=10, save_dir=save_dir):
    accuracies = []
    f1_scores = []

    # Early stopping parameters
    early_stopping_patience = 5
    best_val_loss = float('inf')
    early_stopping_counter = 0

    for epoch in range(num_epochs):
        # Training loop
        model.train()
        correct_train = 0
        total_train = 0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            # Zero the parameter gradients
            optimizer.zero_grad()

            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)

            # Backward pass and optimize
            loss.backward()
            optimizer.step()

            # Calculate training accuracy
            _, predicted = torch.max(outputs, 1)
            total_train += labels.size(0)
            correct_train += (predicted == labels).sum().item()

        train_accuracy = correct_train / total_train

        # Evaluation on the validation set
        model.eval()
        val_loss = 0.0
        correct_val = 0
        total_val = 0
        all_labels = []
        all_preds = []

        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)

                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item()

                _, predicted = torch.max(outputs, 1)
                total_val += labels.size(0)
                correct_val += (predicted == labels).sum().item()

                all_labels.extend(labels.cpu().numpy())
                all_preds.extend(predicted.cpu().numpy())

        val_loss /= len(val_loader)
        val_accuracy = correct_val / total_val
        val_f1 = f1_score(all_labels, all_preds, average='weighted')

        accuracies.append(val_accuracy)
        f1_scores.append(val_f1)

        print(f'Epoch [{epoch+1}/{num_epochs}] - Training Accuracy: {train_accuracy:.4f}, Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy:.4f}, Validation F1 Score: {val_f1:.4f}')

        # Save the model after each epoch
        torch.save(model.state_dict(), os.path.join(save_dir, f'efficientnet_b3_epoch_{epoch+1}.pth'))

        # Check for early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            early_stopping_counter = 0
        else:
            early_stopping_counter += 1
            if early_stopping_counter >= early_stopping_patience:
                print("Early stopping triggered")
                break

        # Step the scheduler
        scheduler.step(val_loss)

    return accuracies, f1_scores

# Train and evaluate the model
accuracies, f1_scores = train_and_evaluate(model, train_loader, val_loader, criterion, optimizer, scheduler, device, num_epochs=50, save_dir=save_dir)


# ## 7.2 Fine Tuning by Unfreezing one block and Retraining

# In[ ]:


# Load the pretrained model
model = models.efficientnet_b3(pretrained=True)
num_ftrs = model.classifier[1].in_features
model.classifier[1] = nn.Sequential(
    nn.Dropout(0.5),
    nn.Linear(num_ftrs, num_classes)
)
model = model.to(device)

# Number of blocks to unfreeze
num_unfrozen_blocks = 1

# Unfreeze the last few layers
for param in model.parameters():
    param.requires_grad = False

# Unfreeze the last `num_unfrozen_blocks` blocks and classifier
for param in model.features[-num_unfrozen_blocks:].parameters():
    param.requires_grad = True
for param in model.classifier.parameters():
    param.requires_grad = True

# Define the optimizer with the best trial parameters
optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=6.881098092316187e-05, weight_decay=4.6048934792804534e-05)

# Define the learning rate scheduler
scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=3, verbose=True)

# Early stopping parameters
early_stopping_patience = 5
best_val_loss = float('inf')
early_stopping_counter = 0

# Modified train_and_evaluate function with early stopping and learning rate reduction
def train_and_evaluate(model, train_loader, val_loader, criterion, optimizer, scheduler, device, num_epochs=10, save_dir=save_dir, num_unfrozen_blocks=1):
    accuracies = []
    f1_scores = []

    # Early stopping parameters
    early_stopping_patience = 5
    best_val_loss = float('inf')
    early_stopping_counter = 0

    for epoch in range(num_epochs):
        # Training loop
        model.train()
        correct_train = 0
        total_train = 0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            # Zero the parameter gradients
            optimizer.zero_grad()

            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)

            # Backward pass and optimize
            loss.backward()
            optimizer.step()

            # Calculate training accuracy
            _, predicted = torch.max(outputs, 1)
            total_train += labels.size(0)
            correct_train += (predicted == labels).sum().item()

        train_accuracy = correct_train / total_train

        # Evaluation on the validation set
        model.eval()
        val_loss = 0.0
        correct_val = 0
        total_val = 0
        all_labels = []
        all_preds = []

        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)

                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item()

                _, predicted = torch.max(outputs, 1)
                total_val += labels.size(0)
                correct_val += (predicted == labels).sum().item()

                all_labels.extend(labels.cpu().numpy())
                all_preds.extend(predicted.cpu().numpy())

        val_loss /= len(val_loader)
        val_accuracy = correct_val / total_val
        val_f1 = f1_score(all_labels, all_preds, average='weighted')

        accuracies.append(val_accuracy)
        f1_scores.append(val_f1)

        print(f'Epoch [{epoch+1}/{num_epochs}] - Training Accuracy: {train_accuracy:.4f}, Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy:.4f}, Validation F1 Score: {val_f1:.4f}')

        # Save the model after each epoch
        torch.save(model.state_dict(), os.path.join(save_dir, f'efficientnet_b3_epoch_{epoch+1}_unfrozen_blocks_{num_unfrozen_blocks}.pth'))

        # Check for early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            early_stopping_counter = 0
        else:
            early_stopping_counter += 1
            if early_stopping_counter >= early_stopping_patience:
                print("Early stopping triggered")
                break

        # Step the scheduler
        scheduler.step(val_loss)

    return accuracies, f1_scores

# Train and evaluate the model
accuracies, f1_scores = train_and_evaluate(model, train_loader, val_loader, criterion, optimizer, scheduler, device, num_epochs=50, save_dir=save_dir, num_unfrozen_blocks=num_unfrozen_blocks)


# ---

# 8. Final Evaluation of the Model

# In[ ]:


# Load the saved models
model_paths = [
    os.path.join(save_dir, 'efficientnet_b3_epoch_6.pth'),
    os.path.join(save_dir, 'efficientnet_b3_epoch_33_unfrozen_blocks_1.pth')
]

# Evaluate a model on the test set
def evaluate_model(model, test_loader, criterion, device):
    model.eval()
    test_loss = 0.0
    correct_test = 0
    total_test = 0
    all_labels = []
    all_preds = []

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)
            test_loss += loss.item()

            _, predicted = torch.max(outputs, 1)
            total_test += labels.size(0)
            correct_test += (predicted == labels).sum().item()

            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(predicted.cpu().numpy())

    test_loss /= len(test_loader)
    test_accuracy = correct_test / total_test
    test_f1 = f1_score(all_labels, all_preds, average='weighted')

    return test_loss, test_accuracy, test_f1

# Evaluate each model and store the results
test_results = {}

for model_path in model_paths:
    model = models.efficientnet_b3(pretrained=False)
    num_ftrs = model.classifier[1].in_features
    model.classifier[1] = nn.Sequential(
        nn.Dropout(0.5),
        nn.Linear(num_ftrs, num_classes)
    )
    model.load_state_dict(torch.load(model_path))
    model = model.to(device)

    test_loss, test_accuracy, test_f1 = evaluate_model(model, test_loader, criterion, device)
    test_results[model_path] = {
        "test_loss": test_loss,
        "test_accuracy": test_accuracy,
        "test_f1": test_f1
    }

    print(f"Model: {model_path} - Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}, Test F1 Score: {test_f1:.4f}")



# In[ ]:


model_names = ['Efficientnet B3', 'Efficientnet B3 (1 Unfrozen Block)']
accuracies = [test_results[model_path]['test_accuracy'] for model_path in model_paths]
f1_scores = [test_results[model_path]['test_f1'] for model_path in model_paths]

fig, ax = plt.subplots(figsize=(10, 6))

n_metrics = 2
index = np.arange(n_metrics)
bar_width = 0.3

for i, model_name in enumerate(model_names):
    performance = [
        accuracies[i],
        f1_scores[i]
    ]
    bars = ax.bar(index + i * bar_width, performance, bar_width, label=model_name)

    for bar in bars:
        yval = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, yval + 0.01, f'{yval*100:.2f}%', ha='center', va='bottom')

ax.set_xlabel('Metrics')
ax.set_ylabel('Scores')
ax.set_title('Comparison of Model Performance')
ax.set_xticks(index + bar_width / len(model_names))
ax.set_xticklabels(['Accuracy', 'F1 Score'])

ax.legend(loc='upper left', bbox_to_anchor=(1, 1))

plt.tight_layout()
plt.show()


# ![Model Performance](https://i.imgur.com/nLYKTEe.png)
# 

# ## Analysis of the Results
# 
# - The base EfficientNet B3 model achieved higher accuracy (**93.54%**) and F1 score (**93.49%**) compared to the model with one unfrozen block (**88.61%** and **88.49%** respectively),indicating that unfreezing a block did not improve generalization.
# 
# #### Impact of Learning Rate
# - The learning rate was not further decreased for the model with one unfrozen block, which likely affected its performance negatively. However, further training was not initiated, considering that the base model achieved impressive results, beating all the other solutions available on Kaggle (at that time).
# 
# #### Potential Improvements
# - **Learning Rate Adjustment:** Use a lower learning rate for the model with unfrozen layers to improve fine-tuning.
# - **Further Fine-Tuning:** Experiment with unfreezing more blocks or different combinations of layers, and consider gradual unfreezing.

# ---

# # 9. Production Readiness: Testing with Unseen Images

# In[39]:


class_names = dataset.classes

def classify_and_display_image(model, image_path, device, class_names):
    # Define the image transformations
    transform = transforms.Compose([
        transforms.Resize((300, 300)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # Load and preprocess the image
    image = Image.open(image_path).convert('RGB')
    image_tensor = transform(image).unsqueeze(0).to(device)

    # Set the model to evaluation mode
    model.eval()

    # Perform the classification
    with torch.no_grad():
        outputs = model(image_tensor)
        _, predicted = torch.max(outputs, 1)
        predicted_class = class_names[predicted.item()]

    # Display the image and the classification result
    plt.imshow(image)
    plt.title(f'Predicted: {predicted_class}')
    plt.axis('off')
    plt.show()

# List of image paths
image_paths = [
    '/content/drive/My Drive/Recyclables Classification Dataset/tea_bag.jpg',
    '/content/drive/My Drive/Recyclables Classification Dataset/plastic_bottle.jpg',
    '/content/drive/My Drive/Recyclables Classification Dataset/egg_shells.jpg'
    # Source: www.istockphoto.com
]

# Run the model over the images (model must be loaded and moved to device)
for image_path in image_paths:
    classify_and_display_image(model, image_path, device, class_names)


# ---

# # 10. Conclusion

# In this project, we successfully developed a robust and accurate waste classification model through a comprehensive and methodical approach. **Key achievements** include thorough data preparation and analysis, which revealed a consistent distribution of classes, with minor deviations in specific categories. After a comprehensive model selection process, the **EfficientNet_b3 model** was selected for its balance between performance and computational efficiency and was fine-tuned with dropout and regularization techniques to enhance its generalization capabilities.
# 
# **Optuna's Tree-structured Parzen Estimator (TPE)** was employed for hyperparameter optimization, optimizing key parameters such as learning rate, batch size, optimizer type, and weight decay to maximize validation accuracy. The model's performance was evaluated using **accuracy and F1 score**, with training and validation loops incorporating early stopping and learning rate reduction to prevent overfitting. The best model showed an accuracy of **93.54%** on the test set, marking the best performance on Kaggle at the time of posting.
# The model was tested with unseen images to ensure **production readiness**, demonstrating the model's practical application.
# 
# Overall, the project achieved its goal, creating a high-performing and reliable waste classification model ready for deployment in real-world applications, making it a valuable tool for environmental sustainability efforts.
