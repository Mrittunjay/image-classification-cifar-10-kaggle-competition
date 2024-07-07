"""
This code will run the inference on test dataset
for kaggle cifer-10 competition
"""
import os
import random
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import torch.cuda
import torchvision.transforms as transforms
from PIL import Image
from torch import nn
from tqdm.auto import tqdm


class TinyVGG(nn.Module):
    """
    Model architecture: TinyVGG architecture copied from CNN Explainer website
    """
    def __init__(self, input_shape: int,
                 hidden_units: int,
                 output_shape: int) -> None:
        super().__init__()
        self.conv_block_1 = nn.Sequential(
            nn.Conv2d(in_channels=input_shape,
                      out_channels=hidden_units,
                      kernel_size=3,
                      stride=1,
                      padding=2),
            nn.BatchNorm2d(hidden_units),
            nn.ReLU(),
            nn.Conv2d(in_channels=hidden_units,
                      out_channels=hidden_units,
                      kernel_size=3,
                      stride=1,
                      padding=2),
            nn.BatchNorm2d(hidden_units),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2,
                         stride=2),     # For maxpool2d layer the default stride is same as kernel size
            nn.Dropout(p=0.1)          # Drop out layer to prevent overfitting
        )
        self.conv_block_2 = nn.Sequential(
            nn.Conv2d(in_channels=hidden_units,
                      out_channels=hidden_units * 2,
                      kernel_size=3,
                      stride=1,
                      padding=2),
            nn.BatchNorm2d(hidden_units * 2),
            nn.ReLU(),
            nn.Conv2d(in_channels=hidden_units * 2,
                      out_channels=hidden_units * 2,
                      kernel_size=3,
                      stride=1,
                      padding=2),
            nn.BatchNorm2d(hidden_units*2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2,
                         stride=2),
            nn.Dropout(p=0.2)
        )
        self.conv_block_3 = nn.Sequential(
            nn.Conv2d(in_channels=hidden_units * 2,
                      out_channels=hidden_units * 4,
                      kernel_size=3,
                      stride=1,
                      padding=2),
            nn.BatchNorm2d(hidden_units * 4),
            nn.ReLU(),
            nn.Conv2d(in_channels=hidden_units * 4,
                      out_channels=hidden_units * 4,
                      kernel_size=3,
                      stride=1,
                      padding=2),
            nn.BatchNorm2d(hidden_units * 4),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2,
                         stride=2),      # For maxpool2d layer the default stride is same as kernel size
            nn.Dropout(p=0.3)  # Drop out layer to prevent overfitting
        )
        self.conv_block_4 = nn.Sequential(
            nn.Conv2d(in_channels=hidden_units * 4,
                      out_channels=hidden_units * 4,
                      kernel_size=3,
                      stride=1,
                      padding=2),
            nn.BatchNorm2d(hidden_units * 4),
            nn.ReLU(),
            nn.Conv2d(in_channels=hidden_units * 4,
                      out_channels=hidden_units * 4,
                      kernel_size=3,
                      stride=1,
                      padding=2),
            nn.BatchNorm2d(hidden_units * 4),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2,
                         stride=2),      # For maxpool2d layer the default stride is same as kernel size
            nn.Dropout(p=0.4)  # Drop out layer to prevent overfitting
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=hidden_units * 7 * 7 * 4,
                      out_features=output_shape)
        )

    def forward(self, x):
        x = self.conv_block_1(x)
        # print(x.shape)
        x = self.conv_block_2(x)
        # print(x.shape)
        x = self.conv_block_3(x)
        # print(x.shape)
        x = self.conv_block_4(x)
        # print(x.shape)
        x = self.classifier(x)
        # print(x.shape)
        return x
# END OF TINY-VGG


def show_random_images(test_images, predictions, model, device, idx_to_class, num_samples=5):
    """
    Display random images with image class and model accuracy

    :param idx_to_class: getting class name from the predicted class index
    :param test_images: list of path of test images
    :param predictions: list of predicted classes for each image
    :param model: loaded pytorch model
    :param device: current device 'cuda' or 'cpu'
    :param num_samples: maximum samples to display
    """
    sample_indices = random.sample(range(len(test_images)), num_samples)
    fig, axes = plt.subplots(1, num_samples, figsize=(15, 3))

    for i, idx in enumerate(sample_indices):
        image_path = test_images[idx]
        image = preprocess_image(image_path).to(device)
        output = model(image)
        _, predicted = torch.max(output, 1)
        predicted_idx = predicted.item()

        # Getting model confidence
        softmax_probs = torch.softmax(output, dim=1).detach().cpu().numpy()[0]
        accuracy = softmax_probs[predicted_idx]

        # Display image
        image = np.transpose(image.squeeze().cpu().numpy(), (1, 2, 0))
        axes[i].imshow(image)
        axes[i].set_title(f"Predicted Class: {idx_to_class[predicted_idx]}\nAccuracy: {accuracy*100:.2f}")
        axes[i].axis('off')

    plt.tight_layout()
    plt.show()


def load_model(model_path, device):
    num_output_classes = 10
    model = TinyVGG(input_shape=3, hidden_units=32, output_shape=num_output_classes).to(device)
    state_dict = torch.load(model_path)
    model.load_state_dict(state_dict)
    model.eval()
    return model


# Function to load and prepare a single image
def preprocess_image(image_path):
    transform = transforms.Compose([
        transforms.Resize(size=(64, 64)),
        transforms.ToTensor()
    ])
    image = Image.open(image_path).convert('RGB')
    image = transform(image).unsqueeze(0)      # Adding batch dimension
    return image


def get_classes(labels_csv):
    """
    Get the class names and create a class to index dictionary map
    :param labels_csv: Path to the CSV file containing labels
    :return: classes, class_to_idx
    """
    ground_truth_df = pd.read_csv(labels_csv)
    classes = list(ground_truth_df['label'].unique())

    # Creating a dictionary for class labels
    keys = [i for i in range(len(classes))]
    class_to_idx = {k: v for k, v in zip(classes, keys)}
    return classes, class_to_idx


def create_idx_to_class(class_to_idx):
    """
    Create an index to class dictionary map from class to index dictionary
    :param class_to_idx: Dictionary mapping class names to indices
    :return: idx_to_class
    """
    idx_to_class = {v: k for k, v in class_to_idx.items()}
    return idx_to_class


def save_predictions_to_csv(image_ids, predictions, output_file, idx_to_class):
    # Extract the image ids
    image_ids = [os.path.splitext(img)[0] for img in image_ids]

    # Convert numeric predictions to labels
    labels = [idx_to_class[pred] for pred in predictions]

    # Create a DataFrame
    df = pd.DataFrame({
        'id': image_ids,
        'label': labels
    })

    # # The values needed to be sorted
    # df = df.sort_values(by='id')    # This did not work

    # Save DataFrame to CSV
    df.to_csv(output_file, index=False)

    copy_df = pd.read_csv(output_file)
    sorted_df = copy_df.sort_values(by='id')
    sorted_df.to_csv(output_file, index=False)

    print(f"Predictions saved to {output_file}")


def main():
    test_root = "cifar-10/test"

    device = ''
    if torch.cuda.is_available():
        device = 'cuda'
        print("Inference running on CUDA device")
    else:
        device = 'cpu'
        print("Inference running on CPU")

    current_dir = os.getcwd()
    model_file = "model_vgg_v0_state_dict.pth"
    model_save_path = os.path.join(current_dir, model_file)

    model = load_model(model_save_path, device)

    # Getting labels
    labels_file = "cifar-10/trainLabels.csv"
    classes, class_to_idx = get_classes(labels_file)
    idx_to_class = create_idx_to_class(class_to_idx)

    # Getting list of images paths from test folder
    test_images = [os.path.join(test_root, img) for img in os.listdir(test_root)]
    # for item in test_images:
    #     print(item)

    # Perform inference and save predictions to csv
    predictions = []
    image_ids = []
    with torch.no_grad():
        for image_path in tqdm(test_images):
            image_name = os.path.basename(image_path)
            image_ids.append(image_name)

            # Preprocess the image
            image = preprocess_image(image_path)

            # Move the image to device
            image = image.to(device)

            # perform inference
            output = model(image)

            # Get prediction class
            _, predicted = torch.max(output, 1)
            predictions.append(predicted.item())

    # Save predictions to csv
    output_file = 'predictions.csv'
    save_predictions_to_csv(image_ids=image_ids,
                            predictions=predictions,
                            output_file=output_file,
                            idx_to_class=idx_to_class)

    print(f"Inference complete, predictions saved to {output_file}")

    # Show random images with predicted classes and model accuracy
    show_random_images(test_images, predictions, model, device, idx_to_class)


if __name__ == '__main__':
    main()






