import torch
import torchvision.datasets
from torch import nn
from torch.utils.data import Dataset, random_split
import torchvision.transforms as transforms
import pandas as pd
import os
from PIL import Image
import matplotlib.pyplot as plt


print(f"Torch version: {torch.__version__}")
print(f"Cuda is available: {torch.cuda.is_available()}")
# print(torch.cuda.current_device())
print(f"Cuda running on device: {torch.cuda.get_device_name(0)}")


# Creating a custom dataset class for my CIFAR 10 dataset
class CustomCIFAR10Dataset(Dataset):
    def __init__(self, root_dir, csv_file, transform=None):
        self.root_dir = root_dir
        self.transform = transform

        # Loading labels from the csv file
        self.labels_df = pd.read_csv(csv_file)
        # Extracting image IDs and Labels
        self.image_ids = self.labels_df['id'].tolist()
        self.image_labels = self.labels_df['label'].tolist()

    def __len__(self):
        return len(self.labels_df)

    def __getitem__(self, idx):
        img_id = self.image_ids[idx]
        img_label = self.image_labels[idx]

        # Constructing image file path
        img_name = f"{img_id}.png"
        img_path = os.path.join(self.root_dir, img_name)

        # Load image
        img = Image.open(img_path)

        if self.transform:
            img = self.transform(img)

        return img, img_label


if __name__ == '__main__':
    # required transformations
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])

    # Paths to dataset and labels file
    root_dir = "cifar-10/train"
    labels_file = "cifar-10/trainLabels.csv"

    # Custom dataset instance
    custom_dataset = CustomCIFAR10Dataset(root_dir=root_dir,
                                          csv_file=labels_file,
                                          transform=transform)

    # displaying a random image to varify the custom dataset
    idx = 50
    image, label = custom_dataset[idx]

    # Converting tensor image to numpy array for display
    image = image.permute(1, 2, 0).numpy()

    plt.imshow(image)
    plt.title(label)
    plt.axis('off')
    plt.show()




#
# # displaying a random image from the dataset to check
# import torchvision.transforms.functional as tf
# import matplotlib.pyplot as plt
# image, label = train_dataset[0]
# image = tf.to_pil_image(image)
# plt.imshow(image)
# plt.title("Test dataset image")
# plt.axis('off')
# plt.show()
#
# # Loading ground truth labels from train.csv
# train_labels = pd.read_csv("cifar-10/trainLabels.csv")
# print(train_labels.head(5))
#
# # Splitting the data into training and validation set at random



