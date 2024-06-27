import torch
import torchvision.datasets
from torch import nn
from torch.utils.data import Dataset, random_split, DataLoader
import torchvision.transforms as transforms
import torch.nn.functional as F
import pandas as pd
import os
from PIL import Image
import matplotlib.pyplot as plt


def get_classes(labels_csv):
    """
    Get the class names and create a class to index dictionary map
    :param labels_csv:
    :return: classes, class_to_idx
    """
    ground_truth_df = pd.read_csv(labels_csv)
    classes = list(ground_truth_df['label'].unique())

    # Creating a dictionary for class labels
    # values = ['frog', 'truck', 'deer', 'automobile', 'bird', 'horse', 'ship', 'cat', 'dog', 'airplane']
    keys = [i for i in range(len(classes))]
    class_to_idx = {k: v for k, v in zip(classes, keys)}
    return classes, class_to_idx


# Creating a custom dataset class for my CIFAR 10 dataset
class CustomCIFAR10Dataset(Dataset):
    def __init__(self, root_dir, csv_file, transform=None):
        self.root_dir = root_dir
        self.transform = transform

        # # Loading labels from the csv file
        self.labels_df = pd.read_csv(csv_file)
        # # Extracting image IDs and Labels
        self.image_ids = self.labels_df['id'].tolist()
        self.class_names = self.labels_df['label'].tolist()
        self.classes, self.class_to_idx = get_classes(csv_file)

    def create_class_dict(self):
        # To check the unique classes
        ground_truth_df = pd.read_csv(self.labels_df)
        values = list(ground_truth_df['label'].unique())

        # Creating a dictionary for class labels
        # values = ['frog', 'truck', 'deer', 'automobile', 'bird', 'horse', 'ship', 'cat', 'dog', 'airplane']
        keys = [i for i in range(len(values))]
        class_dict = {k: v for k, v in zip(keys, values)}
        print(class_dict)

    def __len__(self):
        return len(self.labels_df)

    def __getitem__(self, idx):
        img_id = self.image_ids[idx]
        class_name = self.class_names[idx]
        class_idx = self.class_to_idx[class_name]

        # Constructing image file path
        img_name = f"{img_id}.png"
        img_path = os.path.join(self.root_dir, img_name)

        # Load image
        img = Image.open(img_path).convert('RGB')

        if self.transform:
            image = self.transform(img)

        return img, class_idx


# Creating a simple convolutional neural net
class SimpleNet(nn.Module):
    def __init__(self):
        super(SimpleNet, self).__init__()
        self.Conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.Conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16*53*53, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.Conv1(x)))
        x = self.pool(F.relu(self.Conv2(x)))
        x = x.view(-1, 16*53*53)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


if __name__ == '__main__':
    print(f"Torch version: {torch.__version__}")
    print(f"Cuda is available: {torch.cuda.is_available()}")
    # print(torch.cuda.current_device())
    print(f"Cuda running on device: {torch.cuda.get_device_name(0)}")

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

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

    # # displaying a random image to varify the custom dataset
    # idx = 100
    # image, label = custom_dataset[idx]
    #
    # # Converting tensor image to numpy array for display
    # image = image.permute(1, 2, 0).numpy()
    #
    # plt.imshow(image)
    # plt.title(label)
    # plt.axis('off')
    # plt.show()

    # Splitting data into training and validation set
    train_size = int(0.8 * len(custom_dataset))
    val_size = len(custom_dataset) - train_size
    train_dataset, val_dataset = random_split(custom_dataset, [train_size, val_size])
    print(f"size of train: {len(train_dataset)}\n size of val: {len(val_dataset)}\n")

    # Defining data loaders
    train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=8)
    validation_dataloader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=8)

    img, label = next(iter(train_dataloader))
    print(f"img: {img.shape}") # | Label: {label.shape}")
    print(f"Label: {label.shape}")

    # model class
    model1 = SimpleNet().to(device)

    # Initializing loss_function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model1.parameters(), lr=0.001, momentum=0.9)

    # Training loop
    for epoch in range(10):
        model1.train()
        running_loss = 0.0
        for batch, (image, label) in enumerate(train_dataloader):
            # print(f"shape of inputs: {inputs[0].shape}")
            image, label = image.to(device), label.to(device)
            # print(f"inputs: {inputs}")
            optimizer.zero_grad()

            # Forward pass, backward propagation and optimize
            outputs = model1(image)
            # print("model output:")
            # print(outputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # Model statistics
            running_loss += loss.item()
            if i % 2000 == 1999:    # Printing every 2000 mini-batches
                print(f"[{epoch + 1}, {batch + 1}] loss: {running_loss / 2000:.3f}")
                running_los = 0

        model1.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for batch in validation_dataloader:
                images, labels = batch
                outputs = model1(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item()

                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum(dtype=torch.int32).item()

        val_loss /= len(validation_dataloader)
        accuracy = (correct/total) * 100
        print(f"Validation loss: {val_loss:.3f} | accuracy: {accuracy:.2f}%")

    print("TRAINING FINISHED")
    print("TRAINING FINISHED")
    print("TRAINING FINISHED")



