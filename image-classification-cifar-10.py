import random
import torch
from torch import nn
from torch.utils.data import Dataset, random_split, DataLoader
import torchvision.transforms as transforms
import pandas as pd
import os
from PIL import Image
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
from timeit import default_timer as timer

# Paths to dataset and labels file
root_dir = "cifar-10/train"
labels_file = "cifar-10/trainLabels.csv"


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
# END OF GET CLASSES


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
            img = self.transform(img)

        return img, class_idx
# END OF CUSTOM-CIFAR-10 DATASET


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


# Function to display random images
def display_random_image(dataset: torch.utils.data.Dataset,
                         classes: list[str] = None,
                         n: int = 5,
                         seed: int = None):
    """
    1. Take in a Dataset and number of images to visualize, class names etc.
    2. To prevent visualization to go out of hand keeping number of images 10
    3. Set random seed for reproducibility
    4. Get the list of random sample indexes from the dataset
    5. Set up a matplotlib plot
    6. Loop through the random samples and plot them using matplotlib
    7. Make sure dimensions of our image matches the dimension of matplotlib(HWC)
    :return: NIl
    """

    # 2. To prevent visualization to go out of hand keeping number of images 10
    if n > 10:
        n = 10
        display_shape = False
        print(f"For display purposes number of images to display is fixed to {n}")

    # 3. Set random seed for reproducibility
    if seed:
        random.seed(seed)

    # 4. Get the list of random sample indexes from the dataset
    random_sample_idx = random.sample(range(len(dataset)), k=n)

    # 5. Set up a matplotlib plot
    plt.figure(figsize=(20, 10))

    # 6. Loop through the random samples and plot them using matplotlib
    for i, target in enumerate(random_sample_idx):
        image, label = dataset[target][0], dataset[target][1]

        # 7. Make sure dimensions of our image matches the dimension of matplotlib(HWC)
        image_adjusted = image.permute(1, 2, 0)  # [colour_channel, height, width](CHW) --> (HWC)

        # Plotting adjusted sample images
        # plt.subplot(1, n, i+1)
        plt.imshow(image_adjusted)
        plt.axis('off')
        if classes:
            title = f"Class: {classes[label]} \nShape: {image_adjusted.shape}"
            plt.title(title)
        plt.show()
# END OF DISPLAY RANDOM IMAGE


# Creating train step function
def train_step(model: torch.nn.Module,
               dataloader: torch.utils.data.DataLoader,
               loss_fn: torch.nn.Module,
               optimizer: torch.optim.Optimizer,
               device):
    """
    takes in a model and dataloader and trains the model on the dataloader.
    :return: training loss and training accuracy
    """
    # Put model in train mode
    model.train()

    # Setup train loss and train accuracy values
    train_loss, train_acc = 0, 0

    # Loop through data loader data batches
    for batch, (X, y) in enumerate(dataloader):
        # Send data to the target device
        X, y = X.to(device), y.to(device)

        # 1. Forward pass
        y_pred = model(X)   # output raw values (raw model logits)

        # 2 Calculate the loss
        loss = loss_fn(y_pred, y)
        train_loss += loss.item()

        # 3 Optimizer zero grad
        optimizer.zero_grad()

        # 4 Loss backward
        loss.backward()

        # 5 Optimizer step
        optimizer.step()

        # Calculate accuracy metric
        y_pred_class = torch.argmax(torch.softmax(y_pred, dim=1), dim=1)
        train_acc += (y_pred_class == y).sum().item()/len(y_pred)

    # Adjust metrics to get average loss and accuracy per batch
    train_loss = train_loss / len(dataloader)
    train_acc = train_acc / len(dataloader)
    return train_loss, train_acc
# # END OF TRAIN STEP FUNCTION


# Creating a test step
def val_step(model: torch.nn.Module,
              dataloader: torch.utils.data.DataLoader,
              loss_fn: torch.nn.Module,
              device):
    """
    Takes in a model and dataloader and evaluates the model on the dataloader
    :param model:
    :param dataloader:
    :param loss_fn:
    :param device: device on which model runs
    :return: validation loss and validation accuracy
    """
    # Put model in eval model
    model.eval()

    # Setup test loss and test accuracy values
    val_loss, val_acc = 0, 0

    # Turn on inference mode
    with torch.inference_mode():
        # Loop through DataLoader batches
        for batch, (X, y) in enumerate(dataloader):
            # Send data to the target device
            X, y = X.to(device), y.to(device)

            # 1. Forward pass
            test_pred_logits = model(X)

            # 2. Calculate the loss
            loss = loss_fn(test_pred_logits, y)
            val_loss += loss.item()

            # Calculate the accuracy
            val_pred_labels = test_pred_logits.argmax(dim=1)
            val_acc += ((val_pred_labels == y).sum().item())/len(val_pred_labels)

        # Calculating average loss and accuracy per batch
        val_loss = val_loss/len(dataloader)
        val_acc = val_acc/len(dataloader)
        return val_loss, val_acc
# END OF VALIDATION STEP FUNCTION


# Creating a complete training function
def train(model: torch.nn.Module,
          train_dataloader: torch.utils.data.DataLoader,
          validation_dataloader: torch.utils.data.DataLoader,
          optimizer: torch.optim.Optimizer,
          loss_fn: torch.nn.Module = nn.CrossEntropyLoss,
          device: str = 'cpu',
          epochs: int = 5):

    # 2. Create empty results dictionary
    results = {"train_loss": [],
               "train_acc": [],
               "val_loss": [],
               "val_acc": []}

    # 3. Loop through training and testing steps for a number of epochs
    for epoch in tqdm(range(epochs)):
        train_loss, train_acc = train_step(
            model=model,
            dataloader=train_dataloader,
            loss_fn=loss_fn,
            optimizer=optimizer,
            device=device
        )
        val_loss, val_acc = val_step(
            model=model,
            dataloader=validation_dataloader,
            loss_fn=loss_fn,
            device=device
        )

        # 4. Print out what happening
        print(f"\nEpoch: {epoch} | Train loss: {train_loss:.4f} | Train acc: {train_acc*100:.2f}% | Val loss: {val_loss:.4f} | Val acc: {val_acc*100:.2f}%")

        # 5 Update results dictionary
        results['train_loss'].append(train_loss)
        results['train_acc'].append(train_acc)
        results['val_loss'].append(val_loss)
        results['val_acc'].append(val_loss)

    return results
# END OF TRAIN FUNCTION


if __name__ == '__main__':
    NUM_EPOCHS = 30

    # Device agnostic code
    device = ''
    if torch.cuda.is_available():
        print("Cuda is available")
        device = 'cuda'
    else:
        print("Cuda is not available")
        device = 'cpu'
    print(f"cuda device: {torch.cuda.get_device_name(0)}")

    # Creating a transform for the images
    transform = transforms.Compose([
        transforms.Resize(size=(64, 64)),
        transforms.ToTensor()
    ])

    # Creating a custom dataset instance
    custom_dataset = CustomCIFAR10Dataset(root_dir=root_dir,
                                          csv_file=labels_file,
                                          transform=transform)

    # Visualizing some images from the dataset
    display_random_image(dataset=custom_dataset,
                         classes=get_classes(labels_file)[0],
                         n=2)

    # Splitting data into train and test dataset
    train_size = int(0.8 * len(custom_dataset))
    val_size = len(custom_dataset) - train_size
    train_data, val_data = random_split(custom_dataset, [train_size, val_size])

    # Creating data loaders
    BATCH_SIZE = 32
    NUM_WORKERS = os.cpu_count()
    train_dataloader = DataLoader(dataset=train_data, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
    validation_dataloader = DataLoader(dataset=val_data, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

    # # Checking dataloader shape
    # img, lab = next(iter(train_dataloader))
    # print(f"Shape of train dataloader images: {img.shape} \n Shape of labels: {lab.shape}")

    model_vgg_v0 = TinyVGG(input_shape=3, hidden_units=32, output_shape=len(get_classes(labels_file)[0])).to(device)

    # *************************** TEST THE MODEL SHAPE ************************************************************
    # # Trying a forward pass on a single image to test the model to figure out the flatten layer input shape
    # image_batch, labels_batch = next(iter(train_dataloader))
    # print(image_batch.shape, labels_batch.shape)
    #
    # model_vgg_v0(image_batch.to(device))
    # ************************************************************************************************************

    # *************************** TRAIN THE MODEL ****************************************************************
    # Use torchinfo to get an idea of the shapes going through middle
    from torchinfo import summary
    summary(model_vgg_v0, input_size=[1, 3, 64, 64])

    # Setting loss function and optimizer
    loss_fn = nn.CrossEntropyLoss()
    # optimizer = torch.optim.Adam(params=model_vgg_v0.parameters(),
    #                              lr=0.001)
    optimizer = torch.optim.Adam(params=model_vgg_v0.parameters(),
                                 lr=0.001,
                                 weight_decay=1e-3)

    # Start timer to calculate model training time
    start_time = timer()

    # Train model
    model_vgg_v0_results = train(model=model_vgg_v0,
                                 train_dataloader=train_dataloader,
                                 validation_dataloader=validation_dataloader,
                                 optimizer=optimizer,
                                 loss_fn=loss_fn,
                                 epochs=NUM_EPOCHS,
                                 device=device)

    # End time
    end_time = timer()
    print(f"Total training time: {(end_time - start_time)/60} min")
    # ***********************************************************************************************************
# END OF MAIN
