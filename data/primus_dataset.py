# system imports
import os

# torch imports
import torch.utils.data as data
from torch.utils.data import random_split
from torchvision import transforms

# data handling imports
from PIL import Image

# visualization imports
from prettytable import PrettyTable
import matplotlib.pyplot as plt

class PrimusDataset(data.Dataset):
    """
    A custom dataset class for handling Primus dataset.

    Parameters:
        data_path (str): The path to the directory containing the dataset.
        transform (callable, optional): A function/transform that takes in an image and returns a transformed version. Default is None.

    Attributes:
        data_path (str): The path to the directory containing the dataset.
        transform (callable): A function/transform that takes in an image and returns a transformed version.
        data (list): A list of tuples containing the image file path and semantic file path for each sample in the dataset.
    """

    def __init__(self, data_path, transform=None):
        self.data_path = data_path
        self.transform = transform
        self.data = []

        # iterate through each subdirectory (corresponding to a sample)
        for sample_dir in os.listdir(data_path):

            sample_dir_path = os.path.join(data_path, sample_dir)

            image_file = None
            semantic_file = None

            # .png-file contains image, .semantic-file contains labels
            for file in os.listdir(sample_dir_path):
                if file.endswith(".png"):
                    image_file = os.path.join(sample_dir_path, file)
                elif file.endswith(".semantic"):
                    semantic_file = os.path.join(sample_dir_path, file)

            # check if a (image, label)-pair could be found
            if image_file and semantic_file:
                self.data.append((image_file, semantic_file))
            else:
                print(f"Couldn't find {'Image in ' + str(sample_dir_path) if not image_file else 'Labels in ' + str(sample_dir_path)}!")


    def __getitem__(self, index):
        """
        Retrieves the image and labels at the given index. Returns image as tensor and labels as string.

        Parameters:
            index (int): The index of the image and labels to retrieve.

        Returns:
            tuple: A tuple containing the image and labels.
        """
        # function to transform image to tensor
        to_tensor = transforms.ToTensor()

        # obtain path for image and label at given index
        image_path, labels_path = self.data[index]

        # read image and label
        image = Image.open(image_path).convert('L')
        image = to_tensor(image)
        with open(labels_path, 'r') as file:
            labels = file.read()

        # apply transforms to image if specified
        if self.transform:
            image = self.transform(image)

        return image, labels

    def __len__(self):
        """
        Returns the total number of samples in the dataset.

        Returns:
            int: The total number of samples in the dataset.
        """
        return len(self.data)

def split_data(dataset, ratio=(0.6, 0.2, 0.2)):
    """
    Applies train-validation-test split to a given dataset

    Parameters:
        dataset: Dataset to which split will be applied
        ratio: Ratio represented as tuple of shape (train, val, test)

    Returns:
        train_data: Dataset for Training
            (i.e. fitting models)
        val_data: Dataset for Validation
            (i.e. evaluating performance of models to tune hyperparameters)
        test_data: Dataset for Testing
            (i.e. evaluating final performance)
    """
    # calculate sizes of dataset
    train_size, val_size = int(ratio[0] * len(dataset)), int(ratio[1] * len(dataset))
    test_size = len(dataset) - train_size - val_size

    # apply split
    train_data, val_data, test_data = random_split(dataset, [train_size, val_size, test_size])

    # print table of dataset sizes
    table = PrettyTable()
    table.field_names = ["Dataset", "# Samples"]
    table.align["Dataset"] = "l"
    table.add_row(["Train", len(train_data)])
    table.add_row(["Validation", len(val_data)])
    table.add_row(["Test", len(test_data)])

    print(table)

    return train_data, val_data, test_data

def visualize_sample(dataset):
    """
    Visualizes a sample from the dataset and prints its corresponding labels.

    Parameters:
        dataset (torch.utils.data.Dataset): The dataset containing the samples.

    Returns:
        None
    """
    if len(dataset) > 0:
        tensor_to_image = transforms.ToPILImage()
        image = tensor_to_image(dataset[0][0])

        plt.imshow(image, cmap='gray')
        plt.axis('off')
        plt.show()

        print(f"Labels: {dataset[0][1]}")
    else:
        print("[!] Cannot visualize sample. Dataset is empty.")