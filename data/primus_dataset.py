# system imports
import os

# torch imports
import torch
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

    def __init__(self, data_path, vocabulary_path, transform=None):
        self.data_path = data_path
        self.transform = transform

        # list of tuples containing image and label file paths
        self.data = []

        # vocabulary
        self.vocabulary_to_index = {}
        self.index_to_vocabulary = {}

        # read vocabulary
        dict_file = open(vocabulary_path, 'r')
        
        # leave 0 for padding
        for index, line in enumerate(dict_file.readlines()):
            self.vocabulary_to_index[line.strip()] = index + 1
            self.index_to_vocabulary[index + 1] = line.strip()
        
        dict_file.close()

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
        Resizes the image to a fixed height of 128 pixels while preserving the aspect ratio.

        Parameters:
            index (int): The index of the image and labels to retrieve.

        Returns:
            tuple: A tuple containing the image and labels.
        """
        # Obtain path for image and label at given index
        image_path, labels_path = self.data[index]

        # Read image
        image = Image.open(image_path).convert('L')

        # Calculate new width to preserve aspect ratio
        original_width, original_height = image.size
        aspect_ratio = original_width / original_height
        new_width = int(aspect_ratio * 128)  # Fixed height is 128

        # Define the resize transformation
        resize_transform = transforms.Resize((128, new_width))

        # Apply the resize transformation
        image = resize_transform(image)

        # Convert image to tensor
        to_tensor = transforms.ToTensor()
        image = to_tensor(image)

        # Read and process labels
        with open(labels_path, 'r') as file:
            labels = file.read()
            labels = torch.tensor([self.vocabulary_to_index[label] for label in labels.split('\t') if label != ''])

        # Apply additional transforms to image if specified
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

        print(f"labels.shape: {dataset[0][1].shape}")
    else:
        print("[!] Cannot visualize sample. Dataset is empty.")

def collate_fn(batch):
    """
    Collate function for batch processing in primus dataset. 
    
    Pads data and labels to be compatible with batch processing. Data is padded
    to the maximum width while labels are padded to the longest sequence. While
    padding for height is implemented, it is not used as the height of all images
    in the dataset is 128 pixels.

    Args:
        batch (list): List of tuples containing data and labels.

    Returns:
        tuple: A tuple containing the padded data and labels.
    """
    # Separate data and labels
    data, labels = zip(*batch)

    # Find the maximum width and height in the batch
    max_width = max([d.shape[2] for d in data])
    max_height = max([d.shape[1] for d in data])

    # Handling data (images)
    padded_data = []
    for d in data:
        # Calculate padding size
        padding_left = (max_width - d.shape[2]) // 2
        padding_right = max_width - d.shape[2] - padding_left
        padding_top = (max_height - d.shape[1]) // 2
        padding_bottom = max_height - d.shape[1] - padding_top

        # Apply padding
        padded = torch.nn.functional.pad(d, (padding_left, padding_right, padding_top, padding_bottom), "constant", 0)
        padded_data.append(padded)

    # Stack all the padded images and labels into tensors
    padded_data = torch.stack(padded_data)

    # Handling labels
    # Find the maximum label length
    max_label_len = max([len(l) for l in labels])

    # Pad labels
    padded_labels = []
    for l in labels:
        # Padding length
        padding_len = max_label_len - len(l)

        # Pad and append
        padded_label = torch.cat((l, torch.full((padding_len,), -1, dtype=torch.long))) # Using -1 as padding token
        padded_labels.append(padded_label)

    # Stack padded labels
    labels = torch.stack(padded_labels)

    return padded_data, labels
