import torch
import numpy

def load_data():
    """
    Load the CelebA data and apply the appropriate transfromations.
    """
    raise NotImplementedError

def show_random_celeba():
    """
    Displays random CelebA examples.
    """
    raise NotImplementedError

def create_fnn(in_size, num_classes):
    """
    Create a fully connected neural network.

    Args:
        in_size (int): the input size of the network, should match data shape
        num_classes (int): the # of classes, which is the network output size
    
    Returns:
        fnn (nn.Sequential): the fully connected network whose input is of
        size in_size and output is of size num_classes
    """
    raise NotImplementedError

def compute_accuracy(net, dataloader):
    """
    Return the accuracy of the network on all data points in the dataloader.
    This is the sum of the number of correct predictions divided by the total 
    number of samples in the dataloader.

    Args:
        net (nn.Sequential): the network to compute the accuracy of
        dataloader (utils.DataLoader): a dataloader to compute accuracy over
        device (str): the device to send Tensors to
    
    Returns:
        float: the net's accuracy on the data from dataloader
    """
    raise NotImplementedError

def train_nn(net, trainloader, validloader, eval_freq, num_epochs):
    """
    """
    raise NotImplementedError

def create_acc_curve(train_acc, valide_acc, eval_freq):
    """
    """
    raise NotImplementedError

def predict(image):
    """
    Predicts from the given image.
    """
    raise NotImplementedError