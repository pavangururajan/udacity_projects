import torch
from torch import nn
from torch import optim
from torchvision import datasets, transforms, models
from collections import OrderedDict
import os
import time
import argparse

DEFAULT_MODEL_ARCHITECTURE = 'vgg16'
DEFAULT_LEARNING_RATE = 0.001
DEFAULT_HIDDEN_UNITS = 512
DEFAULT_DROPOUT = 0.2

INFEATURES = {
    'vgg16': 25088,
    'vgg13': 25088,
    'vgg11': 25088,
    'vgg19': 25088,
    'densenet121': 1024,
    'alexnet': 9216
}

MODELS = {
    'vgg16': models.vgg16,
    'densenet121': models.densenet121,
    'alexnet': models.alexnet
}


class FlowerImageClassifier:

    def __init__(self, data_dir, save_dir, arch=DEFAULT_MODEL_ARCHITECTURE, epochs=30, gpu_enabled=False,
                 learning_rate=DEFAULT_LEARNING_RATE, hidden_units=DEFAULT_HIDDEN_UNITS, dropout=DEFAULT_DROPOUT):
        self._data_dir = data_dir
        self._save_dir = save_dir
        self._arch = arch
        self._gpu = gpu_enabled
        self._learning_rate = learning_rate
        self._hidden_units = hidden_units
        self._dropout = dropout
        self._epochs = epochs

        self._train_dir = os.path.join(self._data_dir, 'train')
        self._valid_dir = os.path.join(self._data_dir, 'valid')
        self._test_dir = os.path.join(self._data_dir, 'test')

        if (not os.path.exists(self._valid_dir)):
            self._valid_dir = self._test_dir

        self._total_classes = len(
            [name for name in os.listdir(self._train_dir) if os.path.isdir(os.path.join(self._train_dir, name))])

        self._train_loader, self._valid_loader, self._test_loader = self.get_data_loaders()

        self._model = self.create_model()

    def get_data_loaders(self):
        """
        Returns data loades for all datasets
        :return: tuple containing train, validation and test datasets
        """
        print("Creating data loaders")
        data_transforms = transforms.Compose([transforms.RandomRotation(30),
                                              transforms.RandomResizedCrop(224),
                                              transforms.RandomHorizontalFlip(),
                                              transforms.ToTensor(),
                                              transforms.Normalize([0.485, 0.456, 0.406],
                                                                   [0.229, 0.224, 0.225])])
        # Load the datasets with ImageFolder
        train_datasets = datasets.ImageFolder(self._train_dir, transform=data_transforms)
        test_datasets = datasets.ImageFolder(self._test_dir, transform=data_transforms)
        valid_datasets = datasets.ImageFolder(self._valid_dir, transform=data_transforms)

        # Using the image datasets and the trainforms, define the dataloaders
        train_loader = torch.utils.data.DataLoader(train_datasets, batch_size=64, shuffle=True)
        test_loader = torch.utils.data.DataLoader(test_datasets, batch_size=64, shuffle=True)
        valid_loader = torch.utils.data.DataLoader(valid_datasets, batch_size=64, shuffle=True)

        return (train_loader, valid_loader, test_loader)

    def create_model(self):
        """
        Creates a model with a classifier under a given pretrained architecture
        :return: model
        """
        classifier = nn.Sequential(OrderedDict([
            ('fc1', nn.Linear(INFEATURES[self._arch.lower()], self._hidden_units)),
            ('relu1', nn.ReLU()),
            ('dropout1', nn.Dropout(self._dropout)),
            ('fc2', nn.Linear(self._hidden_units, 256)),
            ('relu2', nn.ReLU()),
            ('dropout2', nn.Dropout(self._dropout)),
            ('fc3', nn.Linear(256, self._total_classes)),
            ('output', nn.LogSoftmax(dim=1))
        ]))

        model_arch = MODELS[self._arch.lower()]
        model = model_arch.__call__(pretrained=True)
        model.parent = self._arch

        for param in model.parameters():
            param.requires_grad = False

        model.classifier = classifier
        return model

    def train(self):
        """
        Train the model
        :return:
        """
        model = self._model
        torch.cuda.empty_cache()

        criterion = nn.NLLLoss()
        optimizer = optim.Adam(model.classifier.parameters(), lr=self._learning_rate)
        device = torch.device("cuda:0" if torch.cuda.is_available() and self._gpu else "cpu")
        model.to(device)

        start_time = time.time()

        train_losses, validation_losses = [], []
        print("================ Started Training ===================")
        for e in range(self._epochs):
            running_loss = 0
            for images, labels in self._train_loader:
                images = images.to(device)
                labels = labels.to(device)
                optimizer.zero_grad()
                outputs = model.forward(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()

            else:
                validation_loss = 0
                accuracy = 0
                model.eval()
                # Turn off gradients for validation, saves memory and computations
                with torch.no_grad():
                    for images, labels in self._valid_loader:
                        images = images.to(device)
                        labels = labels.to(device)
                        log_ps = model.forward(images)
                        validation_loss += criterion(log_ps, labels)

                        ps = torch.exp(log_ps)
                        top_p, top_class = ps.topk(1, dim=1)
                        equals = top_class == labels.view(*top_class.shape)
                        accuracy += torch.mean(equals.type(torch.FloatTensor))

                train_losses.append(running_loss / len(self._train_loader))
                validation_losses.append(validation_loss / len(self._valid_loader))

                print("Epoch: {}/{}.. ".format(e + 1, self._epochs),
                      "Training Loss: {:.3f}.. ".format(running_loss / len(self._train_loader)),
                      "Validation Loss: {:.3f}.. ".format(validation_loss / len(self._valid_loader)),
                      "Validation Accuracy: {:.3f}".format(accuracy / len(self._valid_loader)))
                model.train()

        time_taken = time.time() - start_time
        print("=================Training done. Total Time Taken: %s====================" % time_taken)

    def save_model(self):
        checkpoint_file = os.path.join(self._save_dir, 'checkpoint.pth')
        self._model.class_to_idx = self._train_datasets.class_to_idx

        checkpoint = {'input_size': INFEATURES[self._arch.lower()],
                      'output_size': self._total_classes,
                      'classifier': self._model.classifier,
                      'state_dict': self._model.state_dict(),
                      'class_to_idx': self._model.class_to_idx,
                      'parent_model': self._model.parent}

        torch.save(checkpoint, checkpoint_file)
        print("Model saved to location: %s" % checkpoint_file)


def parse_arguments():
    parser = argparse.ArgumentParser(description="Train Flower images")
    parser.add_argument('--save_dir', help='Save directory for model', default=os.getcwd())
    parser.add_argument('--arch', help='Model architecture', default='vgg16')
    parser.add_argument('--learning_rate', help='Learning rate', default=0.001)
    parser.add_argument('--hidden_units', help='Hidden Units', default=1024)
    parser.add_argument('--gpu', help='Is GPU enabled', default=False, action='store_true')
    parser.add_argument('--epochs', help="number of epochs", default=20)
    parser.add_argument('data_dir', help="Data Directory", default=os.path.join(os.getcwd(), 'data'))
    arguments = parser.parse_args()
    return arguments


def main():
    args = parse_arguments()
    if args.arch not in MODELS:
        print("Error: Architecture not recognized. The following are allowed: %s" % MODELS.keys())
        raise Exception("Architecture not found: %s" % args.arch)

    flowerImageClassifier = FlowerImageClassifier(args.data_dir,
                                                  args.save_dir,
                                                  arch=args.arch,
                                                  epochs=int(args.epochs),
                                                  learning_rate=float(args.learning_rate),
                                                  hidden_units=int(args.hidden_units),
                                                  gpu_enabled=args.gpu)
    flowerImageClassifier.train()
    flowerImageClassifier.save_model()


if __name__ == '__main__':
    main()
