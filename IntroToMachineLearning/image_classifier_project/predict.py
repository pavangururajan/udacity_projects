import torch
from torchvision import datasets, transforms, models
import os
import PIL
import argparse
import json
import numpy as np

MODELS = {
    'vgg16': models.vgg16,
    'densenet121': models.densenet121,
    'alexnet': models.alexnet
}


class ImagePredictor:

    def __init__(self, checkpoint, gpu_enabled=False, top_k=1, category_names_file=None):
        self._checkpoint = checkpoint
        self._gpu = gpu_enabled
        self._top_k = top_k

        self._cat_to_name = None
        if category_names_file:
            with open(category_names_file, 'r') as f:
                self._cat_to_name = json.load(f)

        self._device = torch.device("cuda:0" if torch.cuda.is_available() and self._gpu else "cpu")
        self._model = self.load_model()

    def load_model(self):
        """
        Loads an already trained deep learning model.
        """
        # Load the model
        torch.cuda.empty_cache()
        checkpoint = torch.load(self._checkpoint, map_location=self._device)
        model_arch = checkpoint['parent_model']
        if model_arch not in MODELS:
            print("Error: Unknown model architecture: %s" % model_arch)
            raise Exception("Unknown model architecture")
        model_func = MODELS[model_arch]
        model = model_func.__call__(pretrained=True)

        # Freeze parameters
        for param in model.parameters():
            param.requires_grad = False

        # Load checkpoint
        model.classifier = checkpoint['classifier']
        model.class_to_idx = checkpoint['class_to_idx']
        model.load_state_dict(checkpoint['state_dict'])
        model.parent = checkpoint['parent_model']

        return model

    def resize_with_same_aspect_ratio(self, img, expected_size=256, index=False):
        '''
        Function to resize with smaller side having expected_size while maintaining aspect ratio
        '''
        wpercent = (expected_size / float(img.size[int(index)]))
        other_index = int(not index)
        other_size = int((float(img.size[other_index]) * float(wpercent)))
        if index:
            img = img.resize((other_size, expected_size), PIL.Image.ANTIALIAS)
        else:
            img = img.resize((expected_size, other_size), PIL.Image.ANTIALIAS)
        return img

    def process_image(self, image):
        ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
            returns an Numpy array
        '''

        # Process a PIL image for use in a PyTorch model
        pil_image = PIL.Image.open(image)
        width, height = pil_image.size

        pil_image = self.resize_with_same_aspect_ratio(pil_image, 256, height < width)

        # Crop to 224x224 image
        width, height = pil_image.size
        top = (height - 224) / 2
        left = (width - 224) / 2
        right = left + 224
        bottom = top + 224
        pil_image = pil_image.crop((left, top, right, bottom))

        # Normalize the array
        np_image = np.array(pil_image) / 255
        mean = [0.485, 0.456, 0.406]
        std_dev = [0.229, 0.224, 0.225]
        np_image = (np_image - mean) / std_dev

        # transpose the array
        np_image = np_image.transpose(2, 0, 1)

        return np_image

    def predict(self, image_path):
        ''' Predict the class (or classes) of an image using a trained deep learning model.
        '''
        model = self._model
        top_k = self._top_k

        model.to(self._device)
        model.eval()

        # Convert image from numpy to torch
        torch_image = torch.from_numpy(np.expand_dims(self.process_image(image_path),
                                                      axis=0)).type(torch.FloatTensor).to("cpu")
        torch_image = torch_image.to(self._device)
        log_ps = model.forward(torch_image)
        ps = torch.exp(log_ps)

        # Top 5 results
        top_ps, top_labels = ps.topk(top_k)

        # Detatch all of the details
        top_ps = np.array(top_ps.detach())[0]
        top_labels = np.array(top_labels.detach())[0]

        # Convert to classes
        idx_to_class = {val: key for key, val in model.class_to_idx.items()}
        top_labels = [idx_to_class[lab] for lab in top_labels]
        top_flowers = None
        top_list= list(zip(top_ps, top_labels))
        if self._cat_to_name:
            top_flowers = [self._cat_to_name[lab] for lab in top_labels]
            top_list= list(zip(top_ps, top_labels, top_flowers))

        print("Result: %s" % top_list)
        return top_ps, top_labels, top_flowers


def parse_arguments():
    parser = argparse.ArgumentParser(description="Train Flower images")
    parser.add_argument('--top_k', help='top k results', default=1)
    parser.add_argument('--category_names', help='File containing json of category names')
    parser.add_argument('--gpu', help='Is GPU enabled', default=False, action='store_true')
    parser.add_argument('image_path', help="Image file path")
    parser.add_argument('checkpoint', help="Checkpoint file path")
    arguments = parser.parse_args()
    return arguments


def main():
    args = parse_arguments()
    if args.image_path is None:
        print("Error: Image path is missing")
        raise Exception("Image path missing")

    if args.checkpoint is None or not os.path.exists(args.checkpoint):
        print("Error: Checkpoint file missing: %s" % args.checkpoint)
        raise Exception("Checkpoing file missing")

    image_predictor = ImagePredictor(args.checkpoint,
                                     gpu_enabled=args.gpu,
                                     top_k=int(args.top_k),
                                     category_names_file=args.category_names)

    image_predictor.predict(args.image_path)


if __name__ == '__main__':
    main()
