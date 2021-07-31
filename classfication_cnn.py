import argparse
import sys

import torch
import torchvision.models as models

supported_models = {"alexnet"    : models.alexnet,
                    "resnet50"   : models.resnet50,
                    "vgg16"      :models.vgg16,
                    "resnet18"   :models.resnet18,
                    "squeezenet" :models.squeezenet,
                    "densenet"   :models.densenet,
                    "inception"  :models.inception
                    }

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-model' ,choices=list(supported_models.keys()), type=str ,default='alexnet')
    parser.add_argument('--result', type=str, default=None)
    args = parser.parse_args()

    if args.result is None:
        ouput_format = sys.stdout
    else:
        ouput_format = open(args.result, 'w')

    net = supported_models[args.model]
    

