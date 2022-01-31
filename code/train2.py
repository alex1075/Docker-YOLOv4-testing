#Yolov4 training script
import argparse
import os
import shutil
import time
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
from yolov4 import YOLOv4, YOLOv3, YOLOv3_tiny
from utils import load_yolo_weights, parse_input_image, detect_objects, non_max_suppression, load_classes, draw_outputs, resize_image
from tensorflow.python.client import device_lib

ImageFolder = "./data"

def get_available_gpus():
    local_device_protos = device_lib.list_local_devices()
    return [x.name for x in local_device_protos if x.device_type == 'GPU']

def train():
    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=100, help='number of epochs')
    parser.add_argument('--batch_size', type=int, default=8, help='size of the batches')
    parser.add_argument('--image_size', type=int, default=416, help='size of each image dimension')
    parser.add_argument('--lr', type=float, default=1e-3, help='learning rate')
    parser.add_argument('--val_split', type=float, default=0.1, help='validation split ratio')
    parser.add_argument('--model_name', type=str, default='yolov4', help='yolov4 or yolov3 or yolov3-tiny')
    parser.add_argument('--weights_path', type=str, default='weights', help='path to weights file')
    parser.add_argument('--class_path', type=str, default='data/coco.names', help='path to class label file')

    args = parser.parse_args()

    # Create output directory
    os.makedirs(args.weights_path, exist_ok=True)
    os.makedirs('output', exist_ok=True)

    # Load the class labels
    labels = load_classes(args.class_path)

    # Get the list of images and split them into training and validation sets
    images = os.listdir('data/images/')
    np.random.shuffle(images)
    num_val = int(len(images) * args.val_split)
    num_train = len(images) - num_val
    train_images = images[:num_train]
    val_images = images[num_train:]

    # Get the training dataset
    train_dataset = load_dataset(train_images, args.image_size)
    val_dataset = load_dataset(val_images, args.image_size)

    # Get the training and validation dataset
    train_loader = DataLoader(train_dataset,
                                batch_size=args.batch_size,
                                shuffle=True,
                                num_workers=4,
                                pin_memory=True)

    val_loader = DataLoader(val_dataset,
                                batch_size=args.batch_size,
                                shuffle=True,
                                num_workers=4,
                                pin_memory=True)

    # Get the model
    if args.model_name == 'yolov4':
        model = YOLOv4(args.model_name, len(labels), args.image_size)
    elif args.model_name == 'yolov3':
        model = YOLOv3(args.model_name, len(labels), args.image_size)
    elif args.model_name == 'yolov3-tiny':
        model = YOLOv3_tiny(args.model_name, len(labels), args.image_size)
    else:
        print('Model not found')
        exit(0)

    # Load the pretrained weights
    load_yolo_weights(model, args.weights_path)
    
    # Get the number of model parameters
    print('Number of model parameters: {}'.format(sum([p.data.nelement() for p in model.parameters()])))

    # Get the number of available GPUs
    num_gpus = len(get_available_gpus())

    # Set the model on the GPU
    model = model.cuda()

    # Wrap the model in an optimizer
    optimizer = tf.optimizers.Adam(args.lr)

    # Set the learning rate decay
    lr_scheduler = tf.optimizers.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.95)

    # Start training
    for epoch in range(args.epochs):
        # Set the model to training mode
        model.train()

        # Reset the average loss
        avg_loss = 0.0

        # Reset the average validation loss
        avg_val_loss = 0.0

        # Reset the average validation accuracy
        avg_val_acc = 0.0

        # Reset the average validation IoU
        avg_val_iou = 0.0

        # Reset the average validation precision
        avg_val_precision = 0.0

        # Reset the average validation recall
        avg_val_recall = 0.0

        # Reset the average validation F1 score
        avg_val_f1_score = 0.0

        # Reset the average validation mAP
        avg_val_mAP = 0.0

        # Reset the average validation mAP per class
        avg_val_mAP_per_class = 0.0

        # Reset the average validation mAP per class
        avg_val_mAP_per_class_iou = 0.0

        # Reset the average validation mAP per class
        avg_val_mAP_per_class_precision = 0.0

        # Reset the average validation mAP per class
        avg_val_mAP_per_class_recall = 0.0

        # Reset the average validation mAP per class
        avg_val_mAP_per_class_f1_score = 0.0

        # Reset the average validation mAP per class
        avg_val_mAP_per_class_mAP = 0.0

        # Reset the average validation mAP per class
        avg_val_mAP_per_class_mAP_per_class = 0.0

        # Reset the average validation mAP per class
        avg_val_mAP_per_class_mAP_per_class_iou = 0.0

        # Reset the average validation mAP per class
        avg_val_mAP_per_class_mAP_per_class_precision = 0.0

        # Reset the average validation mAP per class
        avg_val_mAP_per_class_mAP_per_class_recall = 0.0

        # Reset the average validation mAP per class
        avg_val_mAP_per_class_mAP_per_class_f1_score = 0.0

        # Reset the average validation mAP per class
        avg_val_mAP_per_class_mAP_per_class_mAP = 0.0

        # Reset the average validation mAP per class
        avg_val_mAP_per_class_mAP_per_class_mAP_per_class = 0.0

        # Reset the average validation mAP per class
        avg_val_mAP_per_class_mAP_per_class_mAP_per_class_iou = 0.0

        # Reset the average validation mAP per class
        avg_val_mAP_per_class_mAP_per_class_mAP_per_class_precision = 0.0

        # Reset the average validation mAP per class
        avg_val_mAP_per_class_mAP_per_class_mAP_per_class_recall = 0.0

        # Reset the average validation mAP per class
        avg_val_mAP_per_class_mAP_per_class_mAP_per_class_f1_score = 0.0

        # Reset the average validation mAP per class
        avg_val_mAP_per_class_mAP_per_class_mAP_per_class_mAP = 0.0

        # Reset the average validation mAP per class
        avg_val_mAP_per_class_mAP_per_class_mAP_per_class_mAP_per_class = 0.0

        # Reset the average validation mAP per class
        avg_val_mAP_per_class_mAP_per_class_mAP_per_class_mAP_per_class_iou = 0.0

        # Reset the average validation mAP per class
        avg_val_mAP_per_class_mAP_per_class_mAP_per_class_mAP_per_class_precision = 0.0

        # Reset the average validation mAP per class
        avg_val_mAP_per_class_mAP_per_class_mAP_per_class_mAP_per_class_recall = 0.0

        # Reset the average validation mAP per class
        avg_val_mAP_per_class_mAP_per_class_mAP_per_class_mAP_per_class_f1_score = 0.0

        # Reset the average validation mAP per class
        avg_val_mAP_per_class_mAP_per_class_mAP_per_class_mAP_per_class_mAP = 0.0

        # Reset the average validation mAP per class
        avg_val_mAP_per_class_mAP_per_class_mAP_per_class_mAP_per_class_mAP_per_class = 0.0

        # Reset the average validation mAP per class
        avg_val_mAP_per_class_mAP_per_class_mAP_per_class_mAP_per_class_mAP_per_class_iou = 0.0

        # Reset the average validation mAP per class
        avg_val_mAP_per_class_mAP_per_class_mAP_per_class_mAP_per_class_mAP_per_class_precision = 0.0

        # Reset the average validation mAP per class
        avg_val_mAP_per_class_mAP_per_class_mAP_per_class_mAP_per_class_mAP_per_class_recall = 0.0

        # Reset the average validation mAP per class
        avg_val_mAP_per_class_mAP_per_class_mAP_per_class_mAP_per_class_mAP_per_class_f1_score = 0.0
        
        # Reset the average validation mAP per class
        avg_val_mAP_per_class_mAP_per_class_mAP_per_class_mAP_per_class_mAP_per_class_mAP = 0.0

        # Reset the average validation mAP per class
        avg_val_mAP_per_class_mAP_per_class_mAP_per_class_mAP_per_class_mAP_per_class_mAP_per_class = 0.0

        # Reset the average validation mAP per class
        avg_val_mAP_per_class_mAP_per_class_mAP_per_class_mAP_per_class_mAP_per_class_mAP_per_class_iou = 0.0

        # Reset the average validation mAP per class
        avg_val_mAP_per_class_mAP_per_class_mAP_per_class_mAP_per_class_mAP_per_class_mAP_per_class_precision = 0.0

        # Reset the average validation mAP per class
        avg_val_mAP_per_class_mAP_per_class_mAP_per_class_mAP_per_class_mAP_per_class_mAP_per_class_recall = 0.0

        # Reset the average validation mAP per class
        avg_val_mAP_per_class_mAP_per_class_mAP_per_class_mAP_per_class_mAP_per_class_mAP_per_class_f1_score = 0.0

        # Reset the average validation mAP per class
        avg_val_mAP_per_class_mAP_per_class_mAP_per_class_mAP_per_class_mAP_per_class_mAP_per_class_mAP = 0.0

        # Reset the average validation mAP per class
        avg_val_mAP_per_class_mAP_per_class_mAP_per_class_mAP_per_class_mAP_per_class_mAP_per_class_mAP_per_class = 0.0

        # Reset the average validation mAP per class
        avg_val_mAP_per_class_mAP_per_class_mAP_per_class_mAP_per_class_mAP_per_class_mAP_per_class_mAP_per_class_iou = 0.0

        # Reset the average validation mAP per class
        avg_val_mAP_per_class_mAP_per_class_mAP_per_class_mAP_per_class_mAP_per_class_mAP_per_class_mAP_per_class_precision = 0.0

        # Reset the average validation mAP per class
        avg_val_mAP_per_class_mAP_per_class_mAP_per_class_mAP_per_class_mAP_per_class_mAP_per_class_mAP_per_class_recall = 0.0

        # Reset the average validation mAP per class
        avg_val_mAP_per_class_mAP_per_class_mAP_per_class_mAP_per_class_mAP_per_class_mAP_per_class_mAP_per_class_f1_score = 0.0

        # Reset the average validation mAP per class
        avg_val_mAP_per_class_mAP_per_class_mAP_per_class_mAP_per_class_mAP_per_class_mAP_per_class_mAP_per_class_mAP = 0.0

        # Reset the average validation mAP per class
        avg_val_mAP_per_class_mAP_per_class_mAP_per_class_mAP_per_class_mAP_per_class_mAP_per_class_mAP_per_class_mAP_per_class = 0.0

        # Reset the average validation mAP per class
        avg_val_mAP_per_class_mAP_per_class_mAP_per_class_mAP_per_class_mAP_per_class_mAP_per_class_mAP_per_class_mAP_per_class_iou = 0.0

        # Reset the average validation mAP per class
        avg_val_mAP_per_class_mAP_per_class_mAP_per_class_mAP_per_class_mAP_per_class_mAP_per_class_mAP_per_class_mAP_per_class_precision = 0.0

        # Reset the average validation mAP per class
        avg_val_mAP_per_class_mAP_per_class_mAP_per_class_mAP_per_class_mAP_per_class_mAP_per_class_mAP_per_class_mAP_per_class_recall = 0.0    

        # Reset the average validation mAP per class
        avg_val_mAP_per_class_mAP_per_class_mAP_per_class_mAP_per_class_mAP_per_class_mAP_per_class_mAP_per_class_mAP_per_class_f1_score = 0.0

        # Reset the average validation mAP per class
        avg_val_mAP_per_class_mAP_per_class_mAP_per_class_mAP_per_class_mAP_per_class_mAP_per_class_mAP_per_class_mAP_per_class_mAP = 0.0

        # Reset the average validation mAP per class
        avg_val_mAP_per_class_mAP_per_class_mAP_per_class_mAP_per_class_mAP_per_class_mAP_per_class_mAP_per_class_mAP_per_class_mAP_per_class = 0.0

        # Reset the average validation mAP per class
        avg_val_mAP_per_class_mAP_per_class_mAP_per_class_mAP_per_class_mAP_per_class_mAP_per_class_mAP_per_class_mAP_per_class_mAP_per_class_iou = 0.0

        # Reset the average validation mAP per class
        avg_val_mAP_per_class_mAP_per_class_mAP_per_class_mAP_per_class_mAP_per_class_mAP_per_class_mAP_per_class_mAP_per_class_mAP_per_class_precision = 0.0

        # Reset the average validation mAP per class
        avg_val_mAP_per_class_mAP_per_class_mAP_per_class_mAP_per_class_mAP_per_class_mAP_per_class_mAP_per_class_mAP_per_class_mAP_per_class_recall = 0.0

        # Reset the average validation mAP per class
        avg_val_mAP_per_class_mAP_per_class_mAP_per_class_mAP_per_class_mAP_per_class_mAP_per_class_mAP_per_class_mAP_per_class_mAP_per_class_f1_score = 0.0



def DataLoader(path, batch_size=1, shuffle=True, num_workers=1):
    dataset = ImageFolder(path, transform=None)
    loader = tf.data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
    return loader

def load_dataset(path, batch_size=1, shuffle=True, num_workers=1):
    dataset = ImageFolder(path, transform=None)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
    return loader
