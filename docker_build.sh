#!/bin/bash

echo "Starting buildinh docker images"
PS3='Select building os architechture: '
option=("arm64" "amd64" "amd64-GPU" "tensorboard" "tf-arm" "Quit")
select fav in "${option[@]}"; do
    case $fav in
        "amd64")
            echo "Building dockerfile for amd64"
            docker build . -t alex1075/phd:latest -f Dockerfiles/Tensorflow/Dockerfile 
            docker push alex1075/phd:latest 
            ;;
        "amd64-GPU")
            echo "Building dockerfile for amd64 GPU"
            docker build . -t alex1075/phd:latest-gpu -f Dockerfiles/Tensorflow-GPU/Dockerfile 
            docker push alex1075/phd:latest-gpu
            ;;
        "arm64")
            echo "Building dockerfile for arm64"
            docker build . -t alex1075/phd:latest-arm -f Dockerfiles/Tensorflow-arm/Dockerfile
            docker push alex1075/phd:latest-arm
            ;;
        "tensorboard")
            echo "Building dockerfile for tensorboard agnostic"
            docker build . -t alex1075/phd:tensorboard -f Dockerfiles/Tensorboard/Dockerfile --platform linux/amd64 
            docker build . -t alex1075/phd:tensorboard -f Dockerfiles/Tensorboard/Dockerfile --platform linux/arm64 
            docker push alex1075/phd:tensorboard
            ;;
        "tf-arm")
            echo "Building dockerfile for Tensorflow arm"
            docker build . -t alex1075/phd:tf-arm -f Dockerfiles/Tensorflow-arm/TF/Dockerfile
            docker push alex1075/phd:tf-arm
            ;;
        "Quit")
            echo "User requested exit"
            exit
            ;;
        *) echo "invalid option $REPLY";;
    esac
done
