# CIFAR-10 Classification using Spiking Neural Network

This project implements a Spiking Neural Network (SNN) for image classification on the CIFAR-10 dataset using the SpikingJelly framework.

## Features

- Implements a convolutional SNN architecture
- Uses surrogate gradient for training
- Processes information over multiple timesteps
- Includes both training and testing pipelines

## Requirements

- Python 3.8+
- PyTorch
- SpikingJelly
- torchvision
- tqdm

Install dependencies:
```bash
pip install torch torchvision tqdm spikingjelly==0.0.0.0.14
```

## Network Architecture

The network consists of:
- Multiple ConvBlock layers (Conv2D + BatchNorm + IF Neuron)
- MaxPooling layers for dimensionality reduction
- Fully connected layers for classification
- Uses IF (Integrate-and-Fire) neurons with ATan surrogate gradient

## Usage

Run training:
```bash
python snn_cifar10.py
```

The script will:
1. Download CIFAR-10 dataset automatically
2. Train the network for 200 epochs
3. Save the best model to 'snn_cifar10_best.pth'

## Model Parameters

- Simulation timesteps (T): 4
- Learning rate: 1e-3
- Optimizer: Adam
- Learning rate scheduler: CosineAnnealing
- Batch size: 128

## Performance

The model typically achieves:
- Test accuracy: 90.31%
(Results may vary based on hardware and training conditions)

See 'train_snn_cifar10.log' for more details.

## Architecture Details
Overall structure:
```python
ConvBlock(3, 128) -> ConvBlock(128, 128) -> MaxPool2d(2, 2)
-> ConvBlock(128, 256) -> ConvBlock(256, 256) -> MaxPool2d(2, 2)
-> ConvBlock(256, 512) -> ConvBlock(512, 512) -> MaxPool2d(2, 2)
-> Flatten -> Linear(512*4*4, 1024) -> IFNode(surrogate.ATan()) -> Linear(1024, 10)
```

Structure of ConvBlock:
```python
Conv2d(in_channels, out_channels) -> BatchNorm2d(out_channels) -> IFNode(surrogate.ATan())
```

## License

MIT License
