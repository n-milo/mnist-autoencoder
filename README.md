# MNIST Autoencoder

Autoencoder for MNIST digits. Written for ELEC 475.

Usage:

Training, saving weights to a file:

```
python train.py -z 8 -e 50 -b 2048 -s MLP.8.pth -p loss.MLP.8.png
```

Testing the model, showing encoding and decoding, noise removal, and bottleneck interpolation:

```
python lab1.py –l MLP.8.pth
```