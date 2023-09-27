import matplotlib.pyplot as plt

import model
import torch
from torch import nn
import torchvision
from torchvision import transforms
from torchsummary import summary
import argparse


def train(n_epochs, optimizer, model, loss_fn, train_loader, scheduler, device='cpu', save_file=None, plot_file=None):
    print('training...')
    model.train()

    losses_train = []

    for epoch in range(n_epochs):
        loss_train = 0.0
        for imgs, _ in train_loader:
            imgs = imgs.view(-1, 784)
            imgs = imgs.to(device=device)
            output = model(imgs)
            loss = loss_fn(output, imgs)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_train += loss.item()

        scheduler.step(loss_train)
        losses_train += [loss_train / len(train_loader)]
        print('Epoch [{}/{}], Loss: {:.4f}'.format(epoch + 1, n_epochs, loss.item()))

    if save_file is not None:
        torch.save(model.state_dict(), save_file)

    if plot_file is not None:
        plt.figure(2, figsize=(12, 7))
        plt.clf()
        plt.plot(losses_train, label='train')
        plt.xlabel('epoch')
        plt.ylabel('loss')
        plt.legend(loc=1)
        plt.savefig(plot_file)


if __name__ == "__main__":
    # Create an ArgumentParser object
    parser = argparse.ArgumentParser()

    # Define command line arguments
    parser.add_argument("-z", "--bottleneck", type=int, default=8)
    parser.add_argument("-e", "--epochs", type=int, default=50)
    parser.add_argument("-b", "--batchsize", type=int, default=128)
    parser.add_argument("-s", "--savefile", type=str, default="MLP.8.pth")
    parser.add_argument("-p", "--plotfile", type=str, default="loss.MLP.8.png")

    args = parser.parse_args()

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.0,), (1.0,))
    ])

    train_dataset = torchvision.datasets.MNIST(root='./data', train=True, transform=transform, download=True)
    test_dataset = torchvision.datasets.MNIST(root='./data', train=False, transform=transform, download=True)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batchsize, shuffle=True)

    device = 'cpu'
    model = model.Autoencoder(N_bottleneck=args.bottleneck, device=device)
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)
    train(args.epochs, optimizer, model, loss_fn, train_loader, scheduler, device, args.savefile, args.plotfile)

    summary(model)
