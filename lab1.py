import torch
from torch import nn
from model import Autoencoder
import torchvision
from torchvision import transforms
import matplotlib.pyplot as plt
import argparse

batch_size = 128
def test(model, loss_fn, loader, device):
    model.eval()
    loss_fn = loss_fn
    losses = []
    with torch.no_grad():
        for imgs, labels in loader:
            imgs = imgs.view(-1, 784)
            imgs = imgs.to(device=device)
            output = model(imgs)
            loss = loss_fn(imgs, output)
            losses += [loss.item()]

    print('Finished testing. Average loss =', sum(losses) / len(losses))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-l", "--weightsfile", type=str, default='MLP.8.pth')

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.0,), (1.0,))
    ])

    train_dataset = torchvision.datasets.MNIST(root='./data', train=True, transform=transform, download=True)
    test_dataset = torchvision.datasets.MNIST(root='./data', train=False, transform=transform, download=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=128, shuffle=True)

    model = Autoencoder()
    model.load_state_dict(torch.load('MLP.8.pth'))

    test(model, nn.MSELoss(), test_loader, device='cpu')

    # show one image + output
    for imgs, labels in test_loader:
        img = imgs[0]

        img = img.view(1, 784)
        output = model(img)

        plt.figure(figsize=(8, 4))

        plt.subplot(1, 2, 1)
        plt.title("Input Image")
        plt.imshow(img.view(28, 28), cmap='gray')
        plt.axis('off')

        plt.subplot(1, 2, 2)
        plt.title("Output Image")
        plt.imshow(output.view(28, 28).detach().numpy(), cmap='gray')
        plt.axis('off')

        plt.show()
        break

    # show one image + noised image + output
    noise_mag = 1
    for imgs, labels in test_loader:
        img = imgs[0]

        noisy = img + torch.rand(28, 28)
        noisy = noisy.view(1, 784)
        # noisy = noisy.clamp(0, 1)
        output = model(noisy)

        plt.figure(figsize=(12, 4))

        plt.subplot(1, 3, 1)
        plt.title("Input Image")
        plt.imshow(img.view(28, 28), cmap='gray')
        plt.axis('off')

        plt.subplot(1, 3, 2)
        plt.title("Noised Image")
        plt.imshow(noisy.view(28, 28), cmap='gray')
        plt.axis('off')

        plt.subplot(1, 3, 3)
        plt.title("Output Image")
        plt.imshow(output.view(28, 28).detach().numpy(), cmap='gray')
        plt.axis('off')

        plt.show()
        break

    # bottleneck interpolator
    noise_mag = 1
    for imgs, labels in test_loader:
        img1 = imgs[0].view(1, 784)
        img2 = imgs[1].view(1, 784)

        b1 = model.encode(img1)
        b2 = model.encode(img2)

        plt.figure(figsize=(12, 4))

        for step in range(10):
            b = torch.lerp(b1, b2, step/9.0)
            output = model.decode(b)

            plt.subplot(1, 10, step+1)
            plt.title("Step {}".format(step))
            plt.imshow(output.view(28, 28).detach().numpy(), cmap='gray')
            plt.axis('off')

        plt.show()
        break
