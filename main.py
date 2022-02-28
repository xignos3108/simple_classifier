'''
git clone https://github.com/xignos3108/simple_classifier

git add -A

git commit -m "커밋 메세지"

git push

tensorboard --logdir=runs
'''

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms

import models
from configs import CONFIGS

from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter("runs/default_5")

# device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# hyper-parameters 
config = CONFIGS

# dataset has PILImage images of range [0, 1]. 
# we transform them to Tensors of normalized range [-1, 1]
transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

# CIFAR10: 60000 32x32 color images in 10 classes, with 6000 images per class
train_dataset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform)

test_dataset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=transform)

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=config['DEFAULT_CONFIG']['batch_size'],
                                          shuffle=True) #True

test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=config['DEFAULT_CONFIG']['batch_size'],
                                         shuffle=True)

classes = ('airplane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')


# select model
model = models.ConvNet_default().to(device)
# model = models.ConvNet_deeperLayers().to(device)
# model = models.ConvNet_featureMap().to(device)
# model = models.ConvNet_alexnet().to(device)

criterion = nn.CrossEntropyLoss() # softmax is already included
optimizer = torch.optim.Adam(model.parameters(), lr=config['DEFAULT_CONFIG']['learning_rate'])


n_total_steps = len(train_loader)
num_epochs = config['DEFAULT_CONFIG']['num_epochs']
for epoch in range(num_epochs):
    n_samples = 0.0
    train_loss = 0.0
    train_accuracy = 0.0
    
    n_test_samples = 0.0
    test_loss = 0.0
    test_accuracy = 0.0
    
    n_class_accuracy = [0 for i in range(10)]
    n_class_samples = [0 for i in range(10)]
    
    for images, labels in train_loader:
        images = images.to(device)
        labels = labels.to(device)

        # forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)
        # backward pass and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # calculate train loss and accuracy
        _, predicted = torch.max(outputs, 1) # torch.max returns (value ,index)
        n_samples += labels.size(0)
        train_loss += loss.item()
        train_accuracy += (predicted==labels).sum().item()
        
        '''
        # training loss
        if (i+1) % 2000 == 0:
            print (f'Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{n_total_steps}], Loss: {loss.item():.4f}')
            writer.add_scalar('training loss', train_loss / 100, epoch*n_total_steps+i)
            train_accuracy = train_correct / 100 / predicted.size(0)
            writer.add_scalar('accuracy', train_accuracy, epoch * n_total_steps + i)
            train_correct = 0
            train_loss = 0.0
            writer.close()
        '''

    with torch.no_grad():
        for test_images, test_labels in test_loader:
            test_images = test_images.to(device)
            test_labels = test_labels.to(device)
            
            # forward pass
            test_outputs = model(test_images)
            test_loss = criterion(test_outputs, test_labels)

            # calculate test loss and accuracy
            _, predicted = torch.max(test_outputs, 1) # torch.max returns (value ,index)
            n_test_samples += test_labels.size(0)
            test_loss += test_loss.item()
            test_accuracy += (predicted==test_labels).sum().item()
            
            # calculate accuracy per classes
            for i in range(config['DEFAULT_CONFIG']['batch_size']):
                label = labels[i]
                pred = predicted[i]
                if (label == pred):
                    n_class_accuracy[label] += 1
                n_class_samples[label] += 1
        
        print("===================================================")
        print("epoch: ", epoch + 1)
        train_loss = train_loss / len(train_loader)
        writer.add_scalar('train loss', train_loss / 100, epoch*n_total_steps+i)
        train_accuracy = 100 * train_accuracy / n_samples
        writer.add_scalar('train accuracy', train_accuracy / 100, epoch*n_total_steps+i)
        print("train loss: {:.5f}, acc: {:.5f}".format(train_loss, train_accuracy))
        

        test_loss = test_loss / len(test_loader)
        writer.add_scalar('test loss', test_loss / 100, epoch*n_total_steps+i)
        test_accuracy = 100 * test_accuracy / n_test_samples

        writer.add_scalar('test accuracy', test_accuracy / 100, epoch*n_total_steps+i)
        print("test loss: {:.5f}, acc: {:.5f}".format(test_loss, test_accuracy)) 
        writer.close()


print('Finished Training')
PATH = './cnn.pth'
torch.save(model.state_dict(), PATH)

# accuracy
acc = 100.0 * test_accuracy
print(f'Accuracy of the network: {acc} %')

for i in range(10):
    class_acc = 100.0 * n_class_accuracy[i] / n_class_samples[i]
    print(f'Accuracy of {classes[i]}: {class_acc} %')