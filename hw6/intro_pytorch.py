import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# Feel free to import other packages, if needed.
# As long as they are supported by CSL machines.


def get_data_loader(training = True):
    """
    TODO: implement this function.

    INPUT:
        An optional boolean argument (default value is True for training dataset)

    RETURNS:
        Dataloader for the training set (if training = True) or the test set (if training = False)
    """

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    train_set = datasets.FashionMNIST('./ data', train = True, download = True, transform=transform)
    test_set = datasets.FashionMNIST('./ data', train = False,transform=transform)



    trainingSetLoader = DataLoader(train_set,batch_size=64)
    testSetLoader = DataLoader(test_set,batch_size=64,shuffle=False)

    if(training):
        return trainingSetLoader
    return testSetLoader


def build_model():
    """
    TODO: implement this function.

    INPUT:
        None

    RETURNS:
        An untrained neural network model
    """

    model = nn.Sequential(
        nn.Flatten(),
        nn.Linear(784,128),
        nn.ReLU(),
        nn.Linear(128,64),
        nn.ReLU(),
        nn.Linear(64,10)
    )
    return model


def train_model(model, train_loader, criterion, T):
    """
    TODO: implement this function.

    INPUT:
        model - the model produced by the previous function
        train_loader  - the train DataLoader produced by the first function
        criterion   - cross-entropy
        T - number of epochs for training

    RETURNS:
        None
    """

    opt = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    model.train()

    for epoch in range(T):
        runningLoss = 0.0
        length = 0
        totalTested=0
        totalCorrect = 0
        for i,data in enumerate(train_loader,0):
            length+=1
            inputs,labels= data
            # totalTested+=labels.size(0)

            opt.zero_grad()

            outputs = model(inputs)
            loss = criterion(outputs,labels)
            loss.backward()
            opt.step()

            _, predicted = torch.max(outputs.data, 1)
            totalTested += labels.size(0)
            totalCorrect += (predicted == labels).sum().item()

            runningLoss+=loss.item()
        print("Train Epoch: " + str(epoch) + "\t Accuracy: "+ str(totalCorrect)+
              "/"+str(totalTested)+"(" + "{:.2f}".format(totalCorrect/totalTested*100)+"%)\t Loss: "
              + "{:.3f}".format(runningLoss/length))


def evaluate_model(model, test_loader, criterion, show_loss = True):
    """
    TODO: implement this function.

    INPUT:
        model - the the trained model produced by the previous function
        test_loader    - the test DataLoader
        criterion   - cropy-entropy

    RETURNS:
        None
    """

    model.eval()
    with torch.no_grad():
        length = 0
        totalTested = 0
        totalCorrect = 0
        runningLoss = 0
        for i,data in enumerate(test_loader,0):
            length+=1
            inputs,labels= data

            outputs = model(inputs)
            loss = criterion(outputs,labels)
            runningLoss+=loss.item()

            _, predicted = torch.max(outputs.data, 1)
            totalTested += labels.size(0)
            totalCorrect += (predicted == labels).sum().item()

        if show_loss:
            print("Accuracy: "+ str(totalCorrect)+
                  "/"+str(totalTested)+"(" + "{:.2f}".format(totalCorrect/totalTested*100)+"%)\t Loss: "
                  + "{:.4f}".format(runningLoss/length))
        else:
            print("Accuracy: " + str(totalCorrect) +
                  "/" + str(totalTested) + "(" + "{:.2f}".format(totalCorrect / totalTested * 100, ) + "%)")


def predict_label(model, test_images, index):
    """
    TODO: implement this function.

    INPUT:
        model - the trained model
        test_images   -  test image set of shape Nx1x28x28
        index   -  specific index  i of the image to be tested: 0 <= i <= N - 1


    RETURNS:
        None
    """


    testImage = test_images[index]

    probabilties = F.softmax(model(testImage),dim=1)
    class_names = ['T - shirt / top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt'
                    , 'Sneaker', 'Bag', 'Ankle Boot']

    top3Predictions = list()
    for i in range(3):
        indexMax = 0
        maxProb = -1
        for j in range(len(class_names)):
            if j not in top3Predictions and probabilties[0][j].item() > maxProb:
                maxProb = probabilties[0][j].item()
                indexMax = j
        top3Predictions.append(indexMax)

    for i in range(3):
        print(class_names[top3Predictions[i]]+": "+str(round(probabilties[0][top3Predictions[i]].item()*100,2))+"%")







if __name__ == '__main__':
    '''
    Feel free to write your own test code here to examin
    e the correctness of your functions. 
    Note that this part will not be graded.
    '''

    criterion = nn.CrossEntropyLoss()
    train_loader = get_data_loader()
    model = build_model()
    epochs = 5
    train_model(model,train_loader,criterion,epochs)
    test_loader = get_data_loader(False)
    evaluate_model(model,test_loader,criterion,False)
    test_features, test_labels = next(iter(test_loader))
    predict_label(model, test_features, 3)



