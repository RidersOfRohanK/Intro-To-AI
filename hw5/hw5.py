import sys

import numpy
from matplotlib import pyplot as plt

def getYearlyData(fileName):
    with open(fileName, encoding='utf-8') as f:
        fullText = f.read()
    listOfYearlyData = fullText.split("\n")
    return listOfYearlyData

def question3(listOfYearlyData):
    x = []
    y=[]
    xToShow = []
    for i in range(1,len(listOfYearlyData)):
        year = listOfYearlyData[i]
        xi = year.split(",")
        x.append([1,int(xi[0])])
        xToShow.append(int(xi[0]))
        y.append(int(xi[1]))
    X = numpy.array(x)
    Y = numpy.array(y)
    plt.xlabel("Year")
    plt.ylabel("Number of frozen days")
    plt.plot(xToShow,y)
    plt.savefig("plot.jpg")
    print("Q3a:")
    print(X)
    print("Q3b:")
    print(Y)
    XT = X.transpose()
    Z=numpy.matmul(XT,X)
    print("Q3c:")
    print(Z)
    I=numpy.linalg.inv(Z)
    print("Q3d:")
    print(I)
    PI=numpy.matmul(I,XT)
    print("Q3e:")
    print(PI)
    BHat = numpy.matmul(PI,Y)
    print("Q3f:")
    print(BHat)
    return BHat
def question4(bhat):
    b0 = bhat[0]
    b1 = bhat[1]
    ytest = b0+b1*2021
    print("Q4: " + str(ytest))
def question5():
    print("Q5a: <")
    print("Q5b: The B1 value in this case is the slope of the days of ice's change from year to year. Because "
          "it is negative this means that each year the number of days the lake is covered with ice will "
          "be less than year before it."
          "In other words that means there will be less and less days where the lake is frozen as time goes by.")


def question6(bhat):
    b0= bhat[0]
    b1= bhat[1]
    x=((-b0)/b1)
    print("Q6a: "+str(x))
    print("Q6b: This answer makes sense given the data trends because what the trends mostly show us is"
          "that while the ice year over year is decreasing it is not decreasing at a very fast rate. Therefore"
          "it makes sense that ")

if __name__ == "__main__":
    fileName = sys.argv[1]
    # fileName = 'hw5.csv'
    listOfYears = getYearlyData(fileName)
    bhat = question3(listOfYears)
    question4(bhat)
    question5()
    question6(bhat)