import numpy
import scipy.linalg
from scipy.linalg import eigh
import numpy as np
import matplotlib.pyplot as plt

def load_and_center_dataset(filename):
    # Your implementation goes here!
    dataArr = np.load(filename)
    return dataArr-np.mean(dataArr,axis=0)

def get_covariance(dataset):
    # Your implementation goes here!
    return np.dot(np.transpose(dataset),dataset) / (len(dataset)-1)

def get_eig(S, m):
# Your implementation goes here!
    eigenValues,eigenVectors = scipy.linalg.eigh(S)
    sortedEigenValues = np.flipud(eigenValues)
    topMValues = sortedEigenValues[0:m]
    diagMatrix = np.diag(topMValues)

    sortedEigenVectors = np.array(eigenVectors[:len(eigenVectors), ::-1])
    topMVectors = np.array(sortedEigenVectors[:len(sortedEigenValues), 0:m])
    return diagMatrix, topMVectors


def get_eig_prop(S, prop):
    # Your implementation goes here!
    eigenValues, eigenVectors = scipy.linalg.eigh(S)
    sum = 0
    for i in range(len(eigenValues)):
        sum+=eigenValues[i]
    for i in range(len(eigenValues)):
        eigenValues[i] = eigenValues[i]/sum

    propVals = eigenValues[eigenValues[:] > prop]

    for i in range(len(propVals)):
        propVals[i] = propVals[i]*sum

    decendingOrder = np.flipud(propVals)
    diagMatrix = np.diag(decendingOrder)

    vectors = eigh(S, subset_by_value=[sum * prop, np.inf])[1]
    reversedVectors = np.array(vectors[:len(vectors), ::-1])
    propVecs = np.array(reversedVectors[:len(reversedVectors), 0:len(diagMatrix)])
    return diagMatrix,propVecs

def project_image(image, U):
    # Your implementation goes here!
    projArray = np.zeros(shape=(1, 1024))
    for i in range(0,U.shape[1]):
        vector = U[:,i]
        transpose = np.transpose(vector)
        aij = np.dot(transpose,image)
        toAdd = np.dot(aij,vector)
        projArray = np.add(toAdd,projArray)
    return projArray

def display_image(orig, proj):
    # Your implementation goes here!
    orig32x32 = numpy.transpose(numpy.reshape(orig,(32,32)))
    proj32x32 = numpy.transpose(numpy.reshape(proj,(32,32)))

    fig, axis = plt.subplots(1, 2)
    axis[0].set_title('Original')
    axis[1].set_title('Projected')

    originalImage = axis[0].imshow(orig32x32, aspect='equal')
    projectedImage = axis[1].imshow(proj32x32, aspect='equal')
    fig.colorbar(originalImage, ax=axis[0])
    fig.colorbar(projectedImage, ax=axis[1])
    plt.show()