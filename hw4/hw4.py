import numpy as np
from matplotlib import pyplot as plt
from scipy.cluster.hierarchy import linkage, dendrogram


class Cluster():
    def __init__(self, pokemon, clusterNum):
        self.points = []
        if (len(pokemon) != 0):
            self.addPoint(pokemon)
        self.numObj = len(self.points)
        self.clusterNum = clusterNum

    def addPoint(self, pokemon):
        self.points.append(pokemon)

    def setNumObj(self):
        self.numObj = len(self.points)


def load_data(filepath):
    with open(filepath, encoding='utf-8') as f:
        fullText = f.read()
        listOfPokemon = fullText.split("\n")
        listOfDict = []
        keyNames = listOfPokemon[0]
        Keys = listOfPokemon[0].split(",");
        count = 0
    for j in range(1, len(listOfPokemon) - 1):
        pokemon = listOfPokemon[j]
        count = count + 1
        stats = pokemon.split(",")
        if (pokemon != keyNames):
            pokeDict = {}
            for i in range(len(Keys)):
                key = Keys[i]
                pokeDict[key] = stats[i]
            listOfDict.append(pokeDict)
    approvedKeys = ['HP', 'Attack', 'Defense', 'Sp. Atk', 'Sp. Def', 'Speed']
    listOfStats = []
    for d in listOfDict:
        filtered_d = dict((k, d[k]) for k in approvedKeys if k in d)
        listOfStats.append(filtered_d)
    return listOfStats


def calc_features(row):
    arr = [int(row["Attack"]), int(row["Sp. Atk"]), int(row["Speed"]), int(row["Defense"]), int(row["Sp. Def"]),
           int(row["HP"])]
    mon = np.array(arr, dtype='int64')
    return mon


def findEuclidianDistance(pokemon1, pokemon2):
    sum = 0
    for i in range(len(pokemon1)):
        x1 = pokemon1[i]
        x2 = pokemon2[i]
        sum = sum + ((x1 - x2) * (x1 - x2))
    dist = np.sqrt(sum)
    return dist


def findFarthestPoint(cluster1, cluster2):
    max = -1
    list = []  # probably don't need this
    for point1 in cluster1.points:
        for point2 in cluster2.points:
            dist = findEuclidianDistance(point1, point2)
            if (dist > max):
                list.append((point1, point2))  # TODO - find better way to do this
                max = dist
    return max, list[-1]


def findSmallestDist(temp):
    distances = []
    for i in range(len(temp)):
        for j in range(i + 1, len(temp)):
            dist = findFarthestPoint(temp[i], temp[j])
            distances.append((dist, (temp[i], temp[j])))
    return distances


def mergeClusters(cluster1, cluster2, newClusterNum):
    newlyCreated = Cluster([], newClusterNum)

    for point in cluster1.points:
        newlyCreated.addPoint(point)
    for point in cluster2.points:
        newlyCreated.addPoint(point)
    newlyCreated.setNumObj()

    return newlyCreated


def getMin(distances):
    try:
        toReturn = min(distances)
        return toReturn
    except:
        tup = ()
        minDist = 10000000000000000000000000000
        for tupleDist in distances:
            distance = tupleDist[0][0]
            if distance==minDist:
                currentMin = tup[1][0]
                toCheckMin = tupleDist[1][0]
                if(currentMin.numObj>toCheckMin.numObj):
                    tup = tupleDist
            elif distance<minDist:
                minDist = distance
                tup = tupleDist
        return tup


def hac(features):
    n = len(features)

    dendogram = np.empty((n - 1, 4))
    count = 0
    clusters = []

    for pokemon in features:

        newCluster = Cluster(pokemon, count)
        clusters.append(newCluster)
        count += 1
    for i in range(len(dendogram)):
        distances = findSmallestDist(clusters)
        distAndClusters = getMin(distances)
        dist = distAndClusters[0][0]
        cluster1 = distAndClusters[1][0]
        cluster2 = distAndClusters[1][1]

        if (cluster1.clusterNum < cluster2.clusterNum):
            dendogram[i][0] = cluster1.clusterNum
            dendogram[i][1] = cluster2.clusterNum
        else:
            dendogram[i][0] = cluster2.clusterNum
            dendogram[i][1] = cluster1.clusterNum
        dendogram[i][2] = dist
        dendogram[i][3] = cluster1.numObj + cluster2.numObj
        newCluster = mergeClusters(cluster1, cluster2, n + i)
        clusters.remove(cluster1)
        clusters.remove(cluster2)
        clusters.append(newCluster)
    return dendogram


def imshow_hac(Z):
    dendo = dendrogram(Z)
    plt.show()

Z=linkage([calc_features(row) for row in load_data('Pokemon.csv')][:20],method='complete')
print(Z)