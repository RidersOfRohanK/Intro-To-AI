import heapq
import numpy
import copy


def get_manhattan_distance(from_state, to_state=[1, 2, 3, 4, 5, 6, 7, 0, 0]):
    """
    TODO: implement this function. This function will not be tested directly by the grader. 

    INPUT: 
        Two states (if second state is omitted then it is assumed that it is the goal state)

    RETURNS:
        A scalar that is the sum of Manhattan distances for all tiles.
    """
    ansPuzzle = { to_state[0]:[0,0], to_state[1]:[0,1],to_state[2]:[0,2],
                  to_state[3]:[1,0], to_state[4]:[1,1], to_state[5]:[1,2], to_state[6]:[2,0],
                  to_state[7]:[2,1], to_state[8]:[2,2]}

    totalDist = 0
    for i in range(len(from_state)):
        if from_state[i] !=0:
            point = ansPuzzle[from_state[i]]
            puzzleLoc = [int(i / 3), i % 3]
            totalDist += (abs(puzzleLoc[0] - point[0]) + abs(puzzleLoc[1] - point[1]))
    return totalDist


def print_succ(state):
    """
    TODO: This is based on get_succ function below, so should implement that function.

    INPUT: 
        A state (list of length 9)

    WHAT IT DOES:
        Prints the list of all the valid successors in the puzzle. 
    """

    succ_states = get_succ(state)

    for succ_state in succ_states:
        print(succ_state, "h={}".format(get_manhattan_distance(succ_state)))




def get_succ(state):
    """
    TODO: implement this function.

    INPUT: 
        A state (list of length 9)

    RETURNS:
        A list of all the valid successors in the puzzle (don't forget to sort the result as done below). 
    """

    puzzle = numpy.reshape(state,(3,3)).tolist()
    moves = [[1, 0], [0, 1], [-1, 0], [0, -1]]
    emptySlots = list()
    result = []

    for i in range(3):
        for j in range(3):
            if puzzle[i][j] == 0:
                emptySlots.append([i, j])

    for move in moves:
        for emptySlot in emptySlots:
            moveX = emptySlot[0]+move[0]
            moveY = emptySlot[1] + move[1]
            if moveX>=0 and moveX<=2 and moveY>=0 and moveY<=2:
                temp = copy.deepcopy(puzzle)

                curr = temp[emptySlot[0]][emptySlot[1]]
                temp[emptySlot[0]][emptySlot[1]] = temp[moveX][moveY] #make the move
                temp[moveX][moveY] = curr #swap the slots

                if (curr != temp[emptySlot[0]][emptySlot[1]]):#check for swapping zeros
                    temp = numpy.array(temp).flatten().tolist()
                    result.append(temp)
    result = sorted(result)
    return result


def solve(state, goal_state=[1, 2, 3, 4, 5, 6, 7, 0, 0]):
    """
    TODO: Implement the A* algorithm here.

    INPUT: 
        An initial state (list of length 9)

    WHAT IT SHOULD DO:
        Prints a path of configurations from initial state to goal state along  h values, number of moves, and max queue number in the format specified in the pdf.
    """

    pq = list()
    dist = get_manhattan_distance(state,goal_state)
    heapq.heappush(pq, (dist, state, (0, dist, -1)))
    visited = list()
    visited.append(state)
    checked = list()
    finalPath = list()
    count = 0
    maxQueueLength = 1
    while (len(pq) != 0):
        count += 1

        if len(pq)>maxQueueLength:
            maxQueueLength = len(pq)

        current = heapq.heappop(pq)
        checked.append(current)
        curr_state = current[1]

        if get_manhattan_distance(curr_state,goal_state) == 0:#then we have found the end

            temp = checked[len(checked) - 1]
            while temp[2][2] != -1: #until initial
                finalPath.append(temp)
                temp = checked[temp[2][2]]

            finalPath.append(temp)
            finalPath.reverse()
            for choice in finalPath:
                print(str(choice[1])+" h=" + str(choice[2][1])+" moves: "+str(choice[2][0]))
            print("Max queue length: "+str(maxQueueLength))
            return finalPath

        neighbors = get_succ(curr_state)
        for neighbor in neighbors:
            check = []
            for toCheck in checked:
                if neighbor == toCheck[1]:
                    check.append(toCheck)
            dist = get_manhattan_distance(neighbor,goal_state)#check dist to final
            moves = current[2][0]+1 # add to current number of moves
            if neighbor in visited or (len(check)!=0 and moves > check[0][2][0]):
                continue
            else:
                cost = moves + dist
                heapq.heappush(pq, (cost, neighbor, (moves, dist, checked.index(current))))#add to the heap
        visited.append(curr_state)


if __name__ == "__main__":
    """
    Feel free to write your own test code here to exaime the correctness of your functions. 
    Note that this part will not be graded.
    """
    # print_succ([2,5,1,4,0,6,7,0,3])
    # print()
    #
    # print(get_manhattan_distance([2,5,1,4,0,6,7,0,3], [1, 2, 3, 4, 5, 6, 7, 0, 0]))
    # print()
    #
    # # solve([2,5,1,4,0,6,7,0,3])
    # solve([4, 3, 0, 5, 1, 6, 7, 2, 0])
    # print()

    print_succ([3, 4, 6, 0, 0, 1, 7, 2, 5])
    print_succ([6, 0, 0, 3, 5, 1, 7, 2, 4])
    print_succ([0, 4, 7, 1, 3, 0, 6, 2, 5])
    print_succ([5, 2, 3, 0, 6, 4, 7, 1, 0])
    print_succ([1, 7, 0, 6, 3, 2, 0, 4, 5])
    solve([3, 4, 6, 0, 0, 1, 7, 2, 5])
    solve([6, 0, 0, 3, 5, 1, 7, 2, 4])
    solve([0, 4, 7, 1, 3, 0, 6, 2, 5])
    solve([5, 2, 3, 0, 6, 4, 7, 1, 0])
    solve([1, 7, 0, 6, 3, 2, 0, 4, 5])