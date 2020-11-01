#!/usr/bin/python
# -*- coding: utf-8 -*-

import math
import random
from collections import namedtuple
import numpy as np
#from visualize import plot
import matplotlib as plt
import sys


MUTATION_RATE = 60
MUTATION_REPEAT_COUNT = 2
#WEAKNESS_THRESHOLD = 50000

Point = namedtuple("Point", ['x', 'y'])
#----------------------------------------------------------------------------
class Genome():
    chromosomes = []
    fitness = 0


def CreateNewPopulation(size,citySize,points):
    population = []
    for x in range(size):
        newGenome = Genome()
        newGenome.chromosomes = random.sample(range(0, citySize), citySize)
        #newGenome.chromosomes.insert(0, 0)
        f_city=newGenome.chromosomes[0]
        newGenome.chromosomes.append(f_city)
        newGenome.fitness = Evaluate(newGenome.chromosomes,points)
        population.append(newGenome)
    return population

def distance(point1, point2):
    return math.sqrt((point1.x - point2.x)**2 + (point1.y - point2.y)**2)

def length(point1, point2):
    return math.sqrt((point1.x - point2.x)**2 + (point1.y - point2.y)**2)

def Evaluate(chromosomes,points):
    calculatedFitness = 0
    for i in range(len(chromosomes) - 1):
        p1 = points[chromosomes[i]]
        p2 = points[chromosomes[i + 1]]
        calculatedFitness += distance(p1, p2)
    calculatedFitness = np.round(calculatedFitness, 2)
    return calculatedFitness

def findBestGenome(population):
    allFitness = [i.fitness for i in population]
    bestFitness = min(allFitness)
    return population[allFitness.index(bestFitness)]

# In K-Way tournament selection, we select K individuals
# from the population at random and select the best out
# of these to become a parent. The same process is repeated
# for selecting the next parent.

def TournamentSelection(population, k):
    selected = [population[random.randrange(0, len(population))] for i in range(k)]
    bestGenome = findBestGenome(selected)
    return bestGenome


def Reproduction(population,points):
    population.sort(key =lambda f: f.fitness)
    if len(population) > 20000:
        del population[15000:]
    '''parent1 = TournamentSelection(population, 5).chromosomes
    parent2 = TournamentSelection(population, 3).chromosomes
    while parent1 == parent2:
        parent2 = TournamentSelection(population, 20).chromosomes'''
    parent1=population[0].chromosomes
    parent2=population[1].chromosomes

    return OrderOneCrossover(parent1, parent2,points)
# Sample:
# parent1 = [0, 3, 8, 5, 1, 7, 12, 6, 4, 10, 11, 9, 2, 0]
# parent2 = [0, 1, 6, 3, 5, 4, 10, 2, 7, 12, 11, 8, 9, 0]
# child   = [0, 1, 3, 5, 2, 7, 12, 6, 4, 10, 11, 8, 9, 0]
def OrderOneCrossover(parent1, parent2,points):
    size = len(parent1)
    child = [-1] * size
    #print(parent1)
    #print(parent2)
    point=random.randrange(0, size)
    for i in range(0,point):
        child[i]=parent1[i]
    ind=point
    #print(start,end)
    for x in parent2:
        if x not in child:
            child[ind]=x
            ind+=1
    child[size-1]=child[0]
    #print(child)
            
    if random.randrange(0, 100) < MUTATION_RATE:
        #print("Mutated")
        child = SwapMutation(child)

    # Create new genome for child
    newGenome = Genome()
    newGenome.chromosomes = child
    newGenome.fitness = Evaluate(child,points)
    return newGenome

# Sample:
# Chromosomes =         [0, 3, 8, 5, 1, 7, 12, 6, 4, 10, 11, 9, 2, 0]
# Mutated chromosomes = [0, 11, 8, 5, 1, 7, 12, 6, 4, 10, 3, 9, 2, 0]


def SwapMutation(chromo):
    for x in range(MUTATION_REPEAT_COUNT):
        p1, p2 = [random.randrange(1, len(chromo) - 1) for i in range(2)]
        while p1 == p2:
            p2 = random.randrange(1, len(chromo) - 1)
        log = chromo[p1]
        chromo[p1] = chromo[p2]
        chromo[p2] = log
    return chromo
    
def GA_TSP(points,popSize,maxGeneration):
    allBestFitness = []
    citySize=len(points)
    population = CreateNewPopulation(popSize,citySize,points)
    generation = 0
    if citySize<100:
        WEAKNESS_THRESHOLD = 800
    elif citySize<200:
        WEAKNESS_THRESHOLD = 30000
    elif citySize<300:
        WEAKNESS_THRESHOLD = 40000
    elif citySize<1000:
        WEAKNESS_THRESHOLD = 50000
    elif citySize<1000:
        WEAKNESS_THRESHOLD = 500000
    else:
        WEAKNESS_THRESHOLD = 1000000
    while generation < maxGeneration:
        generation += 1

        for i in range(int(popSize / 2)):
            # Select parent, make crossover and
            # after, append in population a new child
            population.append(Reproduction(population,points))
        

        # Kill weakness person
        for genom in population:
            if genom.fitness > WEAKNESS_THRESHOLD:
                population.remove(genom)
        '''population.sort(key =lambda f: f.fitness)
        l=len(population)
        st=int(0.6*l)
        #newpop=[]
        #population.remove(st,l)
        #population=newpop
        del population[st:l]'''
        averageFitness = round(np.sum([genom.fitness for genom in population]) / len(population), 2)
        bestGenome = findBestGenome(population)
        print("\n" * 2)
        print("Generation: {0}\nPopulation Size: {1}\t Average Fitness: {2}\nBest Fitness: {3}"
              .format(generation, len(population), averageFitness,
                      bestGenome.fitness))

        allBestFitness.append(bestGenome.fitness)

    # Visualize
    #plt.plot(generation, allBestFitness, bestGenome, points)
    le=len(bestGenome.chromosomes)
    return (bestGenome.fitness,0,bestGenome.chromosomes[:le-1])
    


def solve_it(input_data):
    # Modify this code to run your optimization algorithm

    # parse the input
    lines = input_data.split('\n')

    nodeCount = int(lines[0])

    points = []
    for i in range(1, nodeCount+1):
        line = lines[i]
        parts = line.split()
        points.append(Point(float(parts[0]), float(parts[1])))

    #-----------------Genetic Algorithm-----------
    #obj,opt,solution=GA_TSP(points,750,500)
    #---------------My greedy approach----------
    obj=0
    solution=[]
    if nodeCount<1000:
        obj,solution=greedy(points)
        if nodeCount<500:
            new_solution=two_opt(solution,points)
            new_obj=tspLength(new_solution,points)
            if new_obj < obj:
                solution=new_solution
                obj=new_obj
    elif nodeCount>1500:
        solution=random.sample(range(nodeCount),nodeCount)
        #obj,opt,solution=GA_TSP(points,750,500)
        obj=tspLength(solution,points)
    #print(nodeCount)
    #-----------------
    
    '''# build a trivial solution
    # visit the nodes in the order they appear in the file
    solution = range(0, nodeCount)

    # calculate the length of the tour
    obj = length(points[solution[-1]], points[solution[0]])
    for index in range(0, nodeCount-1):
        obj += length(points[solution[index]], points[solution[index+1]])'''
    
    # prepare the solution in the specified output format
    output_data = '%.2f' % obj + ' ' + str(0) + '\n'
    output_data += ' '.join(map(str, solution))
    #distance_matrix(points)

    return output_data
#-----------Distance Matrix--------------

def distance_matrix(points):
    dist=np.zeros([len(points),len(points)])
    #dist = [[0 for w in range(len(points))] 
    #   for i in range(len(points))]
    for i,p in enumerate(points):
        for j, q in enumerate(points):
            if i==j:
                continue
            elif i<j:
                dist[i][j]=length(p,q)
            else:
                dist[i][j]=dist[j][i]

    return dist
def find_closest_city(start,visited,dist_list,no_of_cities):
    city=0
    min=sys.maxsize
    for i in range(0,no_of_cities):
        if i!=start and i not in visited and min > dist_list[i]:
            min=dist_list[i]
            city=i
    #visited.append(city)
    return city
#-------------------Nearest Neighbour ------------------
def greedy(points):
    dist=distance_matrix(points)
    no_of_cities=len(points)
    min_path=sys.maxsize
    sol=[]
    for i in range(no_of_cities):
        start=i
        path_length=0
        visited=[]
        visited.append(start)
        counter=1
        #print(counter)
        #print(dist[start])
        #print(counter)
        while counter < no_of_cities:
            next_city=find_closest_city(start,visited,dist[start],no_of_cities)
            visited.append(next_city)
            start=next_city
            counter+=1
        #print(visited)
        #print(len(visited))
        for i in range(0,len(visited)-1):
            #print(visited[i],visited[i+1])
            path_length+=dist[visited[i]][visited[i+1]]
        path_length+=dist[visited[0]][visited[len(visited)-1]]
        #print(path_length)
        if min_path>path_length:
            min_path=path_length
            sol=visited
    print("--------------------------------------")
    return min_path,sol

def tspLength(cycle, points):
    return sum([length(points[cycle[i - 1]], points[cycle[i]]) 
                for i in range(len(points))])


#------------------------2 opt--------------
def two_opt(route,points):
     best = route
     improved = True
     while improved:
          improved = False
          for i in range(1, len(route)-2):
               for j in range(i+1, len(route)):
                    if j-i == 1: continue # changes nothing, skip then
                    new_route = route[:]
                    new_route[i:j] = route[j-1:i-1:-1] # this is the 2woptSwap
                    if tspLength(new_route,points) < tspLength(best,points):  # what should cost be?
                         best = new_route
                         improved = True
          route = best
     return best
    
import sys

if __name__ == '__main__':
    import sys
    if len(sys.argv) > 1:
        file_location = sys.argv[1].strip()
        with open(file_location, 'r') as input_data_file:
            input_data = input_data_file.read()
        print(solve_it(input_data))
    else:
        print('This test requires an input file.  Please select one from the data directory. (i.e. python solver.py ./data/tsp_51_1)')

