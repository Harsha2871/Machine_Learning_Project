#Import Libraries
import numpy as np
import pandas as pd
#import seaborn as sns
from randomForest import randomForest
from bagging import bagging
import time

from functools import partial

import knapsack
import genetic


population, generations = genetic.run_evolution(
		populate_func=partial(genetic.generate_population, size=10, genome_length=30),
		fitness_func=partial(knapsack.fitness,feature_limit=15),
		fitness_limit=0.88,
		generation_limit=50
	)

print(population[0])

X=knapsack.X
Y=knapsack.Y


start1=time.time()
print("accuracy when all features are considered:",randomForest(X,Y))
end1=time.time()
print('\n')
print("time taken to train the model with all the features:",end1-start1)

print('\n\n')

selected_features=[]
for i, choice in enumerate(population[0]):
	if choice == 1:
		selected_features.append(i)

start2=time.time()
print("accuracy when dominant features are selected using Genetic algorithm:", randomForest(X[:, selected_features], Y))
end2=time.time()
print('\n')
print("time to taken to train the model with only dominant features:",end2-start2)

print("dominat features selected by genetic algorithm: ")
for i,choice in enumerate(population[0]):
	if choice==1:
		print(knapsack.features[i])








