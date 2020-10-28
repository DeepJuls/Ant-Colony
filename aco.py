#!/usr/bin/env python 

import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import normalize
from subprocess import Popen

class ACO():

	def __init__(self,cost_array,task_list,oper_list,
					beta,num_ant,generations,evap_ratio,std_counter,penalty,monitor):

		self.cost_array = cost_array
		self.task_list = task_list
		self.oper_list = oper_list

		self.num_task = len(self.task_list)
		self.num_oper = len(self.oper_list)

		self.pher_array = [[1 for col in range(self.num_oper)] for row in range(self.num_oper)]
		self.init_pher = [1]*self.num_oper

		self.best_combination = None
		self.best_fitness = float('inf')

		self.beta = beta
		self.num_ant = num_ant
		self.evap_ratio = evap_ratio
		self.generations = generations
		self.std_counter = std_counter
		self.penalty = penalty
		self.monitor = monitor

		self.Main()

	def Fitness(self,chromosome):
		fit = 0
		free_time = self.oper_list[:]
		for i in range(len(chromosome)):
			fit += self.cost_array[chromosome[i]][i]
			free_time[chromosome[i]] -= self.task_list[i]
		for item in free_time:
			if item < 0:
				fit *= self.penalty
		return fit

	def Initial_distribution(self):
		density = []
		for i in range(self.num_oper):
			density.append(self.init_pher[i]*(1/self.cost_array[i][0])**self.beta)
		total = sum(density)
		distribution = []
		for i in range(len(density)):
			distribution.append(sum(density[:i+1])/total)
		return distribution

	def Transition(self,current,task):
		density = []
		for i in range(self.num_oper):
			a = self.pher_array[current][i]
			b = (1/self.cost_array[i][task])**self.beta
			density.append(a*b)
		total = sum(density)
		distribution = []
		for i in range(len(density)):
			distribution.append(sum(density[:i+1])/total)
		return distribution

	def Roulette(self,distribution):
		u = np.random.uniform()
		for i in range(len(distribution)):
			if distribution[i] >= u:
				return i

	def Generate_ants(self):
		ants = []
		for i in range(self.num_ant):
			ant = [self.Roulette(self.Initial_distribution())]
			for j in range(1,self.num_task):
				ant.append(self.Roulette(self.Transition(ant[-1],j)))
			ants.append(ant)
		return(ants)

	def Evaporate_pheromones(self):
		for i in range(self.num_oper):
			self.init_pher[i] *= (1-self.evap_ratio)
		for i in range(self.num_oper):
			for j in range(self.num_oper):
				self.pher_array[i][j] *= (1-self.evap_ratio)
		return

	def Actualize_pheromones(self,ants):
		for ant in ants:
			ant_fitness = self.Fitness(ant)
			self.init_pher[ant[0]] += 1/ant_fitness
			for i in range(len(ant)-1):
				self.pher_array[ant[i]][ant[i+1]] += 1/ant_fitness
		return 

	def Evaluate_ants(self,ants):
		fitness_values = []
		for ant in ants:
			ant_fitness = self.Fitness(ant)
			fitness_values.append(ant_fitness)
			if ant_fitness < self.best_fitness:
				self.best_fitness = ant_fitness
				self.best_combination = ant
		return fitness_values

	def Best_uniqs(self,ants):
		unique_data = [list(x) for x in set(tuple(x) for x in ants)]
		print("Combinaciones\tFitness")
		for uniqs in unique_data:
			print("{}\t{}".format(str(uniqs),str(self.Fitness(uniqs))))

	def Plot_parameters(self,means,stds,bests):

		fig, axs = plt.subplots(3, sharex=True, sharey=True)
		axs[0].plot(means,"r")
		axs[0].set_title("Media")
		axs[1].plot(stds,"b")
		axs[1].set_title("Desviación típica")
		axs[2].plot(bests,"g")
		axs[2].set_title("Mejor Fitness")
		plt.show()

	def Plot_graph(self):
		G=nx.MultiDiGraph()
		pos = {0:(100,100),1:(0,0)}
		G.add_node(0)
		G.add_node(1)
		G.add_edge(0,0)
		G.add_edge(0,1)
		G.add_edge(1,0)
		plt.show()

	def Plot_map(self,iteration):
		normed_matrix = []
		total = sum(self.init_pher)
		for item in self.init_pher:
			normed_matrix.append(item/total)
		x = [10,15,20,25,30]
		y = [45,0,80,0,45]
		plt.ylim(bottom=-20,top=100)
		plt.xlim(left=0,right=40)
		plt.xticks([])
		plt.yticks([])
		for i in range(len(x)):
			plt.scatter(x[i],y[i],s=2000*normed_matrix[i],c="b")
		for i in range(len(x)):
			plt.annotate(str(i+1), (x[i]-.6,y[i]-1.5),size=20,color="black")

		def drawArrow(A, B,alpha):
			
			plt.arrow(A[0], A[1], B[0] - A[0], B[1] - A[1], head_width=alpha*4, length_includes_head=True,alpha = alpha**3, width = alpha*2)

		normed_matrix = normalize(self.pher_array, axis=1, norm='l1')
		for i in range(5):
			for j in range(5):
				drawArrow([x[i],y[i]],[x[j],y[j]],normed_matrix[i][j])
		plt.savefig(fname="gif/"+str(iteration)+".png")


	def Main(self):
		Popen(["rm","-r","-f","gif"])
		Popen(["mkdir","gif"])
		std = float('inf')
		std_counter = self.std_counter
		gen = 0
		means = []
		stds = []
		bests = []
		for gen in range(self.generations):
			if std == 0:
				std_counter +=1
				if std_counter == self.std_counter:
					self.Best_uniqs(ants)
					self.Plot_parameters(means,stds,bests)
					self.Plot_map(gen)
					return
			else:
				std_counter = 0
			ants = self.Generate_ants()
			fitness_values = self.Evaluate_ants(ants)
			mean = np.mean(fitness_values)
			std = np.std(fitness_values)
			means.append(mean)
			stds.append(std)
			bests.append(self.best_fitness)
			self.Evaporate_pheromones()
			self.Actualize_pheromones(ants)
			if gen%self.monitor == 0:
				print("Generación: {}\nCoste medio: {}\n".format(str(gen),str(sum(fitness_values)/len(fitness_values))))
				self.Plot_map(gen)
		self.Best_uniqs(ants)
		self.Plot_parameters(means,stds,bests)
		return

cost_array = np.array([
					[2,3,4,1],
					[3,2,3,2],
					[2,2,1,2],
					[3,3,3,3],
					[2,1,2,2],
					])
task_list =    [3,2,2,3]
oper_list =    [4,5,3,4,4]

ACO(cost_array = cost_array,
	task_list = task_list,
	oper_list = oper_list,
	beta = 2,
	num_ant = 25,
	evap_ratio = 0.1,
	generations = 200,
	std_counter = 50,
	penalty = 5,
	monitor = 20)
