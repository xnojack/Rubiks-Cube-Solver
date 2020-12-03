import numpy as np
import sys
import threading
import random
from multiprocessing import Process, Queue
import time
import json
import os
import math
import matplotlib.pyplot as plt
import csv

#Global variables adjust according to what's needed
global Pm
Pm = .01
global Pc
Pc = .9
global popsize
popsize = 200
global numGen
numGen = 100
global alpha
alpha = 1
global share
share = .1

#Code for the plotting function in python
plt.ion()
global fig
fig = plt.figure()
ax = fig.add_subplot(111)
global line1
global line2
global line3
line1, = ax.plot([0,1], [0,1], 'b-')
line2, = ax.plot([0,1], [0,1], 'r.')
line3, = ax.plot([0,1], [0,1], 'go')

#Plots the current population according to m1
def m1Plot(pop):
	x1 = np.arange(0,1,0.001)   # start,stop,step
	y1 = []
	for num in x1:
		y1.append(pow(math.sin(5*math.pi*num),6))
	line1.set_ydata(y1)
	line1.set_xdata(x1)

	x2 = []
	y2 = []
	for i in pop:
		x2.append(bin2Fac(i))
		y2.append(pow(math.sin(5*math.pi*bin2Fac(i)),6))
	line2.set_ydata(y2)
	line2.set_xdata(x2)

	#plt.xlabel('x') 
	# Labeling the Y-axis 
	#plt.ylabel('M2') 
	# Give a title to the graph
	#plt.title('Sin(x) and Cos(x) on the same graph') 
	  
	# Show a legend on the plot 
	fig.canvas.draw()
	fig.canvas.flush_events()

#Plots the initial and final populations according to m1
def m1PlotFinal(initial,final):
	x1 = np.arange(0,1,0.001)   # start,stop,step
	y1 = []
	for num in x1:
		y1.append(pow(math.sin(5*math.pi*num),6))
	line1.set_ydata(y1)
	line1.set_xdata(x1)

	x2 = []
	y2 = []
	for i in initial:
		x2.append(bin2Fac(i))
		y2.append(pow(math.sin(5*math.pi*bin2Fac(i)),6))
	line2.set_ydata(y2)
	line2.set_xdata(x2)

	x3 = []
	y3 = []
	for i in final:
		x3.append(bin2Fac(i))
		y3.append(pow(math.sin(5*math.pi*bin2Fac(i)),6))
	line3.set_ydata(y3)
	line3.set_xdata(x3)

	#plt.xlabel('x') 
	# Labeling the Y-axis 
	#plt.ylabel('M2') 
	# Give a title to the graph
	#plt.title('Sin(x) and Cos(x) on the same graph') 
	  
	# Show a legend on the plot 
	fig.canvas.draw()
	fig.canvas.flush_events()

#Plots the current population according to m1
def m4Plot(pop):
	x1 = np.arange(0,1,0.001)   # start,stop,step
	y1 = []
	for num in x1:
		y1.append(math.exp(-2*math.log(2)*pow((num-.08)/.854,2))*pow(math.sin(5*math.pi*(pow(num,.75)-.05)),6))
	line1.set_ydata(y1)
	line1.set_xdata(x1)

	x2 = []
	y2 = []
	for i in pop:
		x2.append(bin2Fac(i))
		y2.append(math.exp(-2*math.log(2)*pow((bin2Fac(i)-.08)/.854,2))*pow(math.sin(5*math.pi*(pow(bin2Fac(i),.75)-.05)),6))
	line2.set_ydata(y2)
	line2.set_xdata(x2)

	#plt.xlabel('x') 
	# Labeling the Y-axis 
	#plt.ylabel('M2') 
	# Give a title to the graph
	#plt.title('Sin(x) and Cos(x) on the same graph') 
	  
	# Show a legend on the plot 
	fig.canvas.draw()
	fig.canvas.flush_events()

#Plots the initial and final populations according to m1
def m4PlotFinal(initial,final):
	x1 = np.arange(0,1,0.001)   # start,stop,step
	y1 = []
	for num in x1:
		y1.append(math.exp(-2*math.log(2)*pow((num-.08)/.854,2))*pow(math.sin(5*math.pi*(pow(num,.75)-.05)),6))
	line1.set_ydata(y1)
	line1.set_xdata(x1)

	x2 = []
	y2 = []
	for i in initial:
		x2.append(bin2Fac(i))
		y2.append(math.exp(-2*math.log(2)*pow((bin2Fac(i)-.08)/.854,2))*pow(math.sin(5*math.pi*(pow(bin2Fac(i),.75)-.05)),6))
	line2.set_ydata(y2)
	line2.set_xdata(x2)

	x3 = []
	y3 = []
	for i in final:
		x3.append(bin2Fac(i))
		y3.append(math.exp(-2*math.log(2)*pow((bin2Fac(i)-.08)/.854,2))*pow(math.sin(5*math.pi*(pow(bin2Fac(i),.75)-.05)),6))
	line3.set_ydata(y3)
	line3.set_xdata(x3)

	#plt.xlabel('x') 
	# Labeling the Y-axis 
	#plt.ylabel('M2') 
	# Give a title to the graph
	#plt.title('Sin(x) and Cos(x) on the same graph') 
	  
	# Show a legend on the plot 
	fig.canvas.draw()
	fig.canvas.flush_events()

#Displays the avg min, avg and max values for the found optima
def plotAvg(avg):
	line1, = ax.plot([0,100], [0,1], 'g-')
	line2, = ax.plot([0,100], [0,1], 'r-')
	line3, = ax.plot([0,100], [0,1], 'b-')

	x = np.arange(0,100,1)
	y1 = []
	y2 = []
	y3 = []
	for i in avg:
		y1.append(i[0])
		y2.append(i[1])
		y3.append(i[2])

	line1.set_ydata(y1)
	line1.set_xdata(x)
	line2.set_ydata(y2)
	line2.set_xdata(x)
	line3.set_ydata(y3)
	line3.set_xdata(x)
	# Show a legend on the plot 
	fig.canvas.draw()
	fig.canvas.flush_events()

#Clears the plotting area
def plotReset():
	line1.set_ydata([])
	line1.set_xdata([])
	line2.set_ydata([])
	line2.set_xdata([])
	line3.set_ydata([])
	line3.set_xdata([])
	fig.canvas.draw()
	fig.canvas.flush_events()

#Generates a random population, is just the x value
def initializePop(size):
	pop = []
	for i in range(0,size):
		pop.append([])
		for y in range(0,10):
			pop[i].append(random.randint(0,1))
	return pop

#Converts a binary fraction to a decimal fraction
def bin2Fac(individual):
	num = 0
	for i in range(0,len(individual)):
		if(individual[i]==1):
			num = num + pow(2,-1*(i+1))
	return num

#M1 fitness function, automatically does an insertion sort as well
def m1Fit(pop):
	ranked = []
	for i in pop:
		num = bin2Fac(i)
		fitness = pow(math.sin(5*math.pi*num),6)
		if(len(ranked)==0):
			ranked.append([fitness,i])
		else:
			inserted = False
			for y in range(0,len(ranked)):
				if(fitness > ranked[y][0]):
					ranked.insert(y,[fitness,i])
					inserted = True
					break
			if not inserted:
				ranked.append([fitness,i])
	return ranked

#M4 fitness function also does an insertion sort
def m4Fit(pop):
	ranked = []
	for i in pop:
		num = bin2Fac(i)
		fitness = math.exp(-2*math.log(2)*pow((num-.08)/.854,2))*pow(math.sin(5*math.pi*(pow(num,.75)-.05)),6)
		if(len(ranked)==0):
			ranked.append([fitness,i])
		else:
			inserted = False
			for y in range(0,len(ranked)):
				if(fitness > ranked[y][0]):
					ranked.insert(y,[fitness,i])
					inserted = True
					break
			if not inserted:
				ranked.append([fitness,i])
	return ranked

#Sharing fitness adjustment function. Calculates a new fitness according to number of individuals in niche
def sharing(ranked):
	#Loop through each element to edit it's fitness
	for i in range(len(ranked)):
		#Share sum
		shSum = 0
		#Loop through each element looking for neighbours
		for y in range(len(ranked)):
			if i == y: continue
			#Calculate the distance between the 2 points
			distance = abs(bin2Fac(ranked[i][1])-bin2Fac(ranked[y][1]))
			#Based off the share variable, if we're too far skip
			if(distance >= share): continue
			#share sum is then calculated
			shSum+=1-pow(distance/share,alpha)
		#Skip divide by zero
		if(shSum == 0): continue
		#Divide by the share sum
		ranked[i][0] = ranked[i][0]/shSum
	return ranked

#Normalizes the fitness so that they add up to equal 1
def normalize(ranked):
	total = 0
	norm = list(ranked)
	for i in ranked:
		total += i[0]
	for i in norm:
		i[0]/=total
	return norm

#Takes first 3 elements from p1, second 3 from p2 and last 4 from p1 to generate a child
def staticCrossover(p1,p2):
	child = []
	for i in range(0,3):
		child.append(p1[i])
	for i in range(3,6):
		child.append(p2[i])
	for i in range(6,10):
		child.append(p1[i])
	return child

#Mutates and individual child according to the mutation rate
def mutate(child):
	if(random.random() < Pm):
		rand = random.randint(0,9)
		if child[rand] == 0:
			child[rand] = 1
		else:
			child[rand] = 0
	return child

#Deterministic crowding method to keep the better/closer parent/child
def deterministicCrowding(pool):
	children = []
	
	for i in range(0,len(pool)):
		distanceThresh = .01
		p1 = pool[random.randint(0,len(pool)-1)]
		p2 = pool[random.randint(0,len(pool)-1)]
		j=0
		while p2 == p1:# and abs(bin2Fac(p1)-bin2Fac(p2))>=distanceThresh:
			p2 = pool[random.randint(0,len(pool)-1)]
			j+=1
			if(j%10 == 0):
				distanceThresh+=.001

		c1 = mutate(staticCrossover(p1,p2))
		c2 = mutate(staticCrossover(p2,p1))

		p1Fit=0
		p2Fit=0
		c1Fit=0
		c2Fit=0

		if sys.argv[2] == "m1":
			num = bin2Fac(p1)
			p1Fit = pow(math.sin(5*math.pi*num),6)
			num = bin2Fac(p2)
			p2Fit = pow(math.sin(5*math.pi*num),6)
			num = bin2Fac(c1)
			c1Fit = pow(math.sin(5*math.pi*num),6)
			num = bin2Fac(c2)
			c2Fit = pow(math.sin(5*math.pi*num),6)
		elif sys.argv[2] == "m4":
			num = bin2Fac(p1)
			p1Fit = math.exp(-2*math.log(2)*pow((num-.1)/.8,2))*pow(math.sin(5*math.pi*(pow(num,.75)-.05)),6)
			num = bin2Fac(p2)
			p2Fit = math.exp(-2*math.log(2)*pow((num-.1)/.8,2))*pow(math.sin(5*math.pi*(pow(num,.75)-.05)),6)
			num = bin2Fac(c1)
			c1Fit = math.exp(-2*math.log(2)*pow((num-.1)/.8,2))*pow(math.sin(5*math.pi*(pow(num,.75)-.05)),6)
			num = bin2Fac(c2)
			c2Fit = math.exp(-2*math.log(2)*pow((num-.1)/.8,2))*pow(math.sin(5*math.pi*(pow(num,.75)-.05)),6)

		if abs(bin2Fac(p1)-bin2Fac(c1))+abs(bin2Fac(p2)-bin2Fac(c2))<=abs(bin2Fac(p1)-bin2Fac(c2))+abs(bin2Fac(p2)-bin2Fac(c1)):
			if(c1Fit > p1Fit):
				children.append(c1)
			else:
				children.append(p1)
			if(c2Fit > p2Fit):
				children.append(c2)
			else:
				children.append(p2)
		else:
			if(c2Fit > p1Fit):
				children.append(c2)
			else:
				children.append(p1)
			if(c1Fit > p2Fit):
				children.append(c1)
			else:
				children.append(p2)


	# print(len(children))
	return children

#Attempt at preventing cross niche breeding, doesn't work
def noCrossMating(pool):
	children = []
	
	for i in range(0,len(pool)):
		distanceThresh = .01
		p1 = pool[random.randint(0,len(pool)-1)]
		p2 = pool[random.randint(0,len(pool)-1)]
		j=0
		while p2 == p1 and abs(bin2Fac(p1)-bin2Fac(p2))>=distanceThresh:
			p2 = pool[random.randint(0,len(pool)-1)]
			j+=1
			if(j%10 == 0):
				distanceThresh+=.001
		if(random.random()<=Pc):
			children.append(staticCrossover(p1,p2))
			children.append(staticCrossover(p2,p1))
		else:
			children.append(p1)
			children.append(p2)

	# print(len(children))
	return children

#Randomly selects 2 parents from the pool to mate
def randomMating(pool):
	children = []
	for i in range(0,len(pool)):
		p1 = pool[random.randint(0,len(pool)-1)]
		p2 = pool[random.randint(0,len(pool)-1)]
		while p2 == p1:
			p2 = pool[random.randint(0,len(pool)-1)]
		if(random.random()<=Pc):
			children.append(staticCrossover(p1,p2))
			children.append(staticCrossover(p2,p1))
		else:
			children.append(p1)
			children.append(p2)

	# print(len(children))
	return children

#Adds up all the fitness in a ranked population
def sumFit(ranked):
	total = 0
	for i in ranked:
		total+=i[0]
	return total

#Stochastic Universal Sampling, tries to give them all a porportional chance
def SUS(ranked):
	f=sumFit(ranked)
	n=int(popsize/2)
	p=f/n
	start=random.random()*p
	pointers = []
	
	for i in range(n):
		pointers.append(start+i*p)

	pool = []
	for point in pointers:
		i=0
		while sumFit(ranked[:i]) <= point and i<len(ranked)-1:
			i+=1
		pool.append(ranked[i][1])

	return pool

#Mutates a group of children according to the mutation rate
def staticMutate(children):
	for child in children:
		if(random.random() < Pm):
			rand = random.randint(0,9)
			if child[rand] == 0:
				child[rand] = 1
			else:
				child[rand] = 0
	return children

#This just randomly picks out popsize/2 individuals for mating
def staticSelection(ranked):
	pool = []
	temp = normalize(ranked)

	while len(pool) < len(ranked)/2:
		i = random.randint(0,len(temp)-1)
		pool.append(temp[i][1])
		temp.pop(i)

	return pool

#Returns the min, max and avg of a population according to m1
def rankM1(pop):
	max = [0,[]]
	min = [1,[]]
	avg = 0
	fit = m1Fit(pop)

	for individual in fit:
		if individual[0] > max[0]:
			max[0] = individual[0]
			max[1] = individual[1]
		if individual[0] < min[0]:
			min[0] = individual[0]
			min[1] = individual[1]
		avg+=individual[0]

	avg/=popsize
	return [max,min,avg]

#Returns the min, max and avg of a population according to m4
def rankM4(pop):
	max = [0,[]]
	min = [1,[]]
	avg = 0
	fit = m4Fit(pop)

	for individual in fit:
		if individual[0] > max[0]:
			max[0] = individual[0]
			max[1] = individual[1]
		if individual[0] < min[0]:
			min[0] = individual[0]
			min[1] = individual[1]
		avg+=individual[0]

	avg/=popsize
	return [max,min,avg]

#Classic GA with M1 fitness
if sys.argv[1] == "classic" and sys.argv[2] == "m1":
	#Store data for csvs
	stats = [['run number','gen number','min','avg','max']]
	avg = []
	print("Running classic")
	#Make 10 runs
	for i in range(10):
		#Clear line3 (Final pop) from plot
		line3.set_ydata([])
		line3.set_xdata([])

		print("Run ",i+1)

		#Make the initial population
		population = initializePop(popsize)
		#Store the initial population
		initial = list(population)
		#Loop for numGen generations
		for j in range(numGen):
			#Get the stats of the current generation
			stat = rankM1(population)
			stats.append([i,j,stat[1][0],stat[2],stat[0][0]])
			#For storing the avg results
			if j < len(avg):
				avg[j][0]+=stat[1][0]
				avg[j][1]+=stat[2]
				avg[j][2]+=stat[0][0]
			else:
				avg.append([stat[1][0],stat[2],stat[0][0]])

			#Run the fitness of the population
			ranked = m1Fit(population)

			#Plot the population
			m1Plot(population)
			#time.sleep(.1)

			#Get the individuals who'll be mating
			pool = SUS(ranked)
			#Get the children from mating individuals
			children = noCrossMating(pool)
			#Mutate the group of children
			population = staticMutate(children)
		#Once out of the loop plot the final population
		m1PlotFinal(initial,population)
		#Save a pic of it
		plt.savefig("pictures/run "+str(i+1)+" Classic M1.png")
		#time.sleep(1)
	#Save the stats to their csv files
	myFile = open('data/Classic M1 raw.csv', 'w')
	with myFile:
	    writer = csv.writer(myFile)
	    writer.writerows(stats)
	myFile.close()
	for i in range(len(avg)):
		avg[i][0]/=10
		avg[i][1]/=10
		avg[i][2]/=10

	data = [['gen','min','avg','max']]
	for i in range(len(avg)):
		data.append([i,avg[i][0],avg[i][1],avg[i][2]])
	myFile = open('data/Classic M1 avg.csv', 'w')
	with myFile:
	    writer = csv.writer(myFile)
	    writer.writerows(data)
	myFile.close()

	#reset the plot and plot the avg for all 10 runs
	plotReset()
	plotAvg(avg)
	plt.savefig("pictures/Classic M1 Avg.png")
	time.sleep(10)
#Classic GA with M4 fitness
elif sys.argv[1] == "classic" and sys.argv[2] == "m4":
	stats = [['run number','gen number','min','avg','max']]
	avg = []
	print("Running classic")
	for i in range(10):
		line3.set_ydata([])
		line3.set_xdata([])
		print("Run ",i+1)
		population = initializePop(popsize)
		initial = list(population)
		for j in range(numGen):
			stat = rankM4(population)
			stats.append([i,j,stat[1][0],stat[2],stat[0][0]])
			if j < len(avg):
				avg[j][0]+=stat[1][0]
				avg[j][1]+=stat[2]
				avg[j][2]+=stat[0][0]
			else:
				avg.append([stat[1][0],stat[2],stat[0][0]])

			ranked = m4Fit(population)

			m4Plot(population)
			#time.sleep(.1)

			pool = SUS(ranked)
			children = noCrossMating(pool)
			population = staticMutate(children)
		m4PlotFinal(initial,population)
		plt.savefig("pictures/run "+str(i+1)+" Classic M4.png")
		#time.sleep(1)
	myFile = open('data/Classic M4 raw.csv', 'w')
	with myFile:
	    writer = csv.writer(myFile)
	    writer.writerows(stats)
	myFile.close()
	for i in range(len(avg)):
		avg[i][0]/=10
		avg[i][1]/=10
		avg[i][2]/=10

	data = [['gen','min','avg','max']]
	for i in range(len(avg)):
		data.append([i,avg[i][0],avg[i][1],avg[i][2]])
	myFile = open('data/Classic M4 avg.csv', 'w')
	with myFile:
	    writer = csv.writer(myFile)
	    writer.writerows(data)
	myFile.close()

	plotReset()
	plotAvg(avg)
	plt.savefig("pictures/Classic M4 Avg.png")
	time.sleep(10)
#Sharing GA with M1 fitness
elif sys.argv[1] == "sharing" and sys.argv[2] == "m1":
	stats = [['run number','gen number','min','avg','max']]
	avg = []
	print("Running sharing m1")
	for i in range(10):
		line3.set_ydata([])
		line3.set_xdata([])
		print("Run ",i+1)
		population = initializePop(popsize)
		initial = list(population)
		for j in range(numGen):
			stat = rankM1(population)
			stats.append([i,j,stat[1][0],stat[2],stat[0][0]])
			if j < len(avg):
				avg[j][0]+=stat[1][0]
				avg[j][1]+=stat[2]
				avg[j][2]+=stat[0][0]
			else:
				avg.append([stat[1][0],stat[2],stat[0][0]])

			ranked = sharing(m1Fit(population))

			m1Plot(population)
			#time.sleep(.1)

			pool = SUS(ranked)
			children = noCrossMating(pool)
			population = staticMutate(children)
		m1PlotFinal(initial,population)
		plt.savefig("pictures/run "+str(i+1)+" Sharing M1.png")
		#time.sleep(1)
	myFile = open('data/Sharing M1 raw.csv', 'w')
	with myFile:
	    writer = csv.writer(myFile)
	    writer.writerows(stats)
	myFile.close()
	for i in range(len(avg)):
		avg[i][0]/=10
		avg[i][1]/=10
		avg[i][2]/=10

	data = [['gen','min','avg','max']]
	for i in range(len(avg)):
		data.append([i,avg[i][0],avg[i][1],avg[i][2]])
	myFile = open('data/Sharing M1 avg.csv', 'w')
	with myFile:
	    writer = csv.writer(myFile)
	    writer.writerows(data)
	myFile.close()

	plotReset()
	plotAvg(avg)
	plt.savefig("pictures/Sharing M1 Avg.png")
	time.sleep(10)
#Sharing GA with M4 fitness
elif sys.argv[1] == "sharing" and sys.argv[2] == "m4":
	stats = [['run number','gen number','min','avg','max']]
	avg = []
	print("Running sharing m4")
	for i in range(10):
		line3.set_ydata([])
		line3.set_xdata([])
		print("Run ",i+1)
		population = initializePop(popsize)
		initial = list(population)
		for j in range(numGen):
			stat = rankM4(population)
			stats.append([i,j,stat[1][0],stat[2],stat[0][0]])
			if j < len(avg):
				avg[j][0]+=stat[1][0]
				avg[j][1]+=stat[2]
				avg[j][2]+=stat[0][0]
			else:
				avg.append([stat[1][0],stat[2],stat[0][0]])

			ranked = sharing(m4Fit(population))

			m4Plot(population)
			#time.sleep(.1)

			pool = SUS(ranked)
			children = noCrossMating(pool)
			population = staticMutate(children)
		m4PlotFinal(initial,population)
		plt.savefig("pictures/run "+str(i+1)+" Sharing M4.png")
		#time.sleep(1)
	myFile = open('data/Sharing M4 raw.csv', 'w')
	with myFile:
	    writer = csv.writer(myFile)
	    writer.writerows(stats)
	myFile.close()
	for i in range(len(avg)):
		avg[i][0]/=10
		avg[i][1]/=10
		avg[i][2]/=10

	data = [['gen','min','avg','max']]
	for i in range(len(avg)):
		data.append([i,avg[i][0],avg[i][1],avg[i][2]])
	myFile = open('data/Sharing M4 avg.csv', 'w')
	with myFile:
	    writer = csv.writer(myFile)
	    writer.writerows(data)
	myFile.close()

	plotReset()
	plotAvg(avg)
	plt.savefig("pictures/Sharing M4 Avg.png")
	time.sleep(10)
#Crowding GA with M1 fitness
elif sys.argv[1] == "crowding" and sys.argv[2] == "m1":
	stats = [['run number','gen number','min','avg','max']]
	avg = []
	print("Running crowding m1")
	for i in range(10):
		line3.set_ydata([])
		line3.set_xdata([])
		print("Run ",i+1)
		population = initializePop(popsize)
		initial = list(population)
		for j in range(numGen):
			stat = rankM1(population)
			stats.append([i,j,stat[1][0],stat[2],stat[0][0]])
			if j < len(avg):
				avg[j][0]+=stat[1][0]
				avg[j][1]+=stat[2]
				avg[j][2]+=stat[0][0]
			else:
				avg.append([stat[1][0],stat[2],stat[0][0]])

			ranked = m1Fit(population)

			m1Plot(population)
			#time.sleep(.1)

			pool = SUS(ranked)
			population = deterministicCrowding(pool)
		m1PlotFinal(initial,population)
		plt.savefig("pictures/run "+str(i+1)+" Crowding M1.png")
		#time.sleep(1)
	myFile = open('data/Crowding M1 raw.csv', 'w')
	with myFile:
	    writer = csv.writer(myFile)
	    writer.writerows(stats)
	myFile.close()
	for i in range(len(avg)):
		avg[i][0]/=10
		avg[i][1]/=10
		avg[i][2]/=10

	data = [['gen','min','avg','max']]
	for i in range(len(avg)):
		data.append([i,avg[i][0],avg[i][1],avg[i][2]])
	myFile = open('data/Crowding M1 avg.csv', 'w')
	with myFile:
	    writer = csv.writer(myFile)
	    writer.writerows(data)
	myFile.close()

	plotReset()
	plotAvg(avg)
	plt.savefig("pictures/Crowding M1 Avg.png")
	time.sleep(10)
#Crowding GA with M4 fitness
elif sys.argv[1] == "crowding" and sys.argv[2] == "m4":
	stats = [['run number','gen number','min','avg','max']]
	avg = []
	print("Running Crowding m4")
	for i in range(10):
		line3.set_ydata([])
		line3.set_xdata([])
		print("Run ",i+1)
		population = initializePop(popsize)
		initial = list(population)
		for j in range(numGen):
			stat = rankM4(population)
			stats.append([i,j,stat[1][0],stat[2],stat[0][0]])
			if j < len(avg):
				avg[j][0]+=stat[1][0]
				avg[j][1]+=stat[2]
				avg[j][2]+=stat[0][0]
			else:
				avg.append([stat[1][0],stat[2],stat[0][0]])

			ranked = m4Fit(population)

			m4Plot(population)
			#time.sleep(.1)

			pool = SUS(ranked)
			population = deterministicCrowding(pool)
		m4PlotFinal(initial,population)
		plt.savefig("pictures/run "+str(i+1)+" Crowding M4.png")
		#time.sleep(1)
	myFile = open('data/Crowding M4 raw.csv', 'w')
	with myFile:
	    writer = csv.writer(myFile)
	    writer.writerows(stats)
	myFile.close()
	for i in range(len(avg)):
		avg[i][0]/=10
		avg[i][1]/=10
		avg[i][2]/=10

	data = [['gen','min','avg','max']]
	for i in range(len(avg)):
		data.append([i,avg[i][0],avg[i][1],avg[i][2]])
	myFile = open('data/Crowding M4 avg.csv', 'w')
	with myFile:
	    writer = csv.writer(myFile)
	    writer.writerows(data)
	myFile.close()

	plotReset()
	plotAvg(avg)
	plt.savefig("pictures/Crowding M4 Avg.png")
	time.sleep(10)
#Crowding and sharing GA with M1 fitness
elif sys.argv[1] == "crowding+sharing" and sys.argv[2] == "m1":
	stats = [['run number','gen number','min','avg','max']]
	avg = []
	print("Running crowding and sharing m1")
	for i in range(10):
		line3.set_ydata([])
		line3.set_xdata([])
		print("Run ",i+1)
		population = initializePop(popsize)
		initial = list(population)
		for j in range(numGen):
			stat = rankM1(population)
			stats.append([i,j,stat[1][0],stat[2],stat[0][0]])
			if j < len(avg):
				avg[j][0]+=stat[1][0]
				avg[j][1]+=stat[2]
				avg[j][2]+=stat[0][0]
			else:
				avg.append([stat[1][0],stat[2],stat[0][0]])

			ranked = sharing(m1Fit(population))

			m1Plot(population)
			#time.sleep(.1)

			pool = SUS(ranked)
			population = deterministicCrowding(pool)
		m1PlotFinal(initial,population)
		plt.savefig("pictures/run "+str(i+1)+" Crowding and sharing M1.png")
		#time.sleep(1)
	myFile = open('data/Crowding and sharing M1 raw.csv', 'w')
	with myFile:
	    writer = csv.writer(myFile)
	    writer.writerows(stats)
	myFile.close()
	for i in range(len(avg)):
		avg[i][0]/=10
		avg[i][1]/=10
		avg[i][2]/=10

	data = [['gen','min','avg','max']]
	for i in range(len(avg)):
		data.append([i,avg[i][0],avg[i][1],avg[i][2]])
	myFile = open('data/Crowding and sharing M1 avg.csv', 'w')
	with myFile:
	    writer = csv.writer(myFile)
	    writer.writerows(data)
	myFile.close()

	plotReset()
	plotAvg(avg)
	plt.savefig("pictures/Crowding and sharing M1 Avg.png")
	time.sleep(10)
#Crowding and sharing GA with M4 fitness
elif sys.argv[1] == "crowding+sharing" and sys.argv[2] == "m4":
	stats = [['run number','gen number','min','avg','max']]
	avg = []
	print("Running Crowding and sharing m4")
	for i in range(10):
		line3.set_ydata([])
		line3.set_xdata([])
		print("Run ",i+1)
		population = initializePop(popsize)
		initial = list(population)
		for j in range(numGen):
			stat = rankM4(population)
			stats.append([i,j,stat[1][0],stat[2],stat[0][0]])
			if j < len(avg):
				avg[j][0]+=stat[1][0]
				avg[j][1]+=stat[2]
				avg[j][2]+=stat[0][0]
			else:
				avg.append([stat[1][0],stat[2],stat[0][0]])

			ranked = sharing(m4Fit(population))

			m4Plot(population)
			#time.sleep(.1)

			pool = SUS(ranked)
			population = deterministicCrowding(pool)
		m4PlotFinal(initial,population)
		plt.savefig("pictures/run "+str(i+1)+" Crowding and sharing M4.png")
		#time.sleep(1)
	myFile = open('data/Crowding and sharing M4 raw.csv', 'w')
	with myFile:
	    writer = csv.writer(myFile)
	    writer.writerows(stats)
	myFile.close()
	for i in range(len(avg)):
		avg[i][0]/=10
		avg[i][1]/=10
		avg[i][2]/=10

	data = [['gen','min','avg','max']]
	for i in range(len(avg)):
		data.append([i,avg[i][0],avg[i][1],avg[i][2]])
	myFile = open('data/Crowding and sharing M4 avg.csv', 'w')
	with myFile:
	    writer = csv.writer(myFile)
	    writer.writerows(data)
	myFile.close()

	plotReset()
	plotAvg(avg)
	plt.savefig("pictures/Crowding and sharing M4 Avg.png")
	time.sleep(10)
