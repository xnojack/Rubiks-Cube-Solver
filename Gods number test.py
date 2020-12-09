import numpy as np
import sys
from threading import Thread
import random
from multiprocessing import Queue, Process
import time
import json
import os
import pycuber as pc
import math
import matplotlib.pyplot as plt
import csv

global faces
global moves
global solved
global random_alg

faces = ['U','L','F','R','B','D']
moves = ["U","L","F","R","B","D","U'","L'","F'","R'","B'","D'","U2","L2","F2","R2","B2","D2"]

solved = pc.Cube()

global Pm
Pm = .5
global Pc
Pc = .9
global alpha
alpha = 1
global share
share = .005


if __name__ == '__main__':
	#Code for the plotting function in python
	plt.ion()
	global fig
	fig = plt.figure()
	ax = fig.add_subplot(111)
	global line1
	global line2
	global line3
	line1, = ax.plot([0,.18], [0,54], 'r.')
	line2, = ax.plot([0,.18], [0,54], 'go')
	line3, = ax.plot([0,.18], [0,54], 'b.')

	if len(sys.argv) < 4:
		print("Must include 'numGens popSize numberOfMoves numThreads'")
		exit()
	elif int(sys.argv[2])<2 or int(sys.argv[2])%2!=0:
		print("Pop size must be 2 or greater and even")
		exit()
	elif int(sys.argv[3])<1:
		print("numAllowedMoves must be 1 or more")
		exit()

	alg = pc.Formula()
	random_alg = alg.random()

#Converts moves to a decimal fraction, used for plotting and sharing
def moves2dec(movesCube):
	fraction = "0."
	for move in movesCube:
		index = moves.index(move)
		if(index<10):
			fraction+="0"+str(index)
		else:
			fraction+=str(index)
	return float(fraction)

#Sees how close a move set gets to solving the cube
def howClose(moves):
	count = 0
	space = " "
	string = space.join(moves)
	my_formula = pc.Formula(string)
	cube = pc.Cube()
	cube(random_alg)
	cube(my_formula)
	for face in faces:
		solvedFace = solved.get_face(face)
		cubeFace = cube.get_face(face)
		for i in range(0,3):
			for y in range(0,3):
				if(cubeFace[i][y] == solvedFace[i][y]):
					count = count + 1
	return count

#Bruteforce plot
if __name__ == '__main__':
	x1 = []
	y1 = []

	plotGraph = []
	length = 20
	for i in range(length):
		plotGraph.append(moves[0])

	count = 0
	while moves.index(plotGraph[length-1]) < 18:
		
		newIndex = moves.index(plotGraph[0])+1
		if(newIndex>=18):
			carried = False
			y=1
			plotGraph[0]=moves[0]
			while not carried and y<length:
				index = moves.index(plotGraph[y])+1
				if(index>=18):
					plotGraph[y] = moves[0]
					y+=1
				else:
					plotGraph[y] = moves[index]
					carried = True
		else:
			plotGraph[0]=moves[newIndex]

		test = howClose(plotGraph)
		if(test >= 25):
			count+=1
			x1.append(moves2dec(plotGraph))
			y1.append(howClose(plotGraph))
			#if count%100==0:
			print(plotGraph)
			if count%5==0:
				line3.set_ydata(y1)
				line3.set_xdata(x1)
				fig.canvas.draw()


def plot(pop):
	x2 = []
	y2 = []
	for i in pop:
		x2.append(moves2dec(i[1]))
		y2.append(i[0])
	line1.set_ydata(y2)
	line1.set_xdata(x2)

	

	#plt.xlabel('x') 
	# Labeling the Y-axis 
	#plt.ylabel('M2') 
	# Give a title to the graph
	#plt.title('Sin(x) and Cos(x) on the same graph') 
	  
	# Show a legend on the plot 
	fig.canvas.draw()
	fig.canvas.flush_events()

def plotFinal(initial,final):
	x2 = []
	y2 = []
	for i in initial:
		x2.append(moves2dec(i))
		y2.append(howClose(i))
	line1.set_ydata(y2)
	line1.set_xdata(x2)

	x3 = []
	y3 = []
	for i in final:
		x3.append(moves2dec(i))
		y3.append(howClose(i))
	line2.set_ydata(y3)
	line2.set_xdata(x3)

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
	line1, = ax.plot([0,100], [0,54], 'g-')
	line2, = ax.plot([0,100], [0,54], 'r-')
	line3, = ax.plot([0,100], [0,54], 'b-')

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

#Instead of doing all moves at once like above, we work slowly to the end result
def cubeSolve(moves):
	count = 0
	space = " "
	cube = pc.Cube()
	cube(random_alg)
	
	#First solve the Bottom face
	if count == 0:
		face = 'D'
		solvedFace = solved.get_face(face)
		tempMoves = list(moves)
		for i in range(len(moves)):
			move = tempMoves.pop(0)
			count = 0
			my_formula = pc.Formula(move)
			cube(my_formula)
			cubeFace = cube.get_face(face)
			for i in range(0,3):
				for y in range(0,3):
					if(cubeFace[i][y] == solvedFace[i][y]):
						count+=1

			if count >= 9:
				print(fCount)
				count+=9+len(tempMoves)
				break
	#If bottom face is solved solve the lower parts of the surrounding faces
	if count >= 9:
		faceLoop = ['D','F','R','B','L']
		
		rMoves = len(tempMoves)
		
		for i in range(rMoves):
			move = tempMoves.pop(0)
			fCount = 0
			my_formula = pc.Formula(move)
			cube(my_formula)
			for face in faces:
				solvedFace = solved.get_face(face)
				cubeFace = cube.get_face(face)
				for i in range(0,3):
					for y in range(0,3):
						if(cubeFace[i][y] == solvedFace[i][y]):
							fCount+=1
			if fCount >= 33:
				count+=33
				break
	if(count==0):
		count = howClose(moves)%3
	return count

#Generates a set of random moves of length N
def randomMoves():
	length = int(sys.argv[3])
	sample = []
	while len(sample) < length:
		sample.append(random.choice(moves))
	return sample

#Create population of size N
def createPopulation(size):
	population = []
	for i in range(0,size):
		population.append(randomMoves())
	return population

#Normalizes the fitness so that they add up to equal 1
def normalize(ranked):
	total = 0
	norm = list(ranked)
	for i in ranked:
		total += i[0]
	for i in norm:
		i[0]/=total
	return norm

#Creates multiple processes to rank individuals based upon how close they come to solving the cube
def fitPop(pop):
	ranked = []
	for i in pop:
		fitness = howClose(i)
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

#Adds up all the fitness in a ranked population
def sumFit(ranked):
	total = 0
	for i in ranked:
		total+=i[0]
	return total

#Returns the min, max and avg of a population according to the fitness
def rank(fit):
	max = [0,[]]
	min = [1,[]]
	avg = 0

	for individual in fit:
		if individual[0] > max[0]:
			max[0] = individual[0]
			max[1] = individual[1]
		if individual[0] < min[0]:
			min[0] = individual[0]
			min[1] = individual[1]
		avg+=individual[0]

	avg/=int(sys.argv[2])
	return [max,min,avg]

#Stochastic Universal Sampling, tries to give them all a porportional chance
def SUS(ranked):
	f=sumFit(ranked)
	n=int(int(sys.argv[2])/2)
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

#Mutates and individual child according to the mutation rate
def mutate(child):
	if(random.random() < Pm):
		rand = random.randint(0,int(sys.argv[3])-1)
		child[rand] = random.choice(moves)
	if(random.random() < Pm):
		rand = random.randint(0,int(sys.argv[3])-1)
		child[rand] = random.choice(moves)
	if(random.random() < Pm):
		rand = random.randint(0,int(sys.argv[3])-1)
		child[rand] = random.choice(moves)
	if(random.random() < Pm):
		rand = random.randint(0,int(sys.argv[3])-1)
		child[rand] = random.choice(moves)
	if(random.random() < Pm):
		rand = random.randint(0,int(sys.argv[3])-1)
		child[rand] = random.choice(moves)
	return child

#Takes first 3 elements from p1, second 3 from p2 and last 4 from p1 to generate a child
def staticCrossover(p1,p2):
	child = []
	for i in range(0,int(int(sys.argv[3])/3)):
		child.append(p1[i])
	for i in range(int(int(sys.argv[3])/3),int(int(sys.argv[3])/3)*2):
		child.append(p2[i])
	for i in range(int(int(sys.argv[3])/3)*2,int(sys.argv[3])):
		child.append(p1[i])
	return child

def randomCrossover(p1,p2):
	child = []
	x=0
	y=random.random()
	p1Part = p1[int(x*len(p1)):int(y*len(p1))]
	p2Part = p2[int(x*len(p2)):int(y*len(p2))]

	

#Deterministic crowding method to keep the better/closer parent/child
def deterministicCrowding(pool):
	children = []
	for i in range(0,len(pool)):
		x = random.randint(0,len(pool)-1)
		y = random.randint(0,len(pool)-1)
		while x == y:
			y = random.randint(0,len(pool)-1)

		p1 = pool[x]
		p2 = pool[y]

		c1 = mutate(staticCrossover(p1,p2))
		c2 = mutate(staticCrossover(p2,p1))

		p1Fit=howClose(p1)
		p2Fit=howClose(p2)
		c1Fit=howClose(c1)
		c2Fit=howClose(c2)

		if abs(moves2dec(p1)-moves2dec(c1))+abs(moves2dec(p2)-moves2dec(c2))<=abs(moves2dec(p1)-moves2dec(c2))+abs(moves2dec(p2)-moves2dec(c1)):
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
			distance = abs(moves2dec(ranked[i][1])-moves2dec(ranked[y][1]))
			#Based off the share variable, if we're too far skip
			if(distance >= share): continue
			#share sum is then calculated
			shSum+=1-pow(distance/share,alpha)
		#Skip divide by zero
		if(shSum == 0): continue
		#Divide by the share sum
		ranked[i][0] = ranked[i][0]/shSum
	return ranked

#Mutates a group of children according to the mutation rate
def staticMutate(children):
	for child in children:
		child = mutate(child)
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

def wipePop(ranked):
	keep = SUS(sharing(ranked))
	while len(keep) < len(ranked):
		keep.append(randomMoves())
	return keep


if __name__ == '__main__':
	line3.set_ydata([])
	line3.set_xdata([])
	#Store data for csvs
	stats = [['run number','gen number','min','avg','max']]
	avg = []
	print("Random Alg: ")
	print(random_alg)
	randomCube = pc.Cube()
	randomCube(random_alg)
	print(randomCube)
	#Make 10 runs
	for i in range(10):
		#Clear line2 (Final pop) from plot
		line2.set_ydata([])
		line2.set_xdata([])

		print("Run ",i+1)

		#Make the initial population
		population = createPopulation(int(sys.argv[2]))
		#Store the initial population
		initial = list(population)
		#Loop for numGen generations
		for j in range(int(sys.argv[1])):
			if(j%10==0):
				print("Gen "+str(j))
				print(len(population))
			#Run the fitness of the population
			ranked = fitPop(population)
			# if(j%10==0 and j!=0):
			# 	ranked = fitPop(wipePop(ranked))
			#Plot the population
			plot(ranked)

			#Get the stats of the current generation
			stat = rank(ranked)
			stats.append([i,j,stat[1][0],stat[2],stat[0][0]])
			#For storing the avg results
			if j < len(avg):
				avg[j][0]+=stat[1][0]
				avg[j][1]+=stat[2]
				avg[j][2]+=stat[0][0]
			else:
				avg.append([stat[1][0],stat[2],stat[0][0]])

			#time.sleep(.1)

			#Get the individuals who'll be mating
			#pool = SUS(sharing(ranked))
			pool = SUS(ranked)
			#Get the children from mating individuals
			population = deterministicCrowding(pool)
			#Mutate the group of children
			#population = staticMutate(children)
		#Once out of the loop plot the final population
		plotFinal(initial,population)
		#Save a pic of it
		plt.savefig("pictures/run "+str(i+1)+" GA "+sys.argv[3]+" moves.png")

		temp = fitPop(population)
		print("Closest: "+str(temp[0][0]))
		space = " "
		string = space.join(temp[0][1])
		my_formula = pc.Formula(string)
		randomCube(my_formula)
		print(my_formula)
		print(randomCube)
		#time.sleep(1)
	#Save the stats to their csv files
	myFile = open('data/GA raw '+sys.argv[3]+' moves.csv', 'w')
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
	myFile = open('data/GA avg '+sys.argv[3]+' moves.csv', 'w')
	with myFile:
	    writer = csv.writer(myFile)
	    writer.writerows(data)
	myFile.close()

	#reset the plot and plot the avg for all 10 runs
	plotReset()
	plotAvg(avg)
	plt.savefig("pictures/GA Avg "+sys.argv[3]+" moves.png")
	time.sleep(10)