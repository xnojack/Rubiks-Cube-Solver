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
import matplotlib as mpl
mpl.use('Agg')
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
Pm = .50
global Pc
Pc = .9
global alpha
alpha = 1
global share
share = .0025


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
	line3, = ax.plot([0,.18], [0,54], 'b-')
	
	global currentMax
	currentMax = 20
	global stage
	stage = 0
	global newPop
	newPop = 0

	if len(sys.argv) < 5:
		print("Must include 'numGens popSize numberOfMoves numThreads'")
		exit()
	elif int(sys.argv[2])<2 or int(sys.argv[2])%2!=0:
		print("Pop size must be 2 or greater and even")
		exit()
	elif int(sys.argv[3])<1:
		print("numAllowedMoves must be 1 or more")
		exit()

	#random_alg = pc.Formula("U2 F' R B L2 D' R' B2 D F R D B2 U R' D L2 U' F2 D2 R2 B2 L' U' F")
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
	# fig.canvas.draw()
	# fig.canvas.flush_events()

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
	# fig.canvas.draw()
	# fig.canvas.flush_events()

#Displays the avg min, avg and max values for the found optima
def plotAvg(avg):
	line1, = ax.plot([0,int(sys.argv[1])], [0,54], 'g-')
	line2, = ax.plot([0,int(sys.argv[1])], [0,54], 'r-')
	line3, = ax.plot([0,int(sys.argv[1])], [0,54], 'b-')

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

#Instead of doing all moves at once like above, we work slowly to the end result
def cubeSolve(moves):
	count = 0
	fit = 0
	space = " "
	cube = pc.Cube()
	cube(random_alg)
	tempMoves = list(moves)
	movesNeeded = 100
	global stage
	
	#First solve the bottom white cross
	if count == 0:
		face = 'D'
		surroundFace = ['F','R','B','L']
		solvedFace = solved.get_face(face)
		for i in range(len(moves)):
			move = tempMoves.pop(0)
			count = 0
			my_formula = pc.Formula(move)
			cube(my_formula)
			cubeFace = cube.get_face(face)
			if(cubeFace[0][1] == solvedFace[0][1]):
				count+=1
			for y in range(0,3):
				if(cubeFace[1][y] == solvedFace[1][y]):
					count+=1
			if(cubeFace[2][1] == solvedFace[2][1]):
				count+=1

			for sFace in surroundFace:
				solvFace = solved.get_face(sFace)
				cFace = cube.get_face(sFace)
				for y in range(1,3):
					if(cFace[y][1] == solvFace[y][1]):
						count+=1

			if count >= 13:
				if(newPop == 0):
					print("Cross "+str(count))
					print(cube)
					print(int(sys.argv[3])-len(tempMoves))
				fit+=count+len(tempMoves)
				stage = 1
				movesNeeded = int(sys.argv[3])-len(tempMoves)
				break
			elif count >= 12 and int(sys.argv[3])-len(tempMoves)<currentMax and stage == 0:
				#if(len(tempMoves)>=70):
				print("Cross "+str(count))
				print(cube)
				print(int(sys.argv[3])-len(tempMoves))
		fit+=count

	#Then solve the white face
	if count >= 13:
		face = 'D'
		surroundFace = ['F','R','B','L']
		solvedFace = solved.get_face(face)
		rMoves = len(tempMoves)
		for i in range(rMoves):
			count = 0
			move = tempMoves.pop(0)
			my_formula = pc.Formula(move)
			cube(my_formula)
			cubeFace = cube.get_face(face)
			for j in range(0,3):
				for y in range(0,3):
					if(cubeFace[j][y] == solvedFace[j][y]):
						count+=1

			if count >= 9:
				print("Bottom "+str(count))
				print(cube)
				print(int(sys.argv[3])-len(tempMoves))

			for sFace in surroundFace:
				solvFace = solved.get_face(sFace)
				cFace = cube.get_face(sFace)
				for y in range(3):
					if(cFace[2][y] == solvFace[2][y]):
						count+=1
				if(cFace[1][1] == solvFace[1][1]):
					count+=1

			if count >= 25:
				print("Bottom "+str(count))
				print(cube)
				print(int(sys.argv[3])-len(tempMoves))
				fit+=count+len(tempMoves)
				movesNeeded = int(sys.argv[3])-len(tempMoves)
				stage = 2
				break

		fit+=count

	#If bottom face is solved solve the lower parts of the surrounding faces
	if count >= 25:
		faceLoop = ['D','F','R','B','L']
		
		rMoves = len(tempMoves)
		
		for i in range(rMoves):
			count = 0
			move = tempMoves.pop(0)
			my_formula = pc.Formula(move)
			cube(my_formula)
			for face in faces:
				solvedFace = solved.get_face(face)
				cubeFace = cube.get_face(face)
				for j in range(1,3):
					for y in range(0,3):
						if(cubeFace[j][y] == solvedFace[j][y]):
							count+=1
			if count >= 33:
				print("Next 2 layers "+str(count))
				fit+=count+len(tempMoves)
				stage = 3
				movesNeeded = int(sys.argv[3])-len(tempMoves)
				break
		fit+=count
	if(fit==0):
		fit = howClose(moves)%3
	return [fit,movesNeeded,stage]

#Generates a set of random moves of length N
def randomMoves(length):
	sample = []
	while len(sample) < length:
		sample.append(random.choice(moves))
	return sample

#Create population of size N
def createPopulation(size,num):
	population = []
	for i in range(0,size):
		population.append(randomMoves(num))
	return population

#make a new pop based off a defined beginning
def newPop(beginning):
	pop = []
	for i in range(int(sys.argv[2])):
		temp = []
		for gene in beginning:
			temp.append(gene)
		while len(temp) < int(sys.argv[3]):
			temp.append(random.choice(moves))
		pop.append(temp)
	return pop

#Normalizes the fitness so that they add up to equal 1
def normalize(ranked):
	total = 0
	norm = list(ranked)
	for i in ranked:
		total += i[0]
	for i in norm:
		i[0]/=total
	return norm

#Used for multiprocessed stuff from fitPop
def fitnessPop(pop,q):
	ranked = []
	for i in pop:
		result = cubeSolve(i)
		q.put([result[0],i,result[1],result[2]])

#Creates multiple processes to rank individuals based upon how close they come to solving the cube
def fitPop(pop):
	ranked = []
	que = Queue()
	threads = []
	chunkSize = int(int(sys.argv[2])/int(sys.argv[4]))
	for i in range(int(sys.argv[4])):
		if(i == int(sys.argv[4])-1):
			threads.append(Process(target=fitnessPop,args=[pop[i*chunkSize:],que,]))
		else:
			threads.append(Process(target=fitnessPop,args=[pop[i*chunkSize:i*chunkSize+chunkSize],que,]))
		threads[-1].start()
	# for i in threads:
	# 	print("Waiting on thread "+str(i))
	# 	i.join()
	# 	print("Complete thread "+str(i))

	while not que.empty() or len(ranked)<int(sys.argv[2]):
		item = que.get()
		if(len(ranked)==0):
			ranked.append(item)
		else:
			inserted = False
			for y in range(0,len(ranked)):
				if(item[0] > ranked[y][0]):
					ranked.insert(y,item)
					inserted = True
					break
			if not inserted:
				ranked.append(item)

	if(ranked[0][3] == 1 and ranked[0][2] <= 40):
		global currentMax
		currentMax = ranked[0][2]
		global stage
		stage = 1
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
	for i in range(int(currentMax/10)):
		if(random.random() < Pm):
			rand = random.randint(0,currentMax-1)
			child[rand] = random.choice(moves)
	return child

#Takes first 3 elements from p1, second 3 from p2 and last 4 from p1 to generate a child
def staticCrossover(p1,p2):
	child = []
	for i in range(0,int(currentMax/3)):
		child.append(p1[i])
	for i in range(int(currentMax/3),int(currentMax/3)*2):
		child.append(p2[i])
	for i in range(int(currentMax/3)*2,currentMax):
		child.append(p1[i])
	for i in range(currentMax,int(sys.argv[3])):
		child.append(random.choice(moves))
	return child

#Deterministic crowding method to keep the better/closer parent/child
def deterministicCrowding(pool,length,q):
	
	for i in range(0,length):
		distanceThresh = .01
		x = i
		#x = random.randint(0,len(pool)-1)
		y = random.randint(0,len(pool)-1)
		while x == y:
			y = random.randint(0,len(pool)-1)

		p1 = pool[x]
		p2 = pool[y]
		

		c1 = mutate(staticCrossover(p1,p2))
		c2 = mutate(staticCrossover(p2,p1))

		p1Fit=cubeSolve(p1)
		p2Fit=cubeSolve(p2)
		c1Fit=cubeSolve(c1)
		c2Fit=cubeSolve(c2)

		if abs(moves2dec(p1)-moves2dec(c1))+abs(moves2dec(p2)-moves2dec(c2))<=abs(moves2dec(p1)-moves2dec(c2))+abs(moves2dec(p2)-moves2dec(c1)):
			if(c1Fit > p1Fit):
				q.put(c1)
			else:
				q.put(p1)
			if(c2Fit > p2Fit):
				q.put(c2)
			else:
				q.put(p2)
		else:
			if(c2Fit > p1Fit):
				q.put(c2)
			else:
				q.put(p1)
			if(c1Fit > p2Fit):
				q.put(c1)
			else:
				q.put(p2)

def deterCrowdMulti(pool):
	children = []
	que = Queue()
	threads = []
	chunkSize = int(len(pool)/int(sys.argv[4]))
	for i in range(int(sys.argv[4])):
		if(i == int(sys.argv[4])-1):
			threads.append(Process(target=deterministicCrowding,args=[pool,len(pool)-chunkSize*i,que,]))
		else:
			threads.append(Process(target=deterministicCrowding,args=[pool,chunkSize,que,]))
		threads[-1].start()
	# for i in threads:
	# 	i.join()
	# 	i.close()

	while not que.empty() or len(children)<int(sys.argv[2]):
		children.append(que.get())

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
		population = createPopulation(int(sys.argv[2]),int(sys.argv[3]))
		#Store the initial population
		initial = list(population)
		#Loop for numGen generations
		closest = []
		for j in range(int(sys.argv[1])):
			if(j%10==0):
				print("Gen "+str(j))
				print(currentMax)
			#Run the fitness of the population
			ranked = fitPop(population)
			#Plot the population
			#plot(ranked)

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

			#See if we have a good solution for making the white cross, if we do remake the population using that
			if stage == 1 and ranked[0][0] > 13 and ranked[0][2] <= 15 and newPop == 0:
				population = newPop(ranked[0][1][:ranked[0][2]])
				newPop = 1
				print("Made new pop")
			else:
				#Get the individuals who'll be mating
				#pool = SUS(sharing(ranked))
				print(ranked[0][0],ranked[0][2],ranked[0][3])
				pool = SUS(ranked)
				#Get the children from mating individuals
				population = deterCrowdMulti(pool)
				#Mutate the group of children
				#population = staticMutate(children)
			if closest == []:
				closest = ranked[0]
			elif closest[0] < ranked[0][0]:
				closest = ranked[0]

		#Once out of the loop plot the final population
		plotFinal(initial,population)
		#Save a pic of it
		plt.savefig("pictures/run "+str(i+1)+" GA.png")

		temp = fitPop(population)
		if closest[0] < temp[0][0]:
			closest = temp[0]
		print("Stage: "+str(stage))
		stage = 0
		print("Closest: "+str(howClose(closest[1][:closest[2]])))
		currentMax = 10
		space = " "
		string = space.join(closest[1][:closest[2]])
		my_formula = pc.Formula(string)
		randomCube = pc.Cube()
		randomCube(random_alg)
		randomCube(my_formula)
		print(my_formula)
		print(randomCube)
		#time.sleep(1)
	#Save the stats to their csv files
	myFile = open('data/GA raw.csv', 'w')
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
	myFile = open('data/GA avg.csv', 'w')
	with myFile:
	    writer = csv.writer(myFile)
	    writer.writerows(data)
	myFile.close()

	#reset the plot and plot the avg for all 10 runs
	plotReset()
	plotAvg(avg)
	plt.savefig("pictures/GA Avg.png")
	time.sleep(10)