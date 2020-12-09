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

#List of all faces on a rubik's cube
faces = ['U','L','F','R','B','D']

#List of all possible moves on a rubik's cube
moves = ["U","L","F","R","B","D","U'","L'","F'","R'","B'","D'","U2","L2","F2","R2","B2","D2"]

#Initilizing a cube puts it in the solved state, will be used for comparison
solved = pc.Cube()

#Static variables for GA
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
	
	#Up to what move should be getting crossed, with the rest being replaced
	global currentMax
	currentMax = 20
	#What step is the cube on
	global stage
	stage = 0
	#Store if we made a newPop
	global newPop
	newPop = 0

	#Capture variables from the command line
	if len(sys.argv) < 5:
		print("Must include 'numGens popSize numberOfMoves numThreads'")
		exit()
	elif int(sys.argv[2])<2 or int(sys.argv[2])%2!=0:
		print("Pop size must be 2 or greater and even")
		exit()
	elif int(sys.argv[3])<1:
		print("numAllowedMoves must be 1 or more")
		exit()

	#make an move set to scramble the cube, can be static
	#random_alg = pc.Formula("U2 F' R B L2 D' R' B2 D F R D B2 U R' D L2 U' F2 D2 R2 B2 L' U' F")
	alg = pc.Formula()
	random_alg = alg.random()

#Converts moves to a decimal fraction, used for plotting and sharing
#Each move represents 2 digits as there are 18 moves, so the first move would be .00-.18
def moves2dec(movesCube):
	fraction = "0."
	for move in movesCube:
		index = moves.index(move)
		if(index<10):
			fraction+="0"+str(index)
		else:
			fraction+=str(index)
	return float(fraction)

#Plot the cubes after move sets are applied to them, with the x axis being the fraction made above and y being how close it is
def plot(pop):
	x2 = []
	y2 = []
	for i in pop:
		x2.append(moves2dec(i[1]))
		y2.append(howClose(i[1]))
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

#Plot the initial population vs the final population
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

#Sees how close a move set gets to solving the cube, mainly used for plotting
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

#Instead of doing all moves at once like above, we work slowly following the traditional steps to try and solve the cube
def cubeSolve(moves):
	count = 0
	fit = 0
	space = " "
	cube = pc.Cube()
	cube(random_alg)
	tempMoves = list(moves)
	movesNeeded = 100
	global stage
	
	#First solve the bottom face cross
	if count == 0:
		#Main face to compare
		face = 'D'
		#Surrounding faces to make sure the edges of the cross line up
		surroundFace = ['F','R','B','L']
		#Get the solved face
		solvedFace = solved.get_face(face)
		#loop through the moves
		for i in range(len(moves)):
			#Use a move from the list
			move = tempMoves.pop(0)
			#Reset how many squares match
			count = 0
			#Apply the move to the cube
			my_formula = pc.Formula(move)
			cube(my_formula)
			#Get the bottom face
			cubeFace = cube.get_face(face)
			#Test to see if the cross exists
			if(cubeFace[0][1] == solvedFace[0][1]):
				count+=1
			for y in range(0,3):
				if(cubeFace[1][y] == solvedFace[1][y]):
					count+=1
			if(cubeFace[2][1] == solvedFace[2][1]):
				count+=1

			#For the surrounding faces test the edge that lines up with the cross to make sure it matches
			for sFace in surroundFace:
				solvFace = solved.get_face(sFace)
				cFace = cube.get_face(sFace)
				for y in range(1,3):
					if(cFace[y][1] == solvFace[y][1]):
						count+=1

			#If we achieve a count of 13 that means we have a cross with matching edges, log to console and add to the fitness of the move set along with the remaining moves
			if count >= 13:
				if(newPop == 0):
					print("Cross "+str(count))
					print(cube)
					print(int(sys.argv[3])-len(tempMoves))
				fit+=count+len(tempMoves)
				stage = 1
				#Logs the moves needed passed back along with the fitness of the moveset
				movesNeeded = int(sys.argv[3])-len(tempMoves)
				break
			#Used for debugging purposes, make sure we're getting something
			elif count >= 12 and int(sys.argv[3])-len(tempMoves)<currentMax and stage == 0:
				#if(len(tempMoves)>=70):
				print("Cross "+str(count))
				print(cube)
				print(int(sys.argv[3])-len(tempMoves))
		#If it doesn't manage to achieve it still give it a fitness value
		fit+=count

	#Then solve the whole bottom face if cross was solved, same as previous section but includes the corners now
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
			#See if we fully solved the bottom face at all
			if count >= 9:
				print("Bottom "+str(count))
				print(cube)
				print(int(sys.argv[3])-len(tempMoves))

			#Check surrounding faces
			for sFace in surroundFace:
				solvFace = solved.get_face(sFace)
				cFace = cube.get_face(sFace)
				for y in range(3):
					if(cFace[2][y] == solvFace[2][y]):
						count+=1
				if(cFace[1][1] == solvFace[1][1]):
					count+=1

			#If only all 25 squares match up do we consider this step done
			#Add count and left over moves to the fitness function
			if count >= 25:
				print("Bottom "+str(count))
				print(cube)
				print(int(sys.argv[3])-len(tempMoves))
				fit+=count+len(tempMoves)
				movesNeeded = int(sys.argv[3])-len(tempMoves)
				stage = 2
				break
		#If in this step add what does match up
		fit+=count

	#If bottom face is solved solve the lower parts of the surrounding faces
	#Try to solve F2L, first 2 layers leaving only the top face and upper layer of the surrounding faces unsolved
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
				if face == 'D':
					for j in range(0,3):
						for y in range(0,3):
							if(cubeFace[j][y] == solvedFace[j][y]):
								count+=1
				else:
					for j in range(1,3):
						for y in range(0,3):
							if(cubeFace[j][y] == solvedFace[j][y]):
								count+=1

			#If all 33 squares match this step is complete
			if count >= 33:
				print("Next 2 layers "+str(count))
				fit+=count+len(tempMoves)
				stage = 3
				movesNeeded = int(sys.argv[3])-len(tempMoves)
				break
		fit+=count

	return [fit,movesNeeded,stage]

#Generates a set of random moves of length N
def randomMoves(length):
	sample = []
	while len(sample) < length:
		sample.append(random.choice(moves))
	return sample

#Create population of size N with number of moves num
def createPopulation(size,num):
	population = []
	for i in range(0,size):
		population.append(randomMoves(num))
	return population

#make a new pop based off a defined beginning, forces population into a niche
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

#Used for multiprocessed stuff from fitPop, organizes it and puts it into the queue
def fitnessPop(pop,q):
	ranked = []
	for i in pop:
		result = cubeSolve(i)
		q.put([result[0],i,result[1],result[2]])

#Creates multiple processes to rank individuals based upon how close they come to solving the cube following cubeSolve()
def fitPop(pop):
	ranked = []
	#Establish a queue to get elements back from other processes
	que = Queue()
	threads = []
	#Establishes how many elements per process there'll be
	chunkSize = int(int(sys.argv[2])/int(sys.argv[4]))
	for i in range(int(sys.argv[4])):
		#If we're making the last process put all remaining elements, else put the chunk of the pop array into the process
		if(i == int(sys.argv[4])-1):
			threads.append(Process(target=fitnessPop,args=[pop[i*chunkSize:],que,]))
		else:
			threads.append(Process(target=fitnessPop,args=[pop[i*chunkSize:i*chunkSize+chunkSize],que,]))
		#Start the thread
		threads[-1].start()

	#Watch the queue for elements and wait until we get all elements back
	while not que.empty() or len(ranked)<int(sys.argv[2]):
		#Get item from the queue
		item = que.get()
		#Append item if ranked items is empty
		if(len(ranked)==0):
			ranked.append(item)
		#Insertion sort the item into ranked, keeping the best at the front
		else:
			inserted = False
			for y in range(0,len(ranked)):
				if(item[0] > ranked[y][0]):
					ranked.insert(y,item)
					inserted = True
					break
			#Incase the item isn't inserted before add to the end
			if not inserted:
				ranked.append(item)

	#See if the best ranked is within a margin to move the current max moves used and that it completed step 1
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

#Stochastic Universal Sampling, tries to give them all a porportional chance according to their fitness
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

#Mutates an individual child according to the mutation rate, total amount of possible mutations is dependent on the max moves
def mutate(child):
	for i in range(int(currentMax/10)):
		if(random.random() < Pm):
			rand = random.randint(0,currentMax-1)
			child[rand] = random.choice(moves)
	return child

#Only crosses 1/3s of parents up to currentMax where all remaining elements will be randomly generated. Keep randomness and diversity up
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
	#loop desired amount of times length
	for i in range(0,length):
		#Give each parents in the pool a chance to mate
		x = i
		#x = random.randint(0,len(pool)-1)
		#Find a random partner
		y = random.randint(0,len(pool)-1)
		#Make sure x and y are not the same parent
		while x == y:
			y = random.randint(0,len(pool)-1)

		#Pull the parents from the pool
		p1 = pool[x]
		p2 = pool[y]
		
		#Cross and mutate children from p1 and p2
		c1 = mutate(staticCrossover(p1,p2))
		c2 = mutate(staticCrossover(p2,p1))

		#Get the fitness values of each individual
		p1Fit=cubeSolve(p1)
		p2Fit=cubeSolve(p2)
		c1Fit=cubeSolve(c1)
		c2Fit=cubeSolve(c2)

		#Find if p1,c1 and p2,c2 are closer than p1,c2 and p2,c1
		if abs(moves2dec(p1)-moves2dec(c1))+abs(moves2dec(p2)-moves2dec(c2))<=abs(moves2dec(p1)-moves2dec(c2))+abs(moves2dec(p2)-moves2dec(c1)):
			#Use elitism of the closest parent and child to decide who'll be put into the next generation
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

#Make deterministic crowding multithreaded
def deterCrowdMulti(pool):
	children = []
	que = Queue()
	threads = []
	chunkSize = int(len(pool)/int(sys.argv[4]))
	for i in range(int(sys.argv[4])):
		#Create threads with number of loops equal chunksize, last thread should be the remaining loops
		if(i == int(sys.argv[4])-1):
			threads.append(Process(target=deterministicCrowding,args=[pool,len(pool)-chunkSize*i,que,]))
		else:
			threads.append(Process(target=deterministicCrowding,args=[pool,chunkSize,que,]))
		threads[-1].start()

	#Make sure que is empty and we have all the children
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
	#Make sure line3 is clear
	line3.set_ydata([])
	line3.set_xdata([])
	#Store data for csvs
	stats = [['run number','gen number','min','avg','max']]
	avg = []
	#Display the random cube and algorithm to achieve that cube
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
		#Display the run
		print("Run ",i+1)

		#Make the initial population
		population = createPopulation(int(sys.argv[2]),int(sys.argv[3]))
		#Store the initial population
		initial = list(population)
		#Loop for numGen generations
		closest = []
		for j in range(int(sys.argv[1])):
			#Check for making sure the program is running
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

			#See if we have a good solution for making the bottom cross depending on the number of moves used, if we do remake the population using that
			#Also don't remake it every generation defined by newPop
			if stage == 1 and ranked[0][0] > 13 and ranked[0][2] <= 15 and newPop == 0:
				population = newPop(ranked[0][1][:ranked[0][2]])
				newPop = 1
				print("Made new pop")
			else:
				#Get the individuals who'll be mating
				#pool = SUS(sharing(ranked))
				#print(ranked[0][0],ranked[0][2],ranked[0][3])
				pool = SUS(ranked)
				#Get the children from mating individuals
				population = deterCrowdMulti(pool)
				#Mutate the group of children
				#population = staticMutate(children)
			#Store the best found solution and compare it to the current best solution
			if closest == []:
				closest = ranked[0]
			elif closest[0] < ranked[0][0]:
				closest = ranked[0]

		#Once out of the loop plot the final population
		plotFinal(initial,population)
		#Save a pic of it
		plt.savefig("pictures/run "+str(i+1)+" GA.png")

		#check the fitness of the final population
		temp = fitPop(population)
		#Check that best with the current best
		if closest[0] < temp[0][0]:
			closest = temp[0]
		#Display what step of solving a cube was achieved
		print("Stage: "+str(stage))
		#Reset the stage for the next run
		stage = 0
		#Show the number of matching squares for the closest
		print("Closest: "+str(howClose(closest[1][:closest[2]])))
		#Reset current max
		currentMax = 20
		#Construct the moveset into a usable formula but only utilizing the number of moves needed for the best solved step
		space = " "
		string = space.join(closest[1][:closest[2]])
		my_formula = pc.Formula(string)
		randomCube = pc.Cube()
		#Apply the random alg with our best move set
		randomCube(random_alg)
		randomCube(my_formula)
		#Print the move set and cube
		print(my_formula)
		print(randomCube)
		#time.sleep(1)
	#Save the stats to their csv files
	myFile = open('data/GA raw.csv', 'w')
	with myFile:
	    writer = csv.writer(myFile)
	    writer.writerows(stats)
	myFile.close()
	#Adjust the min, max and avg to be an average of all 10 runs
	for i in range(len(avg)):
		avg[i][0]/=10
		avg[i][1]/=10
		avg[i][2]/=10

	#Generate the csv for the avg data
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