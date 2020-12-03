import numpy as np
import sys
import threading
import random
from multiprocessing import Process, Queue
import time
import json
import os
import pycuber as pc

global faces
global moves
global solved
global random_alg

if len(sys.argv) < 5:
	print("Must include 'numRuns numProcesses popSize numAllowedMoves'")
	exit()
elif int(sys.argv[3])<2 or int(sys.argv[3])%2!=0:
	print("Pop size must be 2 or greater and even")
	exit()
elif int(sys.argv[4])<1:
	print("numAllowedMoves must be 1 or more")
	exit()
	

faces = ['U','L','F','R','B','D']
moves = ["U","L","F","R","B","D","U'","L'","F'","R'","B'","D'","U2","L2","F2","R2","B2","D2"]

solved = pc.Cube()

alg = pc.Formula()
random_alg = alg.random()

#Get the start time to calculate run time
start_time = time.time()

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
	
def randomMoves():
	length = int(random.random()*int(sys.argv[4]))
	sample = []
	while len(sample) < length:
		sample.append(random.choice(moves))
	return sample

#This is where the shortest path will be found
def createPopulation(size):
	population = []
	for i in range(0,size):
		population.append(randomMoves())
	return population

def fitness(population):
	ranked = []

	for i in population:
		facesMatch = howClose(i)
		if len(ranked) < 1:
			ranked.append([facesMatch,i])
		else:
			for y in range(0,len(ranked)):
				if(facesMatch >= ranked[y][0]):
					ranked.insert(y,[facesMatch,i])
					break
			if [facesMatch,i] not in ranked:
				ranked.append([facesMatch,i])
	return ranked

def mate(p1,p2):
	child = []
	x=0
	y=random.random()
	p1Part = p1[int(x*len(p1)):int(y*len(p1))]
	p2Part = p2[int(x*len(p2)):int(y*len(p2))]

	if(howClose(p1Part)<howClose(p2Part)):
		for i in range(0,len(p2)):
			if i == int(x) or i == int(y):
				for z in p1Part:
					child.append(z)
				if x>y:
					i=int(x)
				else:
					i=int(y)
			if p2[i] not in p1Part and len(child) < 21:
				child.append(p2[i])
		return child
	else:
		for i in range(0,len(p1)):
			if i == int(x) or i == int(y):
				for z in p2Part:
					child.append(z)
				if x>y:
					i=int(x)
				else:
					i=int(y)
			if p1[i] not in p2Part and len(child) < 21:
				child.append(p1[i])
		return child

def mateCouples(pool,length):
	children = []
	for i in range(len(pool)):
		p1 = pool[i]
		p2 = pool[len(pool)-i-1]
		child = mate(p1,p2)
		if child not in children:
			children.append(child)
	for i in range(1,len(pool)):
		p1 = pool[i-1]
		p2 = pool[i]
		child = mate(p1,p2)
		if child not in children:
			children.append(child)
	# print(len(children))
	# print(length)
	# print(len(pool))
	while len(children) <max(int(sys.argv[3]),length)-2:
		children.append(mate(random.choice(pool),random.choice(pool)))
	# print(len(children))
	return children

def matingPool(ranked):
	next = []
	pool = []
	next.append(ranked[0][1])
	next.append(ranked[1][1])

	i=2
	#print(len(next))
	while len(pool)<len(ranked)*.4 and i<len(ranked)-1:
		pool.append(ranked[i][1])
		pool.append(ranked[i+1][1])
		i+=2
	children = mateCouples(pool,len(ranked))

	for i in children:
		# if next.count(i)>1:
		# 	print("yeet")
		next.append(i)

	# print(children.count(ranked[0][1]))
	# print(children.count(ranked[1][1]))
	if(len(ranked) < int(sys.argv[3]) or len(next) < int(sys.argv[3])):
		print("Ranked: "+str(len(ranked)))
		print("Next: "+str(len(next)))

	return next

def mutate(children):
	rate = 0
	for i in range(2,len(children)):
		if(random.random() < .01):
			x = random.random()
			if(x >= .33 and x < .66 and len(children[i]) < int(sys.argv[4])):
				children[i].append(random.choice(moves))
			elif(x >= .66 and len(children[i])>0):
				del children[i][random.randrange(len(children[i]))]
			elif(len(children[i])>0):
				children[i][random.randrange(len(children[i]))] = random.choice(moves)
			else:
				children[i].append(random.choice(moves))
	return children

def nextGen(population,generation,gen,process):
	start = time.time()
	if(len(population) < int(sys.argv[3])):
		print("Population low before fitness: "+str(len(population)))
	ranked = fitness(population)
	if(len(ranked) < int(sys.argv[3])):
		print("Ranked low after fitness: "+str(len(ranked)))
	if(generation%10 == 0):
		print("Process: "+str(process)+"\nGeneration: "+str(generation)+"\nMatches: "+str(ranked[0][0]))
		# print(ranked[0][0])
		# print(time.time()-start)
		# start = time.time()
	#time.sleep(.05)
	children = matingPool(ranked)
	if(len(children) < int(sys.argv[3])):
		print("CHildren low after mating: "+str(len(ranked)))
	#print(len(children))
	nextGeneration = mutate(children)
	if(len(nextGeneration) < int(sys.argv[3])):
		print("nextGeneration low after mutate: "+str(len(ranked)))
	#print(len(nextGeneration))
	# if(generation%10 == 0):
	# 	print(time.time()-start)
	if generation < 50+gen:
		return nextGen(nextGeneration,generation+1,gen,process)
	else:
		return nextGeneration

def initialize(q,population,gen,process):
	if len(population) < 1:
		print("Making new population "+str(process))
		population = createPopulation(int(sys.argv[3]))
		print("Created population "+str(process))
	nextGeneration = nextGen(population,gen,gen,process)
	q.put(nextGeneration)

def reconstructRoute(reconstruct):
	#Initialize the route with the first in the sub-routes
	route = reconstruct[0]
	# plotRoute(route)
	del reconstruct[0]
	#Keep looping while there's sub-routes not inserted
	while len(reconstruct)>0:
		#Used to store data of the closest point
		closest = []
		#Loop through all the sub-routes
		for i in reconstruct:
			if i != route:
				#Get the distance from the last point in the route and first in sub-route
				distance = calculateDistancePoint(route[-1],i[0])
				#Store how to combine the route, 1 being the end
				points = [1,0]
				#If the distance of the first point in the route and last point in the sub-route are closer replace
				if(calculateDistancePoint(route[0],i[-1])<distance):
					distance = calculateDistancePoint(route[0],i[-1])
					points = [0,1]
				#store the sub-route if it's shorter than the known closest
				if(len(closest) == 0):
					closest = [distance,i,points]
				elif(closest[0]>distance):
					closest = [distance,i,points]
		#Remove the closest found sub-route from the list of sub-routes
		del reconstruct[reconstruct.index(closest[1])]
		#Insert sub-route into the beginning if the closest sub-route is before the route
		if(closest[2][0] == 0):
			z = 0
			for i in closest[1]:
				if(i not in route):
					route.insert(z,i)
					z=z+1
					
				else:
					print("Uh oh")
		#Else append the sub-route to the end
		elif(closest[2][0] == 1):
			for i in closest[1]:
				if(i not in route):
					route.append(i)
					
				else:
					print(closest[2][0])
		else:
			print(closest[2][0])
		# plotRoute(route)

	return route

def getWisdom(ranked):
	#Used to store all the connected points
	wise = {}
	#Store the max value
	max = 0
	#Loop through the top 30% of the total population
	for temp in range(0,int(len(ranked)*.3)):
		#Make the route a more user friendly variable
		route = ranked[temp][1]
		#Loop through the route
		for i in range(len(route)):
			#If the current city is at the beginning, then the prevCity is the last city
			if(i == 0):
				#See if the current combo of cities is in the wise
				if(route[len(route)-1]+route[i]+route[i+1] not in wise):
					#If not make a new object of the cities and initialize the count to 1
					wise[route[len(route)-1]+route[i]+route[i+1]] = {
						'city': route[i],
						'nextCity': route[i+1],
						'prevCity': route[len(route)-1],
						'count': 1
					}
				else:
					#Else increment the count by 1 making that combo "wiser"
					wise[route[len(route)-1]+route[i]+route[i+1]]['count'] = wise[route[len(route)-1]+route[i]+route[i+1]]['count']+1
				#Set the max if it's passed
				if(wise[route[len(route)-1]+route[i]+route[i+1]]['count']>max):
					max = wise[route[len(route)-1]+route[i]+route[i+1]]['count']
			#If we're in the middle nothing special has to happen
			elif(i != 0 and i != len(route)-1):
				#See if the current combo of cities is in the wise
				if(route[i-1]+route[i]+route[i+1] not in wise):
					#If not make a new object of the cities and initialize the count to 1
					wise[route[i-1]+route[i]+route[i+1]] = {
						'city': route[i],
						'nextCity': route[i+1],
						'prevCity': route[i-1],
						'count': 1
					}
				else:
					#Else increment the count by 1 making that combo "wiser"
					wise[route[i-1]+route[i]+route[i+1]]['count'] = wise[route[i-1]+route[i]+route[i+1]]['count']+1
				#Set the max if it's passed
				if(wise[route[i-1]+route[i]+route[i+1]]['count']>max):
					max = wise[route[i-1]+route[i]+route[i+1]]['count']
			#If we're at the end, next city is the first city
			elif(i==len(route)):
				#See if the current combo of cities is in the wise
				if(route[i-1]+route[i]+route[0] not in wise):
					#If not make a new object of the cities and initialize the count to 1
					wise[route[i-1]+route[i]+route[0]] = {
						'city': route[i],
						'nextCity': route[0],
						'prevCity': route[i-1],
						'count': 1
					}
				else:
					#Else increment the count by 1 making that combo "wiser"
					wise[route[i-1]+route[i]+route[0]]['count'] = wise[route[i-1]+route[i]+route[0]]['count']+1
				#Set the max if it's passed
				if(ise[route[i-1]+route[i]+route[0]]['count']>max):
					max = ise[route[i-1]+route[i]+route[0]]['count']


	#Initialize the array of sub-routes
	reconstructed = [[]]
	#make our temp min allowed the max
	i=max
	#Loop while above half the max and not every city has been inserted
	while i > int(max*.5):
		#Loop through everything in the wise json
		for index in wise:
			#If the value is above the min allowed continue
			if wise[index]['count'] >= i:
				#Loop through the sub-routes
				for y in range(0,len(reconstructed)):
					# print(len(reconstructed))
					# print(len(tempCities))
					# print(len(reconstructed[y]))
					#If we're at the very beginning, initiallize the first sub-route to the first allowed combo
					if len(reconstructed[y]) == 0 and len(reconstructed) == 1:
						reconstructed[y].append(wise[index]['prevCity'])
						reconstructed[y].append(wise[index]['city'])
						reconstructed[y].append(wise[index]['nextCity'])
						break
					#If prevCity of the current combo is at the end of the sub-route, and city and nextCity haven't been inserted, add to the end of the sub-route
					elif reconstructed[y][-1] == wise[index]['prevCity']:
						reconstructed[y].append(wise[index]['city'])
						reconstructed[y].append(wise[index]['nextCity'])
						break
					#If the current city is at the end of the sub-route and nextCity hasn't been inserted append nextCity
					elif reconstructed[y][-1] == wise[index]['city']:
						reconstructed[y].append(wise[index]['nextCity'])
						break
					#If the current city is at the beginning of the sub-route and prevCity hasn't been inserted yet, insert prevCity
					elif reconstructed[y][0] == wise[index]['city']:
						reconstructed[y].insert(0,wise[index]['prevCity'])
						break
					#If nextCity is at the beginning of the sub-route and prevCity and city haven't been inserted yet, insert both
					elif reconstructed[y][0] == wise[index]['nextCity']:
						reconstructed[y].insert(0,wise[index]['city'])
						reconstructed[y].insert(0,wise[index]['prevCity'])
						break
					#If none of the above works and none of the combo has been inserted yet, make a new sub-route
					else:
						reconstructed.append([])
						reconstructed[-1].append(wise[index]['prevCity'])
						reconstructed[-1].append(wise[index]['city'])
						reconstructed[-1].append(wise[index]['nextCity'])
						break
		#Decrement the min allowed
		i=i-1
	# print(len(reconstructed))
	# print(len(tempCities))
	#Get the wisest route, could be missing cities
	reconstruct = fitness(reconstructed)

	return reconstruct[0][1]

def multiPop(number,everyone,lastGen,gen,run):
	#Creating multi core functionality
	threads = []
	count = 0
	#Creates a shared memory space for the child and parent processes
	q = Queue()
	print ("Starting processes")
	if(len(everyone) > 0):
		random.shuffle(everyone)
	for i in range(number):
		#Creates a process for every starting city, allowing for concurrent finding
		if(len(everyone) < 1):
			threads.append(Process(target=initialize, args=(q,[],gen,count,)))
			#Start the child process
			threads[count].start()
		else:
			population = []
			amount = len(everyone)/number
			population = everyone[int(count*amount):int(count*amount+amount)]
			threads.append(Process(target=initialize, args=(q,population,gen,count,)))
			#Start the child process
			threads[count].start()

		#threads.append(threading.Thread(target=findShortestRoute, args=(1,q)))
		count+=1;

	#Stores the results of the child processes
	results= []
	print ("Waiting on processes")
	for i in threads:
		#print(q.get())
		#Gets the results from the child processes
		results.append(q.get())
		#Waits on the child processes to finish
		#i.join()

	#Once all the child processes have completed, loop through the results of each and find the shortest out of the shortest paths
	everyone = []
	for i in results:
		for y in i:
			everyone.append(y)
	ranked = fitness(everyone)
	print(ranked[0][1])
	if (len(lastGen) < 3 or lastGen.count(ranked[0][0]) < 3):
		lastGen.insert(0,ranked[0][0])
		if(len(lastGen) > 3):
			del lastGen[-1]
		return multiPop(number,everyone,lastGen,gen+50,run)
	else:
		wisest = getWisdom(ranked)
		print("Wisest: "+str(wisest)+" "+str(howClose(wisest)))
		print("Ranked: "+str(ranked[0][1])+" "+str(ranked[0][0]))
		if(howClose(wisest)>ranked[0][0]):
			return wisest
		else:
			return ranked[0][1]

global best

results = []
for i in range(int(sys.argv[1])):
	best = ["0,"+str(howClose(randomMoves()))]
	results.append(multiPop(int(sys.argv[2]),[],[],0,i))

avg = 0
min = 0
max = 0
dev = 0
closest = []
for i in results:
	if closest == []:
		closest = list(i)
		max = howClose(i)
	elif howClose(closest)<howClose(i):
		closest = list(i)
		max = howClose(i)
	if min == 0:
		min = howClose(i)
	elif min > howClose(i):
		min = howClose(i)
	avg += howClose(i)
avg/=int(sys.argv[2])
mean = 0
for i in results:
	length = howClose(i)
	mean+=((length-avg)**2)
dev=((mean/int(sys.argv[1]))**.5)

#Print out the result
print("Min: "+str(min)+" Max: "+str(max)+" Avg: "+str(avg)+" Deviation: "+str(dev))
print("Random Alg: ")
print(random_alg)
randomCube = pc.Cube()
randomCube(random_alg)
print(randomCube)
print("Closest: ")
space = " "
string = space.join(closest)
my_formula = pc.Formula(string)
randomCube(random_alg)
print(my_formula)
print(randomCube)
# print(len(route))
print("Distance: "+str(howClose(closest)))
#Print the time it took to complete
print("--- %s seconds ---" % (time.time() - start_time))
