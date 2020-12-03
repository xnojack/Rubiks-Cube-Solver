import pycuber as pc
import random
import copy
import time

global faces
global moves
global solved
global random_alg

faces = ['U','L','F','R','B','D']
moves = ["U","L","F","R","B","D","U'","L'","F'","R'","B'","D'","U2","L2","F2","R2","B2","D2"]

solved = pc.Cube()
randomCube = pc.Cube()

alg = pc.Formula()
random_alg = alg.random()

def randomMoves():
	length = int(random.random()*100)
	sample = []
	while len(sample) < length:
		sample.append(moves[int(random.random()*len(moves))])
	return sample

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
			# if(cube.get_face(face)[i][0] == solved.get_face(face)[i][0]):
			# 	count = count + 1
			# if(cube.get_face(face)[i][1] == solved.get_face(face)[i][1]):
			# 	count = count + 1
			# if(cube.get_face(face)[i][2] == solved.get_face(face)[i][2]):
			# 	count = count + 1
	return count

samples = []
for y in range(0,500):
	samples.append(randomMoves())
start = time.time()
for x in range(0,500):
	howClose(samples[x])

print(time.time() - start)