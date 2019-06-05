#!/bin/env python
#import numpy as np
import random
import math
import pickle

#np.core.arrayprint._line_width = 160
#np.set_printoptions(threshold=np.nan)

Names = dict()
Samples = []
Subsets = []
Classes = set()
GetClass = dict()
ClassesAll = []
Array = []

def import_file(filename):
	global Names
	global Samples
	global Subsets
	global Classes
	global Array
	ArrayLocal=[]
	ClassesLocal = []
	f = open(filename)
	lines = f.readlines()
	f.close()
	count = 0
	mode = 0
	for l in lines:	
		l=l.replace("null",'0')	
		line=l.split()
		if len(line)==0:
			continue
		if l[0]=='#' and "Value" in l:
			count+=1
			continue
		if l[0] in ('!', '^'):
			command = line[0][1:]

			if command == "dataset_table_begin":
				mode = 1
			elif command == "dataset_table_end":
				mode = 0
			elif command == "subset_description":
				Classes.add(l.split("=")[1].strip())
				ClassesLocal.append(l.split("=")[1].strip())
			elif command == "subset_sample_id":
				Subsets.append(set(l.split("=")[1].strip().split(',')))
		else:

			if mode == 1:
				Samples += line[2:2+count]
				mode = 2
			elif mode == 2:
				Names[line[0]]=line[1]
				ArrayLocal.append(line[2:2+count])

	for i in range(len(ArrayLocal)):
		for j in range(count):
			ArrayLocal[i][j]=float(ArrayLocal[i][j])	
	ArrayLocal = zip(*ArrayLocal)#np.array(ArrayLocal).T
	for j in range(count):
		Array.append(ArrayLocal[j])
		
	for c in range(len(ClassesLocal)):
		for element in Subsets[-len(ClassesLocal)+c]:
			GetClass[element]=ClassesLocal[c]
			GetClass[Samples.index(element)]=ClassesLocal[c]

import_file("GDS5401_full.soft")
import_file("GDS5402_full.soft")
import_file("GDS5403_full.soft")

Classes = list(Classes)
GetClassIndex = dict()
for c in Classes:
	GetClassIndex[c] = Classes.index(c)

with open("dataset_dump.dat", "wb") as f:
	pickle.dump((Samples, Subsets, Classes, GetClass, GetClassIndex, Array),f,protocol=pickle.HIGHEST_PROTOCOL)
