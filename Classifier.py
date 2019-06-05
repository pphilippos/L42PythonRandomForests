#!/bin/env pypy

# Author: Philippos Papaphilippou
#
# Instructions:
#
#	First, run DatasetLoader.py to produce the dataset file
#	Run using $python Classifier.py <m> <k> <Impurity measure>
#		Where m: Number of trees
#			  k: depth of each tree (or size of subset of features for each tree)
#			  Impurity measure: 0 for Gini, 1 for Entropy, 2 for Misclassification
#	For multiple experiments use the Threaded.py script 
#
#	Example run: $python Classifier.py 50 20 2

import random
import math
import pickle
import sys

# Import the pre-prepared dataset
with open("dataset_dump.dat") as f:
	Samples, Subsets, Classes, GetClass, GetClassIndex, Array = pickle.load(f)

ImpurityMeasure = ("Gini-Index","Cross-Entropy","Misclassification")
itype = ImpurityMeasure[int(sys.argv[-1])]

# Random forest main algorithm
def RandomForest(Dataset, m, k):
	Trees = []	
	for tree in range(m):
		T = bootstrap(Dataset)
		tree = DecisionTree(T, k)
		Trees.append(tree)
	return Trees

# Function to be usd for sampling
def bootstrap(Dataset):
	replacement = True
	fraction = 0.66
	T=[]
	if replacement:
		for i in range(int(fraction*len(Dataset))):
			T.append(random.choice(Dataset))
	else:
		T = list(Dataset)
		random.shuffle(T)
		T=T[:int(fraction*len(Dataset))]
	return T

# Tree node data structure
class TreeNode:
	def __init__(self):
		self.children = []
		self.parent = None
		self.content = None
		self.depth = 0
		self.id = 0
	def add_content(self, content, k, features):
		self.content = content
		self.generate_children(k)
		self.features = features
	def generate_children(self, k):
		if k-1 == self.depth: 
			return
		classes_set=set()
		for partition in self.content[2]:
			for patient in partition:
				classes_set.add(GetClass[patient])
		if len(classes_set)==1:
			return
		for partition in self.content[2]:						
			self.add_child(TreeNode())
	def add_child(self, child):
		self.children.append(child)
		child.parent = self
		child.depth = self.depth + 1 
		child.id = len(self.children)-1

# Separate Decision Tree Classifier
def DecisionTree(T, k):
	features = len(Array[0])
	features = range(features)
	random.shuffle(features)	
	features = features[:k]
	
	BFSqueue = []
	
	results = []
	for f in features:
		results.append(best_split(T, f))
	results.sort()
	
	content = results[-1]
	features.remove(results[-1][-1])
	tree_root = TreeNode()

	tree_root.add_content(content, k, features)

	BFSqueue += tree_root.children
	
	while len(BFSqueue)!=0:
		node = BFSqueue[0]
		BFSqueue = BFSqueue[1:]
		
		partition = node.parent.content[2][node.id]
		if len(partition)==0:
			continue
			
		results = []

		features_ = list(node.parent.features)

		for f in features_:
			results.append(best_split(partition, f))
		results.sort()
	
		content = results[-1]
		features_.remove(results[-1][-1])
		
		node.add_content(content, k, features_)
		
		BFSqueue += node.children
	return tree_root
	
# Function that determines best split for a certain feature	
def best_split(T, feature_index):
	max_split= 0
	max_rate = float("-inf")
	Tsorted = [[Array[i][feature_index], i] for i in T]
	Tsorted.sort()
	
	for i in range(0, len(Tsorted)):
		rate = split_information_gain(Tsorted, i)

		if rate > max_rate:
			max_rate = rate
			max_split = i

	split_value = Tsorted[max_split][0]
	if len(Tsorted)>1:
		split_value = (Tsorted[max_split][0]+Tsorted[max_split-1][0])/2.0
	
	l = [Tsorted[i][1] for i in range(max_split)]
	r = [Tsorted[i][1] for i in range(max_split,len(T))]
	partitions = []
	for i in (l,r): 
		if len(i)!=0: 
			partitions.append(i)
	
	return (max_rate, split_value, partitions, feature_index)

# Function to calculate the information gain after a split
def split_information_gain(Ts, split_index):
	l = Ts[:split_index]
	r = Ts[split_index:]
	lenT = float(len(Ts))
	Impurity_l = impurity(l)
	Impurity_r = impurity(r)
	Impurity_before = 0 # Correct value: impurity(l+r) (doesn't affect the comparison of splits)
	Gain = Impurity_before - (len(l)/lenT*Impurity_l + len(r)/lenT*Impurity_r)	
	return Gain
	
# Function to calculate the impurity according to the selected type
def impurity(T):
	N = float(len(T))
	class_probabilities = [0 for c in Classes]
	for patient in T:
		class_probabilities[GetClassIndex[GetClass[patient[1]]]] += 1/N
	
	if itype == "Cross-Entropy":
		sum = 0
		for p in class_probabilities:
			if p!=0:
				sum += p * math.log(p, 2)
		return -sum
		
	elif itype == "Misclassification":
		return 1-max(class_probabilities)
		
	elif itype == "Gini-Index":
		sum = 0
		for p in class_probabilities:
			sum += p**2
		return 1-sum
	
# Calculates the distribution of each class in a dataset
def classes_prob(partition):
	probs = []
	for c in Classes:
		count = 0
		for patient in partition:
			if GetClass[patient]==c:
				count+=1
		if len(partition)==0:
			probs.append(0)
		else:
			probs.append(1.0*count/len(partition))
	return probs

# Predicts class from a tree estimator	
def predict_class(tree_root, patient):

	node = tree_root
	while 1:		
		
		feature_index = node.content[3]
		threshold = node.content[1]

		direction = 0
		if Array[patient][feature_index] >= threshold:
			direction=min(1, len(node.children)-1)

		if len(node.children)==0:
			probabilities = classes_prob(node.content[-2][direction])
			return Classes[probabilities.index(max(probabilities))]
		else:
			node = node.children[direction]

# Predicts class from a random forest estimator	
def predict_class_forest(forest, patient):
	votes = [0 for i in range(len(Classes))]
	
	for tree_root in forest:
		votes[GetClassIndex[predict_class(tree_root, patient)]] += 1

	return Classes[votes.index(max(votes))]

# Returns the misclassification rate from a forest
def Test_Error_Rate(test_set, forest):
	count = 0.0
	for patient in test_set:
		if GetClass[patient] != predict_class_forest(forest, patient):
			count+=1
	return count/len(test_set)

# Performs k-folds cross validation end returns the average true error
def k_folds_CrossValidation():
	meanTestRate = 0
	CV_k = 10
	iterations = 10
	
	for iteration in range(iterations):
		records = range(len(Samples))
		random.shuffle(records)
	
		fold = int(float(len(records))/CV_k+0.5)
		for i in range(CV_k):
			test_set = records[i*fold:min((i+1)*fold, len(records))]
			train_set = list(set(records).difference(set(test_set)))
			Forest = RandomForest(train_set, int(sys.argv[-3]), int(sys.argv[-2]))
			meanTestRate += Test_Error_Rate(test_set, Forest)
			
	meanTestRate /= CV_k*iterations
	
	return meanTestRate
	
# Performs the OOB cross validation end returns the average OOB error rate
def OOB_CrossValidation():
	import numpy as np
	iterations = 10
	OOBs = []
	train_set = range(len(Samples))
	for iteration in range(iterations):
		forest = RandomForest(train_set, int(sys.argv[-3]), int(sys.argv[-2]))
		mispredictions = 0
		for patient in train_set:
			votes = [0 for i in range(len(Classes))]
			for tree in forest:
				Tbootstrap = [val for sublist in tree.content[2] for val in sublist]
				if patient not in Tbootstrap:
					votes[GetClassIndex[predict_class(tree, patient)]]+=1
			if votes.index(max(votes))!=GetClassIndex[GetClass[patient]]:
				mispredictions+=1			
		OOB = float(mispredictions)/len(train_set)
		OOBs.append(OOB)
	OOBs = np.array(OOBs)
	return str(np.mean(OOBs)) + " " + str(np.std(OOBs))
	
#print k_folds_CrossValidation()
print OOB_CrossValidation()
