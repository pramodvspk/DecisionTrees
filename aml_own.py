from __future__ import division
import sys
import math
import random
import operator
import matplotlib.pyplot as plt

def find_entropy(train_data, class_column):
	classes_dict=count_classes_in_train_data(train_data,class_column)
	entropy=0
	for key in classes_dict.keys():
		probability=classes_dict[key]/len(train_data)
		entropy+=-probability*math.log(probability)
	return entropy

#This method splits the dataset and returns the dataset containing the rows which are greater or less-than-equal to the value passed depending upon the split type
def split_dataset(train_data,feature_no,value,split_type):
	return_subset = []
	if split_type == "eq":
		for data in train_data:
			if data[feature_no] == value:
				return_subset.append(data)
	elif split_type == "neq":
		for data in train_data:
			if data[feature_no] != value:
				return_subset.append(data)
	return return_subset

def find_best_attribute_to_split(train_data, class_column):
	best_feature=-1
	best_split_value=0
	base_entropy = find_entropy(train_data,class_column)
	best_info_gain=0
	for i in range(0,len(train_data[0])-1):
		feature_values_list=[data[i] for data in train_data]
		unique_feature_values_list = list(set(feature_values_list))
		#Iterating through the unique values of the features
		for j in range(0,len(unique_feature_values_list)):
			current_split_value = unique_feature_values_list[j]
			subset_right=split_dataset(train_data,i,current_split_value, "eq")
			subset_left=split_dataset(train_data,i,current_split_value, "neq")
			probability_right=len(subset_right)/len(train_data)
			probability_left=len(subset_left)/len(train_data)
			entropy_right=find_entropy(subset_right,class_column)
			entropy_left=find_entropy(subset_left,class_column)
			split_entropy=probability_right*entropy_right+probability_left*entropy_left
			info_gain=base_entropy-split_entropy
			#If the information gain is greater than the best information gain, then replace the best information gain and the best split value
			if info_gain>best_info_gain:
				best_info_gain=info_gain
				best_feature=i
				best_split_value=current_split_value
	return best_feature, best_split_value


#This method returns a dictionary with the class labels as keys and the total number of occurances of the class as value
def count_classes_in_train_data(train_data, class_column):
	train_data_length = len(train_data)
	classes_dict = classes_dict=dict((data[class_column],0) for data in train_data)
	for data in train_data:
		classes_dict[data[class_column]]+=1
	return classes_dict

#Returns the count of the most occuring item in the dictionary
def find_max_count_item_dict(dictionary):
	return max(dictionary.items(), key=operator.itemgetter(1))

#This method builds a decision tree
def build_decision_tree(train_data, feature_labels, class_column, required_depth, current_depth):

	#Find all the unique class labels
	class_labels = set(point[class_column] for point in train_data)
	if len(train_data) == 0:
		return None
	#Return the class label if all the class labels are same in the train dataset
	if len(class_labels) == 1:
		return class_labels.pop()
	#Return the max class label if only one attribute is present or the required depth is reached
	if(len(train_data[0]) == 2 or current_depth == required_depth):
		classes_dict = count_classes_in_train_data(train_data, class_column)
		return(find_max_count_item_dict(classes_dict)[0])
	best_feature,best_split_value = find_best_attribute_to_split(train_data, class_column)
	best_feature_label = feature_labels[best_feature-1]
	subset_right = split_dataset(train_data, best_feature, best_split_value, "eq")
	subset_left = split_dataset(train_data, best_feature, best_split_value, "neq")
	tree={best_feature_label:{}}
	tree[best_feature_label]["eq:" + str(best_split_value)] = build_decision_tree(subset_right, feature_labels, class_column, required_depth,current_depth + 1)
	tree[best_feature_label]["neq:" + str(best_split_value)] = build_decision_tree(subset_left, feature_labels, class_column, required_depth, current_depth + 1)
	return tree	
#This method predicts the labels	
def predict_labels(tree,testdata):
	#The node key ex: f19
	key = list(tree.keys())[0]
	#The child dictionary of the key 
	childTree = tree[key]
	#The index in the test data on which you have to check your value ex: 19
	compare_index = int(key[1:])
	#print(key, compare_index)
	classLabel=None
	#Iterating through the child tree
	for key in childTree.keys():
		#The child key contains neq:t, the operation is neq, and comparevalue is t
		operation=key.split(":")[0]
		compare_value=key.split(":")[1]
		if operation=="eq":
			#If the testdata feature value is equal to the compare value
			if testdata[compare_index]==compare_value:
				if type(childTree[key])== dict:
					classLabel = predict_labels(childTree[key],testdata)
				else:classLabel = childTree[key]
		elif operation=="neq":
			#If the testdata feature value is equal to the compare value
			if testdata[compare_index]!=compare_value:
				if type(childTree[key])==dict:
					classLabel = predict_labels(childTree[key],testdata)
				else: classLabel = childTree[key]
	return classLabel

#This method parses the file and returns the dataset
def parse_file(file_to_parsed):
	with open(file_to_parsed) as file:
		parsed_data = []
		file_content = file.read().splitlines()
		for line in file_content:
			line = line.split(",")
			parsed_data.append(line)
		#The last two lines are empty, so returing the list except the last two lines
		return parsed_data[:-2]

def implement_decision_tree(required_depth, train_data, test_data, class_column, fold_no):	
	#Opening the training datafile and appending the data into dataset list
	#Generating feature labels
	feature_labels = ["f"+str(i) for i in range(0,len(train_data[0])-1)]
	decision_tree = build_decision_tree(train_data, feature_labels, class_column, required_depth, 0)
	if(required_depth ==1 or required_depth == 2):
		print(decision_tree)
	key= list(decision_tree.keys())[0]
	total_predictions = 0
	false_predictions = 0
	correct_predictions = 0
	true_positives = 0
	false_postives = 0
	true_negatives = 0
	false_negatives = 0
	for data in test_data:
		predicted_label = predict_labels(decision_tree,data)
		actual_label = data[class_column]
		if predicted_label == actual_label:
			correct_predictions = correct_predictions + 1
			if actual_label == "won" and predicted_label == "won":
				true_positives = true_positives + 1
			if actual_label == 'nowin' and predicted_label == 'nowin':
				true_negatives = true_negatives + 1
		else:
			false_predictions = false_predictions + 1
			if actual_label == 'won' and predicted_label == 'nowin':
				false_negatives = false_negatives + 1
			if actual_label == 'nowin' and predicted_label == 'won':
				false_postives = false_postives + 1
		total_predictions = total_predictions + 1
	accuracy = correct_predictions/total_predictions
	misclassification_rate = 1-accuracy
	
	if(required_depth ==1 or required_depth == 2):
		print(required_depth)
		print("Confusion Matrix")
		print("------------")
		print("| " + str(true_negatives) + "|" + str(false_postives) + "|")
		print("------")
		print("| " + str(false_negatives) + "|" + str(true_positives) + "|")
		print("-------------")
	

	return accuracy
#Main function
def main():
	#take depth, no of cross folds, class column
	accuracy = []
	depths = [0]
	data_set = parse_file("kr-vs-kp.txt")
	initial_dict = count_classes_in_train_data(data_set,0)
	max_label_count = find_max_count_item_dict(initial_dict)[1]
	depth_0_accuracy = (max_label_count/len(data_set))*100
	accuracy.append(depth_0_accuracy)
	for depth in range(1,15):
		temp_accuracy = []
		cross_folds = 10
		class_column = 36
		data_set = parse_file("kr-vs-kp.txt")
		#Generating random numbers in the range of the length of the dataset
		random_number_list=random.sample(range(len(data_set)), len(data_set))
		#Depending on the number of crossfolds, determining the splitsize
		split_size=int(math.ceil(len(data_set)/cross_folds))
		for i in range(0,len(data_set), split_size):
			#Picking the test_data indices depending upon the split size from the randomly generated indices
			test_data_indices=random_number_list[i:i+split_size]
			#Picking train_data indices before the test data indices
			train_data_indices=random_number_list[:i]
			#Extending the train_data indices after the test data indices
			train_data_indices.extend(random_number_list[i+split_size:])
			#Building the training dataset
			train_data=[data_set[i] for i in train_data_indices]
			#Building the test dataset
			test_data=[data_set[i] for i in test_data_indices]

			return_accuracy = implement_decision_tree(depth, train_data, test_data,class_column, i)
			temp_accuracy.append(return_accuracy)
		average_accuracy = (sum(temp_accuracy)/cross_folds)*100
		accuracy.append(average_accuracy)
		depths.append(depth)
	print(depths,accuracy)
	plt.plot(depths,accuracy, color ='k', lw=2)
	plt.xlabel("Depths")
	plt.ylabel("Accuracy%")
	plt.savefig("OwnDataset.png")
	plt.close()
main()




