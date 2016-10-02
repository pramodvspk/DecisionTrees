from __future__ import division
import sys
import math
import operator
import matplotlib.pyplot as plt

#This method calculates the entropy of a dataset
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

#This method finds the best attribute to split and the best value in the best attribute
def find_best_attribute_to_split(train_data, class_column):
	best_feature=-1
	best_split_value=0
	base_entropy = find_entropy(train_data,class_column)
	best_info_gain=0
	for i in range(1,len(train_data[0])):
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
	#Find the set of all class labels
	class_labels = set(point[class_column] for point in train_data)
	#Return the class label if all the class labels are same in the train dataset
	if len(train_data) == 0:
		#print("Entering 0")
		return None
	#Return the clas
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

#This method parses the file and returns the dataset as a list
def parse_monk_file(file_to_parsed):
	with open(file_to_parsed) as file:
		parsed_data = []
		file_content = file.read().splitlines()
		for line in file_content:
			line = line.split(" ")[1:]
			line = line[:7]
			parsed_data.append(line)
		return parsed_data


def implement_decision_tree(required_depth, train_file, test_file, parse_function):
	#total_files = [["monks-1.train", "monks-1.test"], ["monks-2.train", "monks-2.test"], ["monks-3.train", "monks-3.test"]]
	#total_files = [["monks-1.train", "monks-1.test"]]
	#required_depth = int(sys.argv[1])
	
	#Opening the training datafile and appending the data into dataset list
	train_data = parse_function(train_file)
	#Generating feature labels
	feature_labels = ["f"+str(i) for i in range(1,len(train_data[0]))]
	decision_tree = build_decision_tree(train_data, feature_labels, 0, required_depth, 0)
	#Printing decision trees pf depth 2 and 3
	if(required_depth == 1 or required_depth == 2):
		print(decision_tree)
	key= list(decision_tree.keys())[0]
	test_data = parse_function(test_file)
	total_predictions = 0
	correct_predictions = 0
	true_positives = 0
	true_negatives = 0
	false_postives = 0
	false_negatives = 0
	for data in test_data:
		predicted_label = predict_labels(decision_tree,data)
		actual_label = data[0]
		if predicted_label == actual_label:
			correct_predictions = correct_predictions + 1
			if actual_label == '1' and predicted_label == '1':
				true_positives = true_positives + 1
			if actual_label == '0' and predicted_label == '0':
				true_negatives = true_negatives + 1
		else:
			if actual_label == '1' and predicted_label == '0':
				false_negatives = false_negatives + 1
			if actual_label == '0' and predicted_label == '1':
				false_postives = false_postives + 1
		total_predictions = total_predictions + 1
	accuracy = correct_predictions/total_predictions

	
	print("Accuracy is: " ,(correct_predictions/total_predictions)*100)
	
	print("Confusion Matrix")
	print("------------")
	print("| " + str(true_negatives) + "|" + str(false_postives) + "|")
	print("------")
	print("| " + str(false_negatives) + "|" + str(true_positives) + "|")
	print("-------------")
	

	return accuracy * 100

def main():
	total_files = [["monks-1.train", "monks-1.test"], ["monks-2.train", "monks-2.test"], ["monks-3.train", "monks-3.test"]]
	for file in total_files:
		depths = [0]
		accuracy = []
		parsed_train_file = parse_monk_file(file[0])
		initial_dict = count_classes_in_train_data(parsed_train_file,0)
		max_label_count = find_max_count_item_dict(initial_dict)[1]
		depth_0_accuracy = (max_label_count/len(parsed_train_file))*100
		accuracy.append(depth_0_accuracy)
		for required_depth in range(1,9):
			accuracy_percent = implement_decision_tree(required_depth,file[0], file[1], parse_monk_file)
			depths.append(required_depth)
			accuracy.append(accuracy_percent)
		plt.plot(depths,accuracy, color ='k', lw=2)
		plt.xlabel("Depths")
		plt.ylabel("Accuracy%")
		plt.savefig(file[1]+ ".png")
		plt.close()
main()




