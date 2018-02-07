# -*- coding: utf-8 -*-
"""
Created on Mon Feb  5 01:13:37 2018

@author: Mehak Beri
"""
import pandas as pd
import information_gain as ig
import variance_impurity as vi
import post_pruning as pp

training=input("Please enter location for the training set: ")
training_set=pd.read_csv(training)
test=input("Please enter location for the test set: ")
test_set=pd.read_csv(test)
validation=input("Please enter location for the validation set: ")
validation_set=pd.read_csv(validation)
toPrint=input("Do you want to print the decision tree? <Yes/No>: ")
l=int(input("Enter L: integer (used in the post-pruning algorithm): "))
k=int(input("Enter K: integer (used in the post-pruning algorithm): "))

pp.training_set=training_set
data_attr= training_set.columns.values
data_attributes= data_attr.tolist()
data_attributes.remove('Class')
pp.data_attributes=data_attributes

#train model using info_gain
print("______________________________________________________________________")
print("Training model using info_gain..")
ig.node_no=0
ig.leaf=0
ig.node_list=[]
ig.leaf_list=[]
root_info = ig.decision_tree(training_set,'Class',data_attributes)
print("root:",root_info.data)
print("number of nodes:",ig.node_no)
print("number of leaf nodes:", ig.leaf)

#test accuracy on test set for info_gain
print("______________________________________________________________________")
print("Testing accuracy on test set for info_gain..")
m=ig.measure_accuracy(root_info, test_set)
print("Accuracy on this dataset is:",m,"%")

#test accuracy post pruning- info_gain
print("______________________________________________________________________") 
print("Testing accuracy on validation set post pruning- info_gain..")
r1,acc1 = pp.post_pruning(l,k,validation_set,ig)
print("After pruning, tree has accuracy:", acc1)
print("______________________________________________________________________") 
print("Testing accuracy on test set post pruning- info_gain..")
r1,acc1 = pp.post_pruning(l,k,test_set,ig)
print("After pruning, tree has accuracy:", acc1)

data_attr1= training_set.columns.values
data_attributes1= data_attr1.tolist()
data_attributes1.remove('Class')
pp.data_attributes=data_attributes1
#train model using variance_impurity
print("______________________________________________________________________") 
print("Training model using variance_impurity..")
vi.node_no=0
vi.leaf=0
vi.node_list=[]
vi.leaf_list=[]
root_var = vi.decision_tree(training_set, 'Class', data_attributes1 )
print("root:",root_var.data)
print("number of nodes:",vi.node_no)
print("number of leaf nodes:", vi.leaf)

#test accuracy on test set for variance_impurity
print("______________________________________________________________________")
print("Testing accuracy on test set for variance_impurity..")
m1=vi.measure_accuracy(root_var, test_set)
print("Accuracy on this dataset is:",m1,"%")

#test accuracy post pruning
print("______________________________________________________________________") 
print("Testing accuracy on validation set post pruning- variance_impurity..")
r2,acc2 = pp.post_pruning(l,k,validation_set,vi)
print("After pruning, tree has accuracy:", acc2)
print("______________________________________________________________________") 
print("Testing accuracy on test set post pruning- variance_impurity..")
r2,acc2 = pp.post_pruning(l,k,test_set,vi)
print("After pruning, tree has accuracy:", acc2)

#if to print is yes, print tree
if(toPrint=="Yes"):
    print("______________________________________________________________________") 
    print("Printing decision tree formed from information gain heuristic..")
    ig.print_tree("",root_info)
    print("______________________________________________________________________") 
    print("Printing decision tree formed from variance impurity heuristic..")
    ig.print_tree("",root_var)
