# -*- coding: utf-8 -*-
"""
Created on Sat Feb  3 16:48:23 2018

@author: Mehak Beri
"""

import pandas as pd

def find_variance_impurity(data):
    
    freq= data.value_counts()
    if(0 not in freq.keys()):
        freq[0]=0
    if(1 not in freq.keys()):
        freq[1]=0
        
    totalVals= freq[0] + freq[1]
    val1= freq[0]/totalVals
    val2= freq[1]/totalVals    
    S = val1*val2   
    return S

def calculate_gain(S,A,a,dataset):
    freq = A.value_counts()       
    row_zero = dataset.loc[dataset[a]==0, [a,"Class"]]
    row_one = dataset.loc[dataset[a]==1, [a,"Class"]]
    
    if(0 not in freq.keys()):
        freq[0]=0
        variance_zero=0
    else:
        variance_zero = find_variance_impurity(row_zero['Class'])
    if(1 not in freq.keys()):
        freq[1]=0
        variance_one=0
    else:
        variance_one = find_variance_impurity(row_one['Class']) 
    total=freq[0]+freq[1]
    gain = S- ( (freq[0]/total)*(variance_zero) + (freq[1]/total)*(variance_one))

    return gain
    
def find_gain(dataset, attributes):
    #entropy of all data
    S= find_variance_impurity(dataset['Class'])
    gain={}
    for attribute in attributes:
        gain[attribute] = calculate_gain(S,dataset[attribute],attribute, dataset)
    return gain

def sort_gain(Info_gain):
    sorted_gain = sorted(Info_gain, key=lambda x: Info_gain[x])
#    for k in sorted_gain:
#      print("{} : {}".format(k, Info_gain[k]))
    return sorted_gain[-1]

class Node(object):
    def __init__(self, attribute):
        self.data = attribute
        self.label= None
        self.zero = None #left=zero=negative
        self.one = None #right=one=positive
        
#recursive loop for decision tree
def decision_tree(dataset, target, attributes):
    #create a root node for the tree
    global node_no
    global leaf
    global node_list
    global leaf_list
    
    #if number of predicting attr is empty, then Return the single node tree Root,
    #with label = most common value of the target attribute in the examples
    root=Node('root')
    node_no= node_no+1
    root.id=node_no
#    print(root.id)
    node_list.append(root)     
    freq= dataset[target].value_counts()
    
    # if all examples are positive => there is no 0 , return single node tree with label 1
    if (0 not in freq.keys()):        
        root.label=1
        leaf +=1
        leaf_list.append(root)
#        print("base case 1: all ex positive, return with label:",root.label)
        return root
    #if all examples negative=> no 1, return root with label 0
    if(1 not in freq.keys()):        
        root.label=0
        leaf +=1
        leaf_list.append(root)
#        print("base case 2: all ex negative, return with label:",root.label)
        return root 
    if attributes==[]:
#        print("case 3: no more attributes")
        return root
    else:
        Info_gain=find_gain(dataset, attributes)   
        root_calculated= sort_gain(Info_gain)
        root.data= root_calculated    
        attributes_m= attributes[:]        
#        print("Node number",node_no, " is:",root.data)
        if freq[0]>freq[1]:
            root.label=0
        else:
            root.label=1    
        root.zero = Node('newNode')
        data_zero = dataset.loc[dataset[root.data]==0]
        if data_zero.empty:
#          print("data empty on zero side, label should be", root.label)
          root.zero.label= root.label
          leaf +=1
          leaf_list.append(root)
          node_no= node_no+1
          root.zero.id=node_no
#          print(root.id)
          node_list.append(root.zero)
          return root.zero
        else:
#            print("zero side: recursing after removing attr",root.data)
            attributes.remove(root.data)
            root.zero= decision_tree(data_zero, target, attributes)
        
        root.one = Node('newNode2')
        data_one= dataset.loc[dataset[root.data]==1]
        if data_one.empty:
#          print("data empty on one side, label should be", root.label)
          root.one.label= root.label
          leaf +=1
          leaf_list.append(root)
          node_no= node_no+1
          root.one.id=node_no
#          print(root.id)
          node_list.append(root.one)
          return root.one
        else:
#            print("one side: recursing after removing attr",root.data)
            attributes_m.remove(root.data)
            root.one= decision_tree(data_one, target, attributes_m)
    return root
    
def measure_accuracy(start_node, dataset):
    columns= dataset.shape[1]
    rows= dataset.shape[0]
    correct=0
    start=start_node
    for i in range(0,rows):
        target = dataset.iloc[i,columns-1]
        while(start_node.zero or start_node.one):
            val= dataset.loc[i,[start_node.data]]
            if(val.iloc[0]==0):
                start_node=start_node.zero
            else:
                start_node=start_node.one
        predicted_target = start_node.label
        start_node=start
        if target == predicted_target:
            correct +=1
#        else:
#            print("check row:",i,"",dataset.iloc[i,])
    accuracy= (correct/rows)*100     
    return accuracy

def print_tree(prefix,root):
    if(not root.zero):
        print ("",root.label)   
    else:
        if(root.zero.zero): 
            prefix+="| "
            print(prefix,root.data,"= 0 :")              
        else:    
            prefix+="| "
            print(prefix,root.data,"= 0 :",end="")
        
        print_tree(prefix,root.zero)        
        if(root.one.one):                     
            print(prefix,root.data,"= 1 :")   
        else:            
            print(prefix,root.data,"= 1 :",end="")
        print_tree(prefix, root.one)
    
if __name__ == '__main__':
    training_set=pd.read_csv("training_set.csv")
    data_attr= training_set.columns.values
    data_attributes= data_attr.tolist()
    data_attributes.remove('Class')
    node_no=0
    leaf=0
    node_list=[]
    leaf_list=[]
    root = decision_tree(training_set, 'Class', data_attributes )
    print("=========================================================")
    print("root:",root.data)
    print("number of nodes:",node_no)
    print("number of leaf nodes:", leaf)
    non_leaf_list=node_list[:]
    for node in leaf_list:
        non_leaf_list.remove(node)       
     
    print("=========================================================")
    print("Now measuring accuracy")
    test_set=pd.read_csv("test_set.csv")
    m=measure_accuracy(root, test_set)
    print("Accuracy on this dataset is:",m,"%")
    
#test tree
#    wesley= Node("Wesley")
#    honor = Node("Honor")
#    barcly= Node("Barcly")
#    tea= Node("Tea")
#    wesley.zero=honor
#    wesley.one= Node("new")
#    wesley.one.label = 0
#    honor.zero=barcly
#    honor.one=tea
#    barcly.zero = Node("new")
#    barcly.zero.label=1
#    barcly.one = Node("new")
#    barcly.one.label=0
#    tea.zero = Node("new")
#    tea.zero.label=1
#    tea.one = Node("new")
#    tea.one.label=0

#    print_tree("",root)