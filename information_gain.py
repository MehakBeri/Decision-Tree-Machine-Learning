# -*- coding: utf-8 -*-
"""
Created on Fri Feb  2 10:35:51 2018

@author: Mehak Beri
"""

import pandas as pd
import math

def find_entropy(data):
    
    freq= data.value_counts()
    if(0 not in freq.keys()):
        freq[0]=0
    if(1 not in freq.keys()):
        freq[1]=0
        
    totalVals= freq[0] + freq[1]
    val1= freq[0]/totalVals
    val2= freq[1]/totalVals
    
    if(freq[0]==0):        
        val1=1
    if(freq[1]==0):        
        val2=1
    
    S = -((val1)*(math.log2(val1)) + (val2)*(math.log2(val2)) )    
    return S

def calculate_gain(S,A,a,dataset):
    freq = A.value_counts()       
    row_zero = dataset.loc[dataset[a]==0, [a,"Class"]]
    row_one = dataset.loc[dataset[a]==1, [a,"Class"]]
    
    if(0 not in freq.keys()):
        freq[0]=0
        entropy_zero=0
    else:
        entropy_zero = find_entropy(row_zero['Class'])
    if(1 not in freq.keys()):
        freq[1]=0
        entropy_one=0
    else:
        entropy_one = find_entropy(row_one['Class']) 
    total=freq[0]+freq[1]
    gain = S- ( (freq[0]/total)*(entropy_zero) + (freq[1]/total)*(entropy_one))

    return gain
    
def find_information_gain(dataset):
    #entropy of all data
    S= find_entropy(dataset['Class'])
    gain={}
    for attribute in dataset.columns:    
        if(attribute=='Class'):
            continue
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
        self.id=None
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
    Info_gain=find_information_gain(dataset)
    root=Node('root')
    root_calculated= sort_gain(Info_gain)
    root.data= root_calculated
    node_no= node_no+1
    root.id=node_no
#    print("Node number",node_no, " is:",root.data)
    node_list.append(root)
    #base cases
    freq= dataset[target].value_counts()
   
    # if all examples are positive => there is no 0 , return single node tree with label 1
    if (0 not in freq.keys()):
        root.label=1
        leaf +=1
        leaf_list.append(root)
#        print("Leaf number",leaf, " is:",root.data, " with label:",root.label)
        return root
    #if all examples negative=> no 1, return root with label 0
    if(1 not in freq.keys()):
        root.label=0
        leaf +=1
        leaf_list.append(root)
#        print("Leaf number",leaf, " is:",root.data, " with label:",root.label)
        return root
    #if number of predicting attr is empty, then Return the single node tree Root,
    #with label = most common value of the target attribute in the examples
    if freq[0]>freq[1]:
        root.label=0
    else:
        root.label=1
    if attributes==[]:
        if freq[0]>freq[1]:
            root.label=0         
            leaf +=1
            leaf_list.append(root)
#            print("Leaf number",leaf, " is:",root.data, " with label:",root.label)
            return root
        if freq[1]>freq[0]:
            root.label=1
            leaf +=1
            leaf_list.append(root)
#            print("Leaf number",leaf, " is:",root.data, " with label:",root.label)
            return root
    #recursive calls
    else:
        newNode = Node('newNode')
        root.zero = newNode
        data_zero = dataset.loc[dataset[root.data]==0]
        if data_zero.empty:
            if freq[0]>freq[1]:
              newNode.label=0
              leaf +=1
              leaf_list.append(root)
#              print("Leaf number",leaf, " is:",root.data, " with label:",root.label)
            if freq[1]>freq[0]:
              newNode.label=1
              leaf +=1
              leaf_list.append(root)
#              print("Leaf number",leaf, " is:",root.data, " with label:",root.label)
        else:
            if root in attributes:
                attributes.remove(root)
            root.zero= decision_tree(data_zero, target, attributes)
#            print('node:',root.data,' zero side:', root.zero.data)
        newNode2 = Node('newNode2')
        root.one = newNode2
        data_one= dataset.loc[dataset[root.data]==1]
        if data_one.empty:
            if freq[0]>freq[1]:
              newNode2.label=0
              leaf +=1
              leaf_list.append(root)
#              print("Leaf number",leaf, " is:",root.data, " with label:",root.label)
            if freq[1]>freq[0]:
              newNode2.label=1
              leaf +=1
              leaf_list.append(root)
#              print("Leaf number",leaf, " is:",root.data, " with label:",root.label)
        else:
            if root in attributes:
                attributes.remove(root)
            root.one= decision_tree(data_one, target, attributes)
#            print('node:',root.data,' one side:', root.one.data)
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
#    print("root.left",root.zero.data," | root.right",root.one.data)
#    l= root.zero
#    r=root.one
#    print(l.data," has left:",l.zero.data," | ", l.data," has right:", l.one.data)
#    print(r.data," has left:",r.zero.data," | ", r.data," has right:", r.one.data)
#    l1= l.zero
#    l2=l.one
#    r1=r.zero
#    r2=r.one
#    print(l1.data," has left:",l1.zero.data," | ", l1.data," has right:", l1.one.data)
#    print(l2.data," has left:",l2.zero.data," | ", l2.data," has right:", l2.one.data)
#    print(r1.data," has left:",r1.zero.data," | ", r1.data," has right:", r1.one.data)
#    print(r2.data," has left:",r2.zero.data," | ", r2.data," has right:", r2.one.data)
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

    print_tree("",root)
    
    
    
    








    


