
# -*- coding: utf-8 -*-
"""
Created on Sat Feb  3 20:13:31 2018

@author: Mehak Beri
"""

import pandas as pd
import information_gain as ig
import variance_impurity as vi
from random import *

def search(start,n):
    if((not start) or (start.id==n.id)):
        ans = start
        return ans
    else:
        if(start.zero):
            ans= search(start.zero,n)
            if (ans):
                return ans
        if(start.one):
            ans= search(start.one,n)
            if (ans):
                return ans
        else:
            ans= None   
            
def remove_from_list(id):
    global non_leaf_list
    found=False
    for node in non_leaf_list:
        if node.id==id:
            remove_node=node
            found=True
            break
    if(found):
        non_leaf_list.remove(remove_node)

def delete_subtree(start,toDel):
#    h.node_no-=1
    node=search(start,toDel)    
    if node.zero or node.one:
        delete_subtree(start,node.zero)
        delete_subtree(start,node.one)
        remove_from_list(node.id)   
    node.zero=None
    node.one=None

#def delete_subtree(start,toDel):
#    ig.node_no-=1
#    global non_leaf_list
#    non_leaf_list.remove(toDel)
#    node=search(start,toDel)
#    if(node):
#        node.zero=None
#        node.one=None
        
#       =======================================       
#    if node.zero or node.one:
#        delete_subtree(start,node.zero)
#        delete_subtree(start,node.one)
#        print("deleting node with id:",node.id," and data",node.data)
#        node=None
#        if search(start,toDel):
#            print("not deleted yet")
#    else:
#        print("deleting node with id:",node.id," and data",node.data)
#        node=None
#        if search(start,toDel):
#            print("not deleted yet")
    
def post_pruning(l,k,validation_set,h):
    print('constructing the decision tree....')
    h.node_no=0
    h.leaf=0
    h.node_list=[]
    h.leaf_list=[]
    global non_leaf_list
    d = h.decision_tree(training_set, 'Class', data_attributes )
    d_best = d
    d_best_accuracy = h.measure_accuracy(d_best, validation_set)
    print("initial accuracy:", d_best_accuracy)
    non_leaf_list=h.node_list[:]
    nl_list=h.node_list[:]
    for node in h.leaf_list:
        non_leaf_list.remove(node)
        nl_list.remove(node)
    for iterator in range(0,l):
        h.node_no=0
        d_dash = h.decision_tree(training_set, 'Class', data_attributes )
        m=randint(1, k)
        for j in range(1,m+1):
            n=len(non_leaf_list)
            if n<2:
                print("out of non leaf nodes!!")
                break
            p=randint(1, n-1)            
            chosen_node = non_leaf_list[p]
#            print("chosen subtree to be deleted starts from node:",chosen_node.data," with id:", chosen_node.id, " non-leaf:", len(non_leaf_list))
            delete_subtree(d_dash,chosen_node)
            cn_actual = search(d_dash,chosen_node)
#            print("after deletion, number of nodes left:",ig.node_no," now the node", cn_actual.data," has label",cn_actual.label, " non-leaf:", len(non_leaf_list))
        d_dash_accuracy = h.measure_accuracy(d_dash, validation_set)        
#        print("this iteration's accuracy :", d_dash_accuracy)
        non_leaf_list=nl_list[:] #replenish list
        if d_dash_accuracy > d_best_accuracy:
#            print("it is better!")
            d_best = d_dash
            d_best_accuracy=d_dash_accuracy
    
    return d_best,d_best_accuracy

if __name__ == '__main__':
    training_set=pd.read_csv("training_set.csv")
    data_attr= training_set.columns.values
    data_attributes= data_attr.tolist()
    data_attributes.remove('Class')
    validation_set=pd.read_csv("validation_set.csv")
    l = int(input("Enter l:"))
    k= int(input("Enter k:"))
    #heuristic
    h=ig
    root_pruned,pp_accuracy = post_pruning(l,k,validation_set,h)
    print("After pruning, the tree starting from node:",root_pruned.data,"has accuracy:", pp_accuracy,"%")

