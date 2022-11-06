import pylab as pl
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from random import *
import pandas as pd

#FUNCTIONS
def AleaGen(q : int): # Function that generates all the possible combinations of q bits
    '''
    Input : q, the number of bits
    Output : a list of all the possible combinations of q bits
    '''
    tab = ["{:0b}".format(i) for i in range(2**q)] # List of all the possible combinations of q bits
    for k in range(0,len(tab)) :  # Loop on the combinations
        tab[k]= ((q-len(list(tab[k])))*['0']+list(tab[k])) # Add 0 to the combinations to have q bits
    return tab

def dif(A,B) :  # Function that calculates the Hamming distance between two combinations of bits
    '''
    Input : A and B are two combinations of bits
    Output : The Hamming distance between A and B   
    '''
    if len(A) == len(B) : # If the two combinations have the same length
        difference = 0 # Hamming distance
        for i in range(0,len(A)) :
            if A[i] != B[i] :
                difference += 1
    return difference/len(A) # Return the size-weighted Hamming distance

def s(A, B, alpha) : # Difference enhancer function
    '''
    Input : A and B are two combinations of bits, alpha is a float
    Output : The size-weighted Hamming distance between A and B multiplied by alpha
    '''
    return dif(A,B)**alpha

def mats(Populations, alpha=1): # Function that generates the affinity matrix of the system
    '''
    Input : Populations is a list of the populations of the system, alpha is a float
    Output : The affinity matrix of the system
    '''
    mat = np.zeros((len(Populations),len(Populations))) # Affinity matrix of the system
    for k in range(1,len(Populations)) :  # Loop on the populations of the system 
        if len(Populations[k]) != len(Populations[k-1]) : # If the two populations have different sizes
            print('Sequences of different lengths') # Print an error message
            break
    for i in range(0,len(Populations)) :  # Loop on the populations of the system 
        for j in range(0,len(Populations)) : 
            if i == j : # If the two populations are the same 
                continue    # Go to the next iteration
            mat[i,j] = s(Populations[i],Populations[j],alpha)   # Add the size-weighted Hamming distance between the two populations to the affinity matrix
    return mat # Return the affinity matrix

def div_score(list_seq, alpha=1):
    '''
    Input : list_seq is a list of sequences, alpha is a float
    Output : A tuple containing the maximum divergence with the group and the mean divergence of the group
    '''
    Mat = np.zeros((len(list_seq), len(list_seq))) # Matrix of zeros
    for seq1 in list_seq: # For each sequence in the list
        for seq2 in list_seq: # For each sequence in the list
            Mat[list_seq.index(seq1), list_seq.index(seq2)] = s(seq1, seq2, alpha) # Fill the matrix with the size-weighted Hamming distance between the two sequences multiplied by alpha
    
    div_score = np.array([])    
    for k in Mat :  # For each line of the matrix
        div_score = np.append(div_score, np.sum(k)/len(k))  # Calculate the divergence of each sequence from the group to the group     

    return np.round(np.sum(div_score)/(len(list_seq)-1),2)

def div_mutant(list_seq, query, alpha=1):
    '''
    Input : list_seq is a list of sequences, query is a sequence, alpha is a float
    Output : A tuple containing the divergence of the query and the maximum divergence with the group
    '''    
    score = np.array([]) # Array of zeros
    for seq in list_seq:    
        score = np.append(score,s(seq, query, alpha))   # Calculate the divergence of the query to the group

    return np.round(np.sum(score)/len(list_seq),2) , np.round(np.max(score),2) # Return the divergence of the query and the maximum divergence with the group

def Za(Z,t, Populations, params) : # Function that calculates the derivative of the populations of the system
    '''
    Input : Z is a list of the populations of the system, t is a float, Populations is a list of the populations of the system
    Output : The derivative of the populations of the system
    '''
    b, d, c, alpha = params
    z = sum(Z) # Total population
    S = mats(Populations, alpha) # Affinity matrix of the system
    res = [Z[k]*(b*(1+(sum(Z*S[k])/z))-d-c*z) for k in range(len(Populations))] # Derivative of the populations of the system
    return res  # Return the derivative of the populations of the system

def combinliste(seq : list, k : int) -> list: # Function that generates all the possible combinations of k elements in a list
    '''
    Input : seq is a list of sequences, k is an integer
    Output : A list of the k-combinations of the sequences
    '''
    if k == 0:
        return [[]]
    if len(seq) == 0:
        return []
    return [ [seq[0]] + x for x in combinliste(seq[1:], k-1) ] + combinliste(seq[1:], k)

def recup_data(recup : str) -> list:
    '''
    Input : recup is a string
    Output : A tuple containing the maximum divergence with the group and the mean divergence of the group
    '''
    recup = recup.replace(" ", "").split(",")
    return [float(k) for k in recup]

def clear_data(df : pd.DataFrame) -> pd.DataFrame:
    '''
    Input : df is a dataframe
    Output : A dataframe without the rows containing NaN
    '''    

    df = df.dropna(axis= 0, how= "all").dropna(axis= 1, how= "all") # Drop the rows and columns with only NaN values

    stock, cpt = [], df.iloc[:,0].min() 
    for j in df.columns:
        stock.append((j, cpt))
        cpt += 1

    for j, cpt in stock[::-1]: # Loop on the columns of the dataframe
        df = df.drop(df[df[j] > cpt].index)

    return df