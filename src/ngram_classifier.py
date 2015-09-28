'''
Created on Sep 14, 2015
@author: uday
classifies data using random forest, svm, after computing ngram frequency vectors for sequences
'''
from Bio import SeqIO
from itertools import product
import sys
from sklearn.ensemble import RandomForestClassifier
from sklearn.cross_validation import cross_val_score
from sklearn.svm import SVC
from numpy import array, ones, zeros, concatenate

def create_keys(n):
    keys = list(product(['A','C','G','T'],repeat=n))
    keys2 = [''.join(key) for key in keys]
    return keys2
    
#to check if an ngram contains other than 'ACGT' characters, ngram = set1, 'ACGT' = str1
def contains_other_than(str1, set1):
    return 1 in [c not in str1 for c in set1]

#calculate ngram frequencies for a given sequence
def calculate_ngram_frequencies(n, sequence):
    allkeys=create_keys(n)
    freq_vector = {}
    
    #initialize all frequencies to zero
    for key in allkeys:
        freq_vector[key]=0
        
    #compute frequencies
    for index in range(len(sequence)-n+1):
        ngram_string = str(sequence[index:index+n])
        if contains_other_than('ACGT', ngram_string):
            continue
        else:
            freq_vector[ngram_string]+=1
            
    return freq_vector

def compute_frequency_matrix(n, sequences):
    matrix=[]
    for sequence in sequences:
        ngram_frequencies=calculate_ngram_frequencies(n, sequence)
        frequencies= ngram_frequencies.values()
        matrix.append(frequencies)
    matrix = array([list(z) for z in matrix]).tolist()
    print('size of matrix ', len(matrix[0]), len(matrix))
    print(matrix[0])
    return matrix

def random_forest(data, labels, n_estimators):
    rfc = RandomForestClassifier(n_estimators=n_estimators,verbose=10)
    rfc.fit(data, labels)
    return rfc

def svm(data, labels):
    svm = SVC()
    svm.fit(data, labels)
    return svm
    
def cross_validate(rfc,data,labels):
    scores = cross_val_score(rfc,data,labels,cv=10,verbose=10)
    print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std()*2))    

def get_sequences(fasta_file):
    sequences = [x.seq for x in SeqIO.parse(fasta_file, "fasta")]
    return sequences
    
if __name__ == '__main__':
    sequences1 = get_sequences(sys.argv[1])
    print(sequences1[0])
    print(sequences1[-1])
    sequences2 = get_sequences(sys.argv[2])
    matrix1 = compute_frequency_matrix(3, sequences1)
    matrix2 = compute_frequency_matrix(3, sequences2)
    len1 = len(matrix1)
    len2 = len(matrix2)
    ones = ones(len1)
    zeros = zeros(len2)
    data=concatenate([matrix1,matrix2])
    labels=concatenate([ones,zeros])
    rfc=random_forest(data, labels, 50)
    cross_validate(rfc, data, labels)
    
    svm = svm(data, labels)
    cross_validate(svm, data, labels)
    #sequences3 = get_sequences(sys.argv[3])
    #test_data = compute_frequency_matrix(3, sequences3)
    #predictions=rfc.predict(test_data)
    #print(predictions)
    
    
    
    
    
