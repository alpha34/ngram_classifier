'''
Created on Sep 14, 2015
@author: uday
classifies data using random forest & svm after computing ngram frequency vectors for sequences
input=fasta files of dna sequences
'''

from Bio import SeqIO
from itertools import product
import sys
from sklearn.ensemble import RandomForestClassifier
from sklearn.cross_validation import cross_val_score, train_test_split
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
    rfc = RandomForestClassifier(n_estimators=n_estimators,verbose=0)
    return rfc

def perform_svm(data, labels, C):
    svm = SVC(kernel='linear', C=C)
    return svm

def test(model, data, labels, test_size):
    train_data, test_data, train_labels, test_labels =  train_test_split(data, labels, test_size=test_size)
    model.fit(train_data, train_labels)
    accuracy=model.score(test_data, test_labels)
    print("mean accuracy: %0.2f " % accuracy)
    
def cross_validate(rfc,data,labels):
    scores = cross_val_score(rfc,data,labels,cv=10,verbose=0)
    print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std()*2))    

def predict(model, test_data):
    predictions=model.predict(test_data)
    print(predictions)

def get_sequences(fasta_file):
    sequences = [x.seq for x in SeqIO.parse(fasta_file, "fasta")]
    return sequences

def run(data, labels):
    print("results from RF")
    rfc=random_forest(data, labels, 50)
    test(rfc,data,labels,0.2)
    print("results from SVM")
    this_svm = perform_svm(data, labels, 1)
    test(this_svm,data,labels,0.2)

if __name__ == '__main__':
    sequences1 = get_sequences(sys.argv[1])
    sequences2 = get_sequences(sys.argv[2])
    matrix1 = compute_frequency_matrix(3, sequences1)
    matrix2 = compute_frequency_matrix(3, sequences2)
    len1 = len(matrix1)
    len2 = len(matrix2)
    ones = ones(len1)
    zeros = zeros(len2)
    data=concatenate([matrix1,matrix2])
    labels=concatenate([ones,zeros])
    run(data,labels)
    


    
    
    
    
    
