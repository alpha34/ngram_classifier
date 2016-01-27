'''
Created on Sep 14, 2015
@author: uday
classifies data using random forest & svm after computing ngram frequency vectors for sequences
input=fasta files of dna sequences
'''

import os
from Bio import SeqIO
from itertools import product
import sys
from sklearn.ensemble import RandomForestClassifier
from sklearn.cross_validation import cross_val_score, train_test_split
from sklearn import metrics
from sklearn.svm import SVC
from numpy import array, ones, zeros, concatenate
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from sklearn.neighbors import KNeighborsClassifier
from difflib import SequenceMatcher
from sklearn.metrics import roc_curve, auc
import csv

#use biopython SeqIO to parse fasta file
#returns a list of sequences
def get_sequences(fasta_file):
    sequences = [x.seq for x in SeqIO.parse(fasta_file, "fasta")]
    return sequences

def remove_duplicates(sequences):
    unique_seqs = set()
    for sequence in sequences:
        if sequence not in unique_seqs:
            unique_seqs.add(sequence)
    return list(unique_seqs)

#remove >95% identical sequences
def remove_similar_sequences(sequences):
    unique_sequences = []
    unique_sequences.append(sequences[0])

    for sequence in sequences[1:]:
        similar = False
        for unique_sequence in unique_sequences:
            similarity = SequenceMatcher(None, sequence, unique_sequence).ratio()
            if similarity>0.95:
                print("similarity ", similarity)
                similar = True
                break
        if similar==False:
            unique_sequences.append(sequence)
    return unique_sequences

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
        
    #compute normalized frequencies using standardized min-max normalization
    for index in range(len(sequence)-n+1):
        ngram_string = str(sequence[index:index+n])
        if contains_other_than('ACGT', ngram_string):
            continue
        else:
            freq_vector[ngram_string]+=1

    min_freq = min(freq_vector.values())
    max_freq = max(freq_vector.values())

    for key, value in freq_vector.items():
        freq_vector[key]=(float(value)-float(min_freq))/(max_freq-min_freq)

    return freq_vector

def compute_frequency_matrix(n, sequences):
    matrix=[]
    for sequence in sequences:
        if len(sequence)>0:
            ngram_frequencies=calculate_ngram_frequencies(n, sequence)
            frequencies= ngram_frequencies.values()
            matrix.append(frequencies)
    matrix = array([list(z) for z in matrix]).tolist()
    return matrix

def random_forest(data, labels, n_estimators):
    rfc = RandomForestClassifier(n_estimators=n_estimators,verbose=0)
    return rfc

def perform_svm(data, labels, C):
    svm = SVC(kernel='linear', C=C)
    return svm

def perform_knn():
    knn = KNeighborsClassifier()
    return knn

def test(model, data, labels, test_size, data_type, method, color):
    train_data, test_data, train_labels, test_labels =  train_test_split(data, labels, test_size=test_size)
    model.fit(train_data, train_labels)
    accuracy=model.score(test_data, test_labels)
    print("mean accuracy: %0.2f " % accuracy)
    predicted_labels = predict(model,test_data)
    roc(predicted_labels, test_labels, data_type,method, color)
    
def cross_validate(rfc,data,labels):
    scores = cross_val_score(rfc,data,labels,cv=10,verbose=0)
    print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std()))    

def predict(model, test_data):
    predictions=model.predict(test_data)
    return predictions

def roc(y_pred, y, data_type, method, color):
    false_positive_rate, true_positive_rate, thresholds = roc_curve(y, y_pred)
    roc_auc = auc(false_positive_rate, true_positive_rate)
    plt.plot(false_positive_rate,true_positive_rate,'b',color=color, label='auc for %s=%0.2f' %(method,roc_auc))

def start_plot(data_type):
    plt.title('roc for %s' %data_type)
    plt.plot([0,1], [0,1], 'r--',color='black')
    plt.xlim([-0.1, 1.2])
    plt.ylim([-0.1, 1.2])
    plt.ylabel('true positives')
    plt.xlabel('false positives')    

def plot_confusion_matrix(y_pred, y, data_type, method):
    plt.imshow(metrics.confusion_matrix(y, y_pred),cmap=cm.get_cmap('summer'),interpolation='nearest')
    plt.colorbar()
    plt.xlabel('true value')
    plt.ylabel('predicted value')
    plt.title('confusion matrix for %s using %s' %(data_type, method), y=1.05)
    plt.show()

def run(data, labels, data_type):
    start_plot(data_type)
    print("results from RF")
    rfc=random_forest(data, labels, 50)
    test(rfc,data,labels,0.33,data_type,'Random Forests','red')
    cross_validate(rfc,data,labels)
    print("results from SVM")
    this_svm = perform_svm(data, labels, 1)
    test(this_svm,data,labels,0.2,data_type,'SVM','green')
    cross_validate(this_svm,data,labels)
    print("results from KNN")
    knn = perform_knn()
    test(knn,data,labels,0.2,data_type,'KNN','blue')
    cross_validate(knn,data,labels)
    plt.legend(loc='lower right')
    #plt.show()
    save_plot(plt, 'C:\\uday\\gmu\\ngrams\\jan_2016_results\\', data_type)

def save_plot(plt, directory, file, ext='png'):
    # If the directory does not exist, create it
    if not os.path.exists(directory):
        os.makedirs(directory)
    filename = "%s.%s" % (file, ext)
    # The final path to save to
    savepath = os.path.join(directory, file)
    print("Saving figure to '%s'..." % savepath),

    # Actually save the figure
    plt.savefig(savepath)
    plt.close()
    
def create_data(sequences1, sequences2):
    print("size of data before duplicate check", len(sequences1), len(sequences2))
    sequences1 = remove_duplicates(sequences1)
    sequences2 = remove_duplicates(sequences2)
    print("size of data after duplicate check", len(sequences1), len(sequences2))
    #seqs1 = remove_similar_sequences(sequences1)
    #seqs2 = remove_similar_sequences(sequences2)
    #print("size of data after similarity check", len(seqs2))
    matrix1 = compute_frequency_matrix(3, sequences1)
    matrix2 = compute_frequency_matrix(3, sequences2)
    len1 = len(matrix1)
    len2 = len(matrix2)
    one_s = ones(len1)
    zero_s = zeros(len2)
    data=concatenate([matrix1,matrix2])
    labels=concatenate([one_s,zero_s])
    return data, labels
    
def get_data_from_files(file1, file2):
    # 2012, 2014
    sequences1 = get_sequences(file1)
    sequences2 = get_sequences(file2)
    data, labels = create_data(sequences1, sequences2)
    return data, labels    

def readFile(filename):
    f = open(filename)
    csv_f = csv.reader(f)
    return csv_f
    
#pipeline
if __name__ == '__main__':
    input_file = sys.argv[1]
    sys.stdout = open("C:\\uday\\gmu\\ngrams\\jan_2016_results\\ngrams_results.txt", "w")
    lines = readFile(input_file)
    for line in lines:
        print("start processing ",line)
        file1=line[0]
        file2=line[1]
        analysis_type=line[2]
        data, labels = get_data_from_files(file1, file2)
        run(data,labels,analysis_type)
        print("done processing ",line)