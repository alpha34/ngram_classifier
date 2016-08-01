'''
Created on Sep 14, 2015
@author: uday
classifies data using random forest & svm after computing ngram frequency vectors for sequences
input=fasta files of dna sequences
'''

import sys
from Bio import SeqIO
from itertools import product
from numpy import array, ones, zeros, concatenate, argsort
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.cross_validation import cross_val_score, train_test_split
from sklearn.svm import SVC
import csv
import ngram_classifier_util as util
from sortedcontainers import SortedDict

threegrams=[]

def create_keys(n):
    keys = list(product(['A','C','G','T'],repeat=n))
    keys2 = [''.join(key) for key in keys]
    return keys2

def get_key(position):
    return threegrams[position]
    
#to check if an ngram contains other than 'ACGT' characters, ngram = set1, 'ACGT' = str1
def contains_other_than(str1, set1):
    return 1 in [c not in str1 for c in set1]

#calculate ngram frequencies for a given sequence
def calculate_ngram_frequencies(n, sequence):
    freq_vector = {}
    #initialize all frequencies to zero
    for key in threegrams:
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

    return SortedDict(freq_vector)

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
    svm = SVC(kernel='linear', C=C,probability=True)
    return svm

def perform_knn():
    knn = KNeighborsClassifier()
    return knn

def perform_naive_bayes():
    nb = GaussianNB()
    return nb
    
def test(model, data, labels, test_size, data_type, method, color):
    train_data, test_data, train_labels, test_labels =  train_test_split(data, labels, test_size=test_size)
    model.fit(train_data, train_labels)
    accuracy=model.score(test_data, test_labels)
    print("mean accuracy: %0.2f " % accuracy)
    predicted_labels = predict(model,test_data)
    if method=="Random Forests":       
        importances = model.feature_importances_
        indices = argsort(importances)[::-1]

        # Print the feature ranking
        print("Feature ranking:")
        for f in range(10):
            index=indices[f]
            print("%d. feature %d %s (%f)" % (f + 1, indices[f], threegrams[index], importances[indices[f]]))
    
def cross_validate(model,data,labels):
    scores = cross_val_score(model, data, labels, cv=10, verbose=0)
    accuracy = float("{0:.2f}".format(scores.mean()))
    print("Accuracy: ", accuracy)
    return accuracy   

def predict(model, test_data):
    predictions = model.predict_proba(test_data)
    print(predictions.shape)
    return predictions[:,1]

def run(data, labels, data_type, plots_folder='C:\\uday\\gmu\\ngrams\\july_2016_results\\'):
    if ('shuffle' in data_type):
        colors = ['red', 'lightsalmon', 'indianred', 'lightcoral']
    else:
        colors = ['chartreuse', 'forestgreen', 'green', 'olivedrab']
        
    print("results from RF")
    rfc = random_forest(data, labels, 50)
    test(rfc, data, labels, 0.33, data_type, 'Random Forests', colors[0])
    rfc_accuracy = cross_validate(rfc, data, labels)
    print("results from SVM")
    this_svm = perform_svm(data, labels, 1)
    test(this_svm, data, labels, 0.33, data_type, 'SVM', colors[1])
    svm_accuracy = cross_validate(this_svm, data, labels)
    print("results from KNN")
    knn = perform_knn()
    test(knn, data, labels, 0.33, data_type, 'KNN', colors[2])
    knn_accuracy = cross_validate(knn, data, labels)
    print("results from Gaussian Naive Bayes")
    nb = perform_naive_bayes()
    test(nb, data, labels, 0.33, data_type, 'Naive Bayes', colors[3])
    nb_accuracy = cross_validate(nb, data, labels)
    
    accuracies = [rfc_accuracy, svm_accuracy, knn_accuracy, nb_accuracy]
    return accuracies 

def create_data(sequences1, sequences2):
    print("size of data before same sizing", len(sequences1), len(sequences2))
    sequences1, sequences2 = util.create_same_size_sequences(sequences1, sequences2)
    print("size of data after same sizing", len(sequences1), len(sequences2))
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
    sequences1 = util.get_sequences(file1)
    sequences2 = util.get_sequences(file2)
    data, labels = create_data(sequences1, sequences2)
    return data, labels    

def get_m1_m2_data_from_files(file1, file2):
    m1_sequences_1, m2_sequences_1 = get_m1_m2_sequences(file1)
    m1_sequences_2, m2_sequences_2 = get_m1_m2_sequences(file2)    
    data1, labels1 = create_data(m1_sequences_1, m1_sequences_2)
    data2, labels2 = create_data(m2_sequences_1, m2_sequences_2)    
    return data1, labels1, data2, labels2
    

def get_m1_m2_sequences(fasta_file):
    m1_sequences = [x.seq for x in SeqIO.parse(fasta_file, "fasta") if len(x.description.split("|"))>3 and x.description.split("|")[3].split(":")[1]=='M1']
    m2_sequences = [x.seq for x in SeqIO.parse(fasta_file, "fasta") if len(x.description.split("|"))>3 and x.description.split("|")[3].split(":")[1]=='M2']
    return m1_sequences, m2_sequences

def readFile(filename):
    f = open(filename)
    csv_f = csv.reader(f)
    return csv_f
    
#pipeline
if __name__ == '__main__':
    input_file = sys.argv[1]
    output_dir = sys.argv[2]
    util.remove_old_output_files(output_dir)
    sys.stdout = open(util.generate_output_filename(output_dir, basename='dna_results'), "w")
    lines = util.readFile(input_file)
    threegrams = create_keys(3)
    proteins = []
    accuracies = []
    
    for line in lines:
        if line:
            print("-----------------------------------------")
            print(line[0], line[1], line[2], line[3])
            print("start processing ", line[2], line[3])
            file1 = line[0]
            file2 = line[1]
            analysis_type = line[2]
            protein = line[3]

            if (file1.find("_m_")>0):
                data1, labels1, data2, labels2 = get_m1_m2_data_from_files(file1, file2)
                accuracy=run(data1,labels1,analysis_type+'-M1')
                accuracies.append(accuracy)
                proteins.append('M1')
                accuracy=run(data2,labels2,analysis_type+'-M2')
                accuracies.append(accuracy)
                proteins.append('M2')
            else:            
                data, labels = get_data_from_files(file1, file2)
                accuracy=run(data, labels, analysis_type + '-' + protein)
                accuracies.append(accuracy)
                proteins.append(protein)
        else:
            util.plot_accuracies(accuracies, proteins, analysis_type, output_dir)
            proteins = []
            accuracies = []
            print("\n\n")
        