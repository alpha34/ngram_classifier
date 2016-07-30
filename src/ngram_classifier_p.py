'''
Created on Sep 14, 2015
@author: uday
classifies protein sequences after computing ngram frequency vectors for sequences
input=fasta files of protein sequences
'''

import sys, matplotlib
from itertools import product
import numpy as np
from numpy import array, ones, zeros, concatenate, argsort, transpose, var
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.cross_validation import cross_val_score, train_test_split
from sklearn.svm import SVC
from sortedcontainers import SortedDict
import ngram_classifier_util as util
import matplotlib.pyplot as plt
import copy

threegrams = []
index_map = {}

# to remove frequency counts for 3 grams where the variance in freq count is zero
def remove_zeros(data):
    dataT = transpose(data)
    indices = [index for index, x in enumerate(dataT) if var(x) != 0]
    for i, index in enumerate(indices):
        index_map[i] = index
    data_trimmed = transpose(array([x for x in dataT if var(x) != 0]))
    return data_trimmed

def create_keys(n):
    keys = list(product(['A', 'I', 'L', 'F', 'V', 'Y', 'C', 'M', 'W', 'Q', 'H', 'S', 'T', 'G', 'R', 'K', 'D', 'E', 'P', 'N'], repeat=n))
    keys2 = [''.join(key) for key in keys]
    return sorted(keys2)

def get_key(position):
    return threegrams[position]

#calculate ngram frequencies for a given sequence
def calculate_ngram_frequencies(n, sequence):
    freq_vector = {}
    #initialize all frequencies to zero
    for key in threegrams:
        freq_vector[key]=0
        
    #compute normalized frequencies using standardized min-max normalization
    for index in range(len(sequence)-n+1):
        ngram_string = str(sequence[index:index+n])
        if util.contains_other_than('AILFVYCMWQHSTGRKDEPN', ngram_string):
            continue
        else:
            freq_vector[ngram_string]+=1

    return SortedDict(freq_vector)

def compute_frequency_matrix(n, sequences):
    matrix = []
    for sequence in sequences:
        if len(sequence) > 0:
            ngram_frequencies = calculate_ngram_frequencies(n, sequence)
            frequencies = ngram_frequencies.values()
            matrix.append(frequencies)     
    matrix = array([list(z) for z in matrix]).tolist()
    return matrix

def random_forest(data, labels, n_estimators):
    rfc = RandomForestClassifier(n_estimators=n_estimators, verbose=0)
    return rfc

def perform_svm(data, labels, C):
    svm = SVC(kernel='linear',C=C,probability=True)
    return svm

def perform_knn():
    knn = KNeighborsClassifier()
    return knn

def perform_naive_bayes():
    nb = GaussianNB()
    return nb

# extract important features for RF model
def features(model):
    importances = model.feature_importances_
    indices = argsort(importances)[::-1]

    # Print the feature ranking
    print("Feature ranking:")
    for f in range(10):
        index = indices[f]
        print("%d. feature %d %s (%f)" % (f + 1, index_map[index], threegrams[index_map[index]], importances[index]))

def test(model, data, labels, test_size, data_type, method, color):
    train_data, test_data, train_labels, test_labels = train_test_split(data, labels, test_size=test_size)
    model.fit(train_data, train_labels)
    accuracy = model.score(test_data, test_labels)
    print("mean accuracy: %0.2f " % accuracy)
    predicted_labels = predict(model, test_data)
    if method == "Random Forests":
        features(model)       
    util.roc(predicted_labels, test_labels, data_type, method, color)
    
def cross_validate(model, data, labels):
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
    
    data = concatenate([matrix1, matrix2])
    print("size of feature vector before zero variance check", len(data), " ", len(data[0]))
    data = remove_zeros(data)
    print("size of feature vector after zero variance check", len(data), " ", len(data[0]))

    len1 = len(matrix1)
    len2 = len(matrix2)
    one_s = ones(len1)
    zero_s = zeros(len2)

    labels = concatenate([one_s, zero_s])
    return data, labels
    
def get_data_from_files(file1, file2):
    sequences1 = util.get_sequences(file1)
    sequences2 = util.get_sequences(file2)
    data, labels = create_data(sequences1, sequences2)
    return data, labels    
    
# pipeline
if __name__ == '__main__':
    
    print("Python version:\n{}\n".format(sys.version))
    print("matplotlib version: {}".format(matplotlib.__version__))
    print(plt.style.available)
    print("numpy version: {}".format(np.__version__))
    
    input_file = sys.argv[1]
    output_dir = sys.argv[2]
    util.remove_old_output_files(output_dir)
    sys.stdout = open(util.generate_output_filename(output_dir), "w")
    lines = util.readFile(input_file)
    threegrams = create_keys(3)
    twograms = create_keys(2)
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
            util.start_roc_plot(analysis_type + '-' + protein)
            data, labels = get_data_from_files(file1, file2)
            accuracy = run(data, labels, analysis_type + '-' + protein)
            accuracies.append(accuracy)
            proteins.append(protein)
            print("&&& &&& &&& &&& &&& &&& &&& &&& &&&")
            accuracy = run(util.shuffle_data(data), labels, analysis_type + '-' + protein + '_shuffle')
            util.save_plot(output_dir, analysis_type + '-' + protein, 'svg', 1200)
            print("done processing ", line[2], line[3])
        else:
            util.plot_accuracies(accuracies, proteins, analysis_type, output_dir)
            proteins = []
            accuracies = []
            print("\n\n")