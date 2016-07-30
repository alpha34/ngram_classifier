'''
Created on Jul 20, 2016
@author: uday
'''
from itertools import product
from Bio import SeqIO, AlignIO
from difflib import SequenceMatcher
import os, csv, datetime, random, matplotlib
from sklearn import metrics
from bokeh.charts import Bar, output_file, show
from sklearn.metrics import roc_curve, auc
import matplotlib.cm as cm
import matplotlib.pyplot as plt
from numpy import arange, linspace
from Bio.SeqUtils import ProtParam
from sortedcontainers import SortedDict
from sys import argv
from scipy.interpolate import spline

#use biopython SeqIO to parse fasta file
#returns a list of sequences
def get_sequences(fasta_file):
    sequences = [x.seq for x in SeqIO.parse(fasta_file, "fasta")]
    return sequences

def get_sub_sequences(fasta_file, start,end):
    alignments = AlignIO.read(fasta_file, 'fasta')
    for alignment in alignments:
        print(alignment.seq[start:end])

#reduce alphabet of sequence using Wang & Wang; 1999
def reduce_alphabet_WW99(sequence):
    new_sequence=[convert_WW99(x) for x in sequence]
    return new_sequence

#reduce alphabet of sequence using Wang & Wang; 1999
def reduce_alphabet_Li(sequence):
    new_sequence=[convert_Li(x) for x in sequence]
    return new_sequence

#reduce alphabet of sequence using Wang & Wang; 1999
def reduce_alphabet_Mekler(sequence):
    new_sequence=[convert_mekler(x) for x in sequence]
    return new_sequence

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

#to check if an ngram contains other than 'ACGT' characters, ngram = set1, 'ACGT' = str1
def contains_other_than(str1, set1):
    return 1 in [c not in str1 for c in set1]

#do conversion for a single aa residue using Wang & Wang; 1999
def convert_WW99(original):
    types = {'X' : ['R','K','D','E','P', 'N'],     ## surface
             'Y' : ['Q','H','S','T','G'], ## neutral
             'Z' : ['A','I','L','F','V','Y','C','M','W'] }   ## buried
    
    result=''
    for t in types.keys():
        if original in types[t]:
            result = t
    if (not result):
        result = 'U'
    return result

#Li et al
def convert_Li(original):
    types = {'X' : ['C','F','Y','W','M','L','I','V'],    ## surface
             'Y' : ['G','P','A','T','S'], 
             'Z' : ['N','H','Q','E','D','R','K'] }   ## buried
    
    result=''
    for t in types.keys():
        if original in types[t]:
            result = t
    if (not result):
        result = 'U'
    return result

#Mekler et al
def convert_mekler(original):
    types = {'X' : ['M','H','V','Y','N','D','I'],    
             'Y' : ['Q','L','E','K','F'], 
             'Z' : ['W','P','R','G','S','A','T','C'] }
    
    result=''
    for t in types.keys():
        if original in types[t]:
            result = t
    if (not result):
        result = 'U'
    return result

def create_keys_xyz(n):
    keys = list(product(['X','Y','Z'],repeat=n))
    keys2 = [''.join(key) for key in keys]
    return keys2

def create_same_size_sequences(sequences1, sequences2):
    len1=len(sequences1)
    len2=len(sequences2)
    if (len1<len2):
        sequences2=sequences2[:len1]
    if (len1>len2):
        sequences1=sequences1[:len2]
    return sequences1, sequences2

def readFile(filename):
    f = open(filename)
    csv_f = csv.reader(f)
    return csv_f

def generate_output_filename(output_dir, basename='protein_results'):
    suffix = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = "_".join([basename, suffix])
    output_filename=output_dir+"\\"+filename+".txt"
    return output_filename

def remove_old_output_files(folder):
    for the_file in os.listdir(folder):
        file_path = os.path.join(folder, the_file)
        try:
            if os.path.isfile(file_path):
                os.unlink(file_path)
        except Exception as e:
            print(e)

def extract_protein_name(line_in_input_file):
    if ('ha' in line_in_input_file):
        return 'ha'
    else:
        if ('na' in line_in_input_file):
            return 'na'
        else:
            return 'p4'

def shuffle_data(data):
    for row in data:
        random.shuffle(row)
    return data

def shuffle_labels(labels):
    random.shuffle(labels)
    return labels

def roc(y_pred, y, data_type, method, color):    
    false_positive_rate, true_positive_rate, thresholds = roc_curve(y, y_pred, drop_intermediate=False)
    roc_auc = auc(false_positive_rate, true_positive_rate)
    if ('shuffle' in data_type):
        method = method + '-CONTROL'
    plt.plot(false_positive_rate, true_positive_rate, color=color, alpha=0.7, label='%s (%0.2f)' %(method,roc_auc))
    plt.legend(loc="lower right", prop={'size':11})

def start_roc_plot(data_type):
    plt.style.use(['seaborn-white', 'seaborn-paper'])
    matplotlib.rc("font", family="Times New Roman")
    plt.figure()
    plt.title('roc for %s' % data_type, fontsize=24)
    plt.plot([0, 1], [0, 1], 'k--', color='black')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.ylabel('true positive rate', fontsize=24)
    plt.xlabel('false positive rate', fontsize=24)
    plt.tick_params(axis='y', labelsize=20)
    plt.tick_params(axis='x', labelsize=20)    

def start_bar_plot(data_type):
    plt.style.use(['seaborn-white', 'seaborn-paper'])
    matplotlib.rc("font", family="Times New Roman")
    fig, ax = plt.subplots()
    
    return fig, ax

def plot_confusion_matrix(y_pred, y, data_type, method):
    plt.imshow(metrics.confusion_matrix(y, y_pred), cmap=cm.get_cmap('summer'), interpolation='nearest')
    plt.colorbar()
    plt.xlabel('true value')
    plt.ylabel('predicted value')
    plt.title('confusion matrix for %s using %s' % (data_type, method), y=1.05)
    plt.show()

def plot_accuracies(accuracies, proteins, title, directory):
    plt.style.use(['seaborn-white', 'seaborn-paper'])
    matplotlib.rc("font", family="Times New Roman")
    fig, ax = plt.subplots(1)
    x           = arange(1, 5)
    bar_width   = 0.2
    patterns    = ['-', 'x', 'o', '/']
    
    for index in range(len(accuracies)):
        ax.bar(x + bar_width * index, accuracies[index], bar_width, color='grey', edgecolor='black', alpha=0.7, hatch=patterns[index], label=proteins[index])
    
    ax.set_xticks(x + (bar_width*len(proteins)/2))
    ax.set_xticklabels(['RF', 'SVM', 'KNN', 'GNB'], fontsize=20)
    ax.set_title(title, fontsize=24)
    ax.set_ylim(0.0, 1.0)
    ax.set_ylabel('classification accuracy', fontsize=24)
    ax.tick_params(axis='y', labelsize=20)
    ax.legend(bbox_to_anchor=(1.01, 0.5),loc=2,borderaxespad=0.,fontsize=20)
    
    if not os.path.exists(directory):
        os.makedirs(directory)
    filename = "%s.%s" % (title, 'svg')
    savepath = os.path.join(directory, filename)
    print("Saving figure to '%s'..." % savepath)
    fig.set_size_inches(8, 6)

    plt.margins(0.01, 0)
    plt.tight_layout()
    plt.savefig(savepath, dpi=1200, format='svg')
    plt.close('all')
    
def plot_accuracies_bokeh(accuracies, proteins, title, directory='C:\\uday\\gmu\\ngrams\\july_2016_results\\', ext='html'):
    if not os.path.exists(directory):
        os.makedirs(directory)
    filename = "%s.%s" % (title, ext)
    path_plus_filename = os.path.join(directory, filename)
    output_file(path_plus_filename)
    methods = ['RF', 'SVM', 'KNN', 'GNB']
    accuracies_dict = {}
    proteins_and_accuracies = zip(proteins, accuracies)
    TOOLS = "pan,wheel_zoom,box_zoom,reset,save"
    for protein, accuracy in proteins_and_accuracies:
        accuracies_dict[protein] = accuracy
    bar = Bar(SortedDict(accuracies_dict), methods, title=title, stacked=False, legend='top_right', ylabel="accuracy", tools=TOOLS)
    show(bar)

def change_legend_font_to_small():
    leg = plt.gca().get_legend()
    ltext  = leg.get_texts()  # all the text.Text instance in the legend
    plt.setp(ltext, fontsize=20)    # the legend text fontsize

def save_plot(directory, file, ext, dpi):
    if not os.path.exists(directory):
        os.makedirs(directory)
    filename = "%s.%s" % (file, ext)
    savepath = os.path.join(directory, filename)
    print("Saving figure to '%s'..." % savepath)
    figure = plt.gcf() # get current figure
    figure.set_size_inches(8, 6)
    figure.tight_layout()
    figure.savefig(savepath, dpi=dpi, format=ext)
    plt.close('all')

def count_residues_in_sequence(sequence):
        x=ProtParam.ProteinAnalysis(','.join(map(str, sequence)))
        counts=x.count_amino_acids()
        return counts

#calculate n-gram frequencies for xyz/3-letter sequences
def calculate_ngram_frequencies_xyz(n, sequence, threegrams):
    freq_vector = {}
    #initialize all frequencies to zero
    for key in threegrams:
        freq_vector[key]=0
        
    #compute normalized frequencies using standardized min-max normalization
    for index in range(len(sequence)-n+1):
        ngram_string = str(sequence[index:index+n])
        if contains_other_than('XYZ', ngram_string):
            continue
        else:
            freq_vector[ngram_string]+=1

    for key, value in freq_vector.items():
        freq_vector[key]=float(value)/(len(sequence)-n+1)

    return SortedDict(freq_vector)

'''
# calculate ngram frequencies for a given sequence
def calculate_ngram_frequencies(n, sequence):

    new_value = lambda x: 1 if x == 0 else x
    
    one_gram_counts=util.count_residues_in_sequence(sequence)
    two_gram_counts={}
    three_gram_counts={}
    log_freq_vector={}
    
    # initialize all frequencies to zero
    for key in threegrams:
        three_gram_counts[key]=0
        
    for key in twograms:
        two_gram_counts[key]=0
    
    # compute frequency counts
    for index in range(len(sequence)-n+1):
        ngram_string = str(sequence[index:index+n])
        if util.contains_other_than('AILFVYCMWQHSTGRKDEPN', ngram_string):
            continue
        else:
            three_gram_counts[ngram_string] += 1
    
    for index in range(len(sequence)-1):
        ngram_string = str(sequence[index:index+2])
        if util.contains_other_than('AILFVYCMWQHSTGRKDEPN', ngram_string):
            continue
        else:
            two_gram_counts[ngram_string] += 1

    for key,value in three_gram_counts.items():
        f_i=one_gram_counts[key[0]]
        f_j=one_gram_counts[key[1]]
        f_k=one_gram_counts[key[2]]
        
        numerator = (f_i+f_j+f_k)*multiplier+(two_gram_counts[key[0:2]]+two_gram_counts[key[1:3]])*(multiplier^2)+value*(multiplier^3)
        denominator=new_value(f_i)*new_value(f_j)*new_value(f_k)
        
        if (numerator==0):
            log_freq_vector[key]=10000
        else:
            log_freq_vector[key]=log(numerator/denominator)
                
    return SortedDict(log_freq_vector)
'''
    
if __name__ == '__main__':
    start_roc_plot('test')
    plt.close()
    input_file = argv[1]
    start = argv[2]
    end=int(start)+10
    get_sub_sequences(input_file, int(start), end)
    
    list1=[0.5,0.4,0.7,0.76]
    list2=[0.6,0.9,0.76,0.83]
    list3=[0.96,0.84,0.71,0.585]
    mylist=[]
    mylist.append(list1)
    mylist.append(list2)
    mylist.append(list3)
    proteins=['HA', 'NA', 'NP']
    directory='C:\\uday\\gmu\\ngrams\\july_2016_results'
    plot_accuracies(mylist, proteins, 'test11', directory)