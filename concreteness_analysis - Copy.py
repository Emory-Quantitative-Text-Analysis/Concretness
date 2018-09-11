"""
Author: Doris Zhou; modified by Catherine Xiao
Date: February 19, 2018
Performs concreteness analysis on a text file using Brysbaert et al. concreteness ratings.
Parameters:
    --dir [path of directory]
        specifies directory of files to analyze
    --file [path of text file]
        specifies location of specific file to analyze
    --out [path of directory]
        specifies directory to create output files
    --mode [mode]
        takes either "median" or "mean"; determines which is used to calculate sentence concreteness values
"""
# add parameter to exclude duplicates? also mean or median analysis

import csv
import sys
import os
import statistics
import time
import argparse
from stanfordcorenlp import StanfordCoreNLP
import pandas as pd

#nlp = StanfordCoreNLP('C:/Users/Doris/software tools/stanford-corenlp-full-2016-10-31')
nlp = StanfordCoreNLP("C:\Program Files (x86)\PC-ACE\NLP\stanford-corenlp-full-2017-06-09")

from nltk import tokenize
from nltk.corpus import stopwords

stops = set(stopwords.words("english"))
#ratings = "../lib/Concreteness_ratings_Brysbaert_et_al_BRM.csv"
ratings = "C:\Program Files (x86)\PC-ACE\NLP\Miscellaneous\Concreteness_ratings_Brysbaert_et_al_BRM.csv"
data = pd.read_csv(ratings)
data_dict = {col: list(data[col]) for col in data.columns}
#print data_dict
# performs concreteness analysis on inputFile using the Brysbaert et al. concreteness ratings, outputting results to a new CSV file in outputDir
def analyzefile(input_file, output_dir, mode):
    """
    Performs concreteness analysis on the text file given as input using the Brysbaert et al. concreteness ratings.
    Outputs results to a new CSV file in output_dir.
    :param input_file: path of .txt file to analyze
    :param output_dir: path of directory to create new output file
    :param mode: determines how concreteness values for a sentence are computed (median or mean)
    :return:
    """
    output_file = os.path.join(output_dir, "Output Concreteness Ratings " + os.path.basename(input_file).rstrip('.txt') + ".csv")

    # read file into string
    with open(input_file, 'r') as myfile:
        fulltext = myfile.read()
    # end method if file is empty
    if len(fulltext) < 1:
        print('Empty file.')
        return

    from nltk.stem.wordnet import WordNetLemmatizer
    lmtzr = WordNetLemmatizer()

    # otherwise, split into sentences
    sentences = tokenize.sent_tokenize(fulltext)
    i = 1 # to store sentence index
    # check each word in sentence for concreteness and write to output_file
    with open(output_file, 'w') as csvfile:
        fieldnames = ['Sentence ID', 'Sentence','Conc.MEDIAN','Conc.MEAN', 'Conc.SD',
                      '# Words Found','Percentage', 'Found Words', 'All Words']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames, lineterminator='\n')
        writer.writeheader()

        # analyze each sentence for concreteness
        for s in sentences:
            # print("S" + str(i) +": " + s)
            all_words = []
            found_words = []
            total_words = 0
            score_list = [] #use the Conc.M as scores to calculate the concretness
            #cmedian_list = []
            #cmean_list =[]
            #csd_list = []

            # search for each valid word's concreteness ratings
            words = nlp.pos_tag(s.lower())
            for index, p in enumerate(words):
                # don't process stops or words w/ punctuation
                w = p[0]
                pos = p[1]
                if w in stops or not w.isalpha():
                    continue

                # lemmatize word based on pos
                if pos[0] == 'N' or pos[0] == 'V':
                    lemma = lmtzr.lemmatize(w, pos=pos[0].lower())
                else:
                    lemma = w

                all_words.append(str(lemma))

                if lemma in data_dict['Word']:
                    index = data_dict['Word'].index(lemma)
                    score = float(data_dict['Conc.M'][index])
                    found_words.append((str(lemma),score))
                    score_list.append(score)
                # search for lemmatized word in Brysbaert et al. concreteness ratings
                """
                with open(ratings,'rU') as csvfile:
                    reader = csv.DictReader(csvfile)
                    for row in reader:
                        if row['Word'].lower() == lemma.lower():
                            
                            score = float(row['Conc.M'])
                            found_words.append((lemma,score))
                            score_list.append(score)
                            #cmedian = float(row['Conc.M'])
                            #csd = float(row['Conc.SD'])
                            #cmean = float(row['Conc.MEAN'])

                            #cmdian_list.append(cmedian)
                            #csd_list.append(csd)
                            #cmean_list.append(cmean)
                """
            if len(found_words) == 0:  # no words found in Brysbaert et al. concreteness ratings for this sentence
                writer.writerow({'Sentence ID': i,
                                 'Sentence': s,
                                 'CONC.MEDIAN':'N/A',
                                 'Conc.MEAN': 'N/A',
                                 'Conc.SD': 'N/A',
                                 '# Words Found': 0,
                                 'Percentage': '0.0%',
                                 'Found Words': 'N/A',
                                 'All Words': all_words,
                                 
                                 })
                i += 1
            else:  # output concreteness info for this sentence

                # get values
                """
                if mode == 'mean':
                    conc_median = statistics.median(score_list)
                    conc_mean = statistics.mean(score_list)
                    conc_sd = 'N/A'
                    if len(score_list) > 1:
                        conc_sd = statistics.stdev(score_list)
                    #print(conc_m,conc_sd)
                    writer.writerow({'Sentence ID': i,
                                     'Sentence': s,
                                     'Conc.MEDIAN':conc_median,
                                     'Conc.MEAN': conc_mean,
                                     'Conc.SD': conc_sd,
                                     '# Words Found': ("%d out of %d" % (len(found_words), len(all_words))),
                                     'Found Words': found_words,
                                     'All Words': all_words,
                                     'Percentage': str(100*(float(len(found_words))/float(len(all_words))))+'%'

                                     })
                    
                else:
                    #concreteness = statistics.mean(cm_list)
                    #arousal = statistics.mean(csd_list)
                    concreteness = statistics.mean(cm_list)
                    arousal = statistics.mean(csd_list)

                    writer.writerow({'Sentence ID': i,
                                     'Sentence': s,
                                     'Conc.M': concreteness,
                                     'Conc.SD': arousal,
                                     '# Words Found': ("%d out of %d" % (len(found_words), len(all_words))),
                                     'Found Words': found_words,
                                     'All Words': all_words
                                     })
                """
                conc_median = statistics.median(score_list)
                conc_mean = statistics.mean(score_list)
                conc_sd = 'N/A'
                if len(score_list) > 1:
                    conc_sd = statistics.stdev(score_list)
                #print(conc_m,conc_sd)
                writer.writerow({'Sentence ID': i,
                                 'Sentence': s,
                                 'Conc.MEDIAN':conc_median,
                                 'Conc.MEAN': conc_mean,
                                 'Conc.SD': conc_sd,
                                 '# Words Found': "%d out of %d" % (len(found_words), len(all_words)),
                                 'Percentage': str(100*(float(len(found_words))/float(len(all_words))))+'%',
                                 'Found Words': found_words,
                                 'All Words': all_words
                                 
                                 })
                i += 1


def main(input_file, input_dir, output_dir, mode):
    """
    Runs analyzefile on the appropriate files, provided that the input paths are valid.
    :param input_file:
    :param input_dir:
    :param output_dir:
    :param mode:
    :return:
    """

    if len(output_dir) < 0 or not os.path.exists(output_dir):  # empty output
        print('No output directory specified, or path does not exist')
        sys.exit(0)
    elif len(input_file) == 0 and len(input_dir)  == 0:  # empty input
        print('No input specified. Please give either a single file or a directory of files to analyze.')
        sys.exit(1)
    elif len(input_file) > 0:  # handle single file
        if os.path.exists(input_file):
            analyzefile(input_file, output_dir, mode)
        else:
            print('Input file "' + input_file + '" is invalid.')
            sys.exit(0)
    elif len(input_dir) > 0:  # handle directory
        if os.path.isdir(input_dir):
            directory = os.fsencode(input_dir)
            for file in os.listdir(directory):
                filename = os.path.join(input_dir, os.fsdecode(file))
                if filename.endswith(".txt"):
                    start_time = time.time()
                    print("Starting analysis of " + filename + "...")
                    analyzefile(filename, output_dir, mode)
                    print("Finished analyzing " + filename + " in " + str((time.time() - start_time)) + " seconds")
        else:
            print('Input directory "' + input_dir + '" is invalid.')
            sys.exit(0)


if __name__ == '__main__':
    # get arguments from command line
    parser = argparse.ArgumentParser(description='Concreteness analysis with Concreteness ratings by Brysbaert et al.')
    parser.add_argument('--file', type=str, dest='input_file', default='',
                        help='a string to hold the path of one file to process')
    parser.add_argument('--dir', type=str, dest='input_dir', default='',
                        help='a string to hold the path of a directory of files to process')
    parser.add_argument('--out', type=str, dest='output_dir', default='',
                        help='a string to hold the path of the output directory')
    parser.add_argument('--mode', type=str, dest='mode', default='mean',
                        help='mode with which to calculate concreteness in the sentence: mean or median')
    args = parser.parse_args()

    # run main
    sys.exit(main(args.input_file, args.input_dir, args.output_dir, args.mode))

    #example: a single file
    #python concretness_analysis.py --file "C:\Users\rfranzo\Documents\ACCESS Databases\PC-ACE\NEW\DATA\CORPUS DATA\MURPHY\Murphy Miracles thicker than fog CORENLP.txt" --out C:\Users\rfranzo\Desktop\NLP_output