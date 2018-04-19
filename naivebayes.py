from __future__ import division
import sys
import os.path
import numpy as np

import util

def get_counts(file_list):
    '''
    Computes counts for each word that occurs in the files in file_list.
    :param file_list: a list of filenames
    :return: A dictionary whose keys are words, and whose values are the number of files the
    key occurred in.
    '''

    dict_of_words = util.Counter()
    for file in file_list:
        words_in_file = util.get_words_in_file(file)

        #for each unique word in a file, increment the count
        for word in set(words_in_file):
            dict_of_words[word] += 1

    return dict_of_words


def get_log_probabilities(file_list):
    '''
    Computes log-frequencies for each word that occurs in the files in
    file_list.
    :param file_list: a list of filenames
    :return: A dictionary whose keys are words, and whose values are the log of the smoothed
    estimate of the fraction of files the key occurred in
    '''

    num_of_files = len(file_list)
    num_of_categories = 2

    #find the number of files that a word appears in
    word_counts = get_counts(file_list)

    #a word that doesn't exist in the dictionary will have an initial value given in the lambda function
    dict_of_word_frequencies = util.DefaultDict(lambda: -np.log(num_of_files + num_of_categories))

    #find the log-frequencies
    for word in word_counts:
        dict_of_word_frequencies[word] = np.log(word_counts[word] + 1) - np.log(num_of_files + num_of_categories)
        assert dict_of_word_frequencies[word] < 0

    return dict_of_word_frequencies


def learn_distributions(file_lists_by_category):
    '''

    :param file_lists_by_category: A two-element list. The first element is a list of spam files,
    and the second element is a list of ham (non-spam) files.
    :return:
    (log_probabilities_by_category, log_prior)

    log_probabilities_by_category : A list whose first element is a smoothed
                                    estimate for log P(y=w_j|c=spam), and
                                    whose second element is the same for c=ham.

    log_prior_by_category : A list of estimates for the log-probabilities for
                            each class:
                            [est. for log P(c=spam), est. for log P(c=ham)]
    '''

    num_of_spam = len(file_lists_by_category[0])
    num_of_ham = len(file_lists_by_category[1])
    num_of_files = num_of_ham + num_of_spam

    log_priors_by_category = [np.log(num_of_spam/num_of_files), np.log(num_of_ham/num_of_files)]

    log_probabilities_by_category = [get_log_probabilities(file_lists_by_category[0]), get_log_probabilities(file_lists_by_category[1])]

    return (log_probabilities_by_category, log_priors_by_category)



def classify_message(message_filename, log_probabilities_by_category, log_prior_by_category, names = ['spam', 'ham']):
    '''
    classify the message in the given file using learned parameters

    :param message_filename: name of the file containing the message to be classified
    :param log_probabilities_by_category:
    :param log_prior_by_category:
    :param names: class labels
    :return: the predicted class
    '''

    try:
        words_in_file = set(util.get_words_in_file(message_filename))
    except:
        return "file cannot be decoded"
    num_of_categories = len(log_priors_by_category)


    # make list of all the words seen in training
    all_words_from_training = []
    for i in range(num_of_categories):
        all_words_from_training += log_probabilities_by_category[i].keys()
        all_words_from_training = list(set(all_words_from_training))

    log_likelihoods = []
    for i in range(num_of_categories):
        total = 0
        all_word_log_probs = log_probabilities_by_category[i]

        for word in all_words_from_training:
            log_prob = all_word_log_probs[word]
            is_in_file = (word in words_in_file)
            total += is_in_file * log_prob + (1 - is_in_file) * np.log(1 - np.exp(log_prob))
        log_likelihoods.append(total)

    posterior = np.array(log_likelihoods) + np.array(log_prior_by_category)
    predicted_category = np.argmax(posterior)

    return names[predicted_category]


if __name__ == '__main__':

    testing_folder = "../email_classification/data/testing"
    spam_folder = "../email_classification/data/spam"
    ham_folder = "../email_classification/data/ham"

    print("Training the model...")
    ## Train the model
    file_lists = []
    for folder in (spam_folder, ham_folder):
        file_lists.append(util.get_files_in_folder(folder))
    (log_probabilities_by_category, log_priors_by_category) = \
            learn_distributions(file_lists)

    # columns and rows are indexed by 0 = 'spam' and 1 = 'ham'
    # rows correspond to true label, columns correspond to guessed label
    performance_measures = np.zeros([2,2])

    print("Testing the model....")
    ### Classify and measure performance
    for filename in (util.get_files_in_folder(testing_folder)):
        ## Classify
        label = classify_message(filename,
                                 log_probabilities_by_category,
                                 log_priors_by_category,
                                 ['spam', 'ham'])
        ## Measure performance
        # Use the filename to determine the true label
        if label == "file cannot be decoded":
            continue
        base = os.path.basename(filename)
        true_index = 1 if 'ham' in base else 0
        guessed_index = 1 if label == 'ham' else 0
        performance_measures[true_index, guessed_index] += 1

    output_message = "The model has correctly classified %d out of %d spam messages, and %d out of %d ham messages."
    # Correct counts are on the diagonal
    correct = np.diag(performance_measures)
    # totals are obtained by summing across guessed labels
    totals = np.sum(performance_measures, 1)
    print(output_message % (correct[0],
                      totals[0],
                      correct[1],
                      totals[1]))

