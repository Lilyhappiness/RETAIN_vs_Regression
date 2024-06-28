#!/usr/bin/python

"""
Kathryn Egan and Li Li
Created on 2019-12

Runs logistic regression on the claims data in the folder given by [inputpath] and [subfolders] (optional) and
writes results to [resultspath]. Will create [resultspath] if it does not exist. If [resultspath] does exist,
will prompt user before deleting it.

[labelthru] parameter gives the maximum number of days in the last [labelthru] days that will have their
time-varying variables (tvvs) labeled with their index relative to the last claims day. regression.py will then
iterate on x-1 through x=0 (default, no labels).

E.g. if labelthru=2, will label tvvs on last claims day with "1_" and tvvs in second to last claims day
with "2_" and run regression on these sequences. Then will label tvvs on last claims day with "1_" and
run regression on these sequences. Then will label NO tvvs with day index and run regression.

Assumes the following file naming format in path/[subfolder]:

claims_visits_ed_1_data.pkl     # default train data
claims_visits_ed_1_target.pkl   # default train labels
claims_visits_ed_2_data.pkl     # default dev data
claims_visits_ed_2_target.pkl   # default dev labels
claims_visits_ed_3_data.pkl     # default test data
claims_visits_ed_3_target.pkl   # default test labels
claims_visits_ip_1_data.pkl     # default train data
claims_visits_ip_1_target.pkl   # default train labels
claims_visits_ip_2_data.pkl     # default dev data
claims_visits_ip_2_target.pkl   # default dev labels
claims_visits_ip_3_data.pkl     # default test data
claims_visits_ip_3_target.pkl   # default test labels
dictionary.pkl                  # index: feature string mapping

By default, regression.py will train on datasets labeled with "1" and test on datasets labeled with "3". This may
be overridden by the user with [train] and [test] parameters.

Usage:
inputpath and resultspath are required
-- indicates an optional parameter
--help in the command line for help
python regression.py [inputpath] [resultspath] [--subfolders SUBFOLDER1 .. SUBFOLDERn] [--events ed ip] [--train 1 2 3] [--test 1 2 3] [--labelthru INT>=0]

Example 1:

    file structure:
    C:/Users/lli/Downloads/1Match/
        staticonly/
            claims_visits_ed_1_data.pkl ...
        excludelastday/
            claims_visits_ed_1_data.pkl ...
        chemoradonlylastday/
            claims_visits_ed_1_data.pkl ...

    python regression.py "C:/Users/lli/Downloads/1Match" "C:/Users/lli/Downloads/my_results_1" --subfolders excludelastday chemoradonlylastday

Examples 2 and 3:

    file structure:
    C:/Users/lli/Downloads/1Match/
        claims_visits_ed_1_data.pkl ...

    python regression.py "C:/Users/lli/Downloads/1Match" "C:/Users/lli/Downloads/my_results_2" --events ed

    python regression.py "C:/Users/lli/Downloads/1Match" "C:/Users/lli/Downloads/my_results_3" --labelthru 3 --train 1 --test 1
"""

import pickle
import os
import sys
import argparse
import shutil
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import auc
from sklearn.linear_model import LogisticRegression
from scipy.sparse import dok_matrix
import utils
import matplotlib.pyplot as plt


def process_features(data, targets, codedict, daylabel, feature_map):
    """ Maps data and given feature processing specs to a sparse matrix for regression.
    Args:
        data (DataFrame) : unpickled RETAIN data
        targets (DataFrame) : unpickled RETAIN targets
        codedict (dict int:str) : feature index mapped to string rep
        daylabel (int) : extent of days from last day to label
        feature_map (dict str:int) :
            string rep of feature mapped to index for LR's sparse matrix, empty if data is for training
    Returns:
        X (dok matrix) : num instances x num features binary sparse matrix, empties dropped
        y (DataFrame) : patient ids and instance labels, empties dropped
        feature_map (dict str:int) : string rep of feature mapped to index for LR's sparse matrix
    """
    increment = 1000  # amount by which to increase sparse matrix's size in chunks
    n = increment
    m = len(feature_map) if feature_map else increment
    y = targets.sort_index()
    X = dok_matrix((n, m), dtype=int)
    map_features = not feature_map
    feature_index = 0
    X_index = 0
    df_index = 0
    empties = set()
    pids = []

    while df_index < len(data):
        curr_features = set()
        if X_index >= n:
            n += increment
            X.resize((n, m))
        days = data['codes'][df_index]
        pids.append(data['PID'][df_index])

        for day_index in range(len(days)):
            for code in days[day_index]:
                feature = codedict[code]
                prefix = get_prefix(len(days) - 1, day_index, feature, daylabel)
                labeled_feature = prefix + feature
                if labeled_feature in curr_features:
                    continue
                # update feature map with index if a new feature is found in train data
                if map_features and labeled_feature not in feature_map:
                    feature_map[labeled_feature] = feature_index
                    feature_index += 1
                    if feature_index >= m:
                        m += increment
                        X.resize((n, m))
                # skip features in any test set that do not appear in train
                try:
                    X[X_index, feature_map[labeled_feature]] = 1
                except KeyError:
                    continue
                curr_features.add(labeled_feature)
        if not curr_features:
            empties.add(df_index)
        else:
            X_index += 1
        df_index += 1

    # trim excess rows, columns
    X.resize(X_index, len(feature_map))
    y['PID'] = pids
    y = y.drop(empties).reset_index()
    return X, y, feature_map


def get_prefix(last_i, curr_i, feature, daylabel):
    """ Returns the prefix for this feature given the feature and its position in
    day sequence relative to last claims day.
    Args:
        last_i (int) : final index in subsequence
        curr_i (int) : current index of day in subsequence
        feature (str) : feature as string
        daylabel (int) : label for timevarying features
    Returns:
        str : prefix if applicable else empty string
    """
    if not utils.is_timevarying(feature):
        return ''
    prefix = '{}_'.format(last_i - curr_i + 1) if last_i - curr_i < daylabel else ''
    return prefix


def format_coefficients(model, feature_map):
    """ Convert coefficients in given model to string, one coefficient weight
    and feature per line sorted by weight.
    Args:
        model (LogisticRegression) : logistic regression model
        feature_map (dic str:int) : string feature mapped to model's index
    Returns:
        coef (str) : each feature and its weight per line, sorted by weight
    """
    coef = []
    for feature in feature_map:
        c = model.coef_[0][feature_map[feature]]
        coef.append((c, feature))
    coef.sort()
    coef = '\n'.join(['{:.6f}\t{}'.format(c, feature) for c, feature in coef])

    return coef


def get_events(events):
    """ Returns args-given events, if any. Raises ValueError if events are not one or more of ed, ip.
    Args:
        events (list of str) : arguments given by user
    Returns:
        events (list of str) : ed and/or ip
    """
    events = [e.lower() for e in events] if events else ['ed', 'ip']
    diff = set(events).difference({'ed', 'ip'})
    if diff:
        raise ValueError('ERROR: Unrecognized parameter(s): {}'.format(diff))
    return events


def get_subfolders(toppath, subfolders):
    """ Returns args-given subfolders, if any. Raises ValueError if path or
    subfolders are not found
    Args:
        path (str) : path to folders given by user
        subfolders (list of str) : arguments given by user
    Returns:
        subfolders (list of str) : subfolders
    """
    subfolders = [s.strip('/').strip('\\') for s in subfolders] if subfolders else ['']
    paths = []
    for folder in subfolders:
        path = os.path.join(toppath, folder) if folder else toppath
        path = os.path.normpath(path)
        if not os.path.exists(path):
            raise ValueError('ERROR: {} not found'.format(path))
        paths.append(path)

    return paths, subfolders


def get_labelthru(labelthru):
    """ Returns args-given labelthru, if any. Raises ValueError if labelthru is not an int >= 0.
    Args:
        labelthru (list of str) : argument given by user
    Returns:
        labelthru (int) : user setting cast as int
    """
    labelthru = labelthru[0] if labelthru else 0
    try:
        labelthru = int(labelthru)
    except ValueError:
        raise ValueError('ERROR: labelthru must be an integer')
    if labelthru < 0:
        raise ValueError('ERROR: labelthru must by >=0')
    return labelthru


def get_train_test(datatype, default):
    """ Returns args-given datatype or default if none. Raises ValueError if datatype not 1, 2, or 3.
    Args:
        datatype (list of str) : argument given by user
        default (str) : default if no user setting
    Returns:
        datatype (str) : 1, 2, or 3
    """
    datatype = datatype[0] if datatype else default
    if datatype not in ('1', '2', '3'):
        raise ValueError('ERROR: Unrecognized train or test parameter: {}'.format(datatype))
    return datatype


def format_graph(folder, x, y, no_skill_x, no_skill_y, x_label, y_label, filename):
    """ Formats and saves a graph to the given file.
    Args:
        folder (str) : folder to save graph to
        x (list) : x axis values
        y (list) : y axis values
        no_skill_x (list) : x axis values for no skill
        no_skill_y (list) : y axis values for no skill
        x_label (str) : label for x axis
        y_label (str) : label for y axis
        filename (str) : name of resulting file, also graph title
    """
    plt.plot(no_skill_x, no_skill_y, linestyle='--', label='No Skill')
    plt.plot(x, y, marker='.', label='Logistic')
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(filename)
    plt.legend()
    plt.savefig(os.path.join(folder, filename))
    plt.close()


def format_matrix(actuals, predictions):
    """ Formats predictions and actual labels as confusion matrix (string).
    Args:
        actuals (list int) : list of actual labels
        predictions (list int) : list of predicted labels
    Returns:
        matrix (str) : actual x predicted labels in confusion matrix as string
    """
    counts = {}
    for prediction, actual in zip(predictions, actuals):
        counts.setdefault(actual, {}).setdefault(prediction, 0)
        counts[actual][prediction] += 1
    string = 'Rows=Actual; Columns=Predicted\n'
    string += '\t0\t1\n'
    for actual in 0, 1:
        string += str(actual)
        for prediction in 0, 1:
            try:
                count = counts[actual][prediction]
            except KeyError:
                count = 0
            string += '\t' + str(count)
        string += '\n'
    return string


def get_no_skill(targets):
    """ Returns array of no skill predicted labels based on the most frequent label.
    Args:
        targets (Series) : array of actual labels
    Returns:
        no_skill (list) : array of no skill predicted labels
    """
    targets = list(targets)
    counts = {t: targets.count(t) for t in targets}
    no_skill_label = max(counts.items(), key=lambda item: item[1])[0]
    no_skill = [no_skill_label for _ in range(len(targets))]
    return no_skill


def make_path(path, folder):
    """ Makes the given path + folder if it does not exist. Returns joined path + folder.
    Args:
        path (str) : top-level path
        folder (str) : folder
    Returns:
        new_path (str) : path + folder
    """
    new_path = os.path.join(path, folder)
    if not os.path.exists(new_path):
        os.mkdir(new_path)
    return new_path


def format_probs(probs, pids):
    """ Returns the probability of each label for each patient ID as a string.
    Args:
        probs (DataFrame) : probability of 0 label, 1 label
        pids (Series) : patient IDs
    Returns:
        output (str) : patient IDs and probs of each label as string
    """
    output = ['PID\t0 label\t1 label']
    output += ['{}\t{}\t{}'.format(pid, prob_0, prob_1) for pid, (prob_0, prob_1) in zip(pids, probs)]
    output = '\n'.join(output)
    return output


def get_results(
    inputpath, subfolder, resultspath, train_X_df, train_y, test_X_df, test_y, event, daylabel):
    """ Gets results for test_X and test_y given train_X and train_y. Writes results to folder.
    Args:
        inputpath (str) : path to the input data
        subfolder (str) : the subfolder name "n/a" if none
        resultspath (str) : path to results folder
        train_X (DataFrame) : training data
        train_y (DataFrame) : training labels
        test_X (DataFrame) : test data
        test_y (DataFrame) : test labels
        event (str) : ed or ip
        daylabel (int) : extent of days to label with claims day index from last
    Returns:
        roc (float) : ROC AUC
        prc (float) : PRC AUC
    """
    subfolder = subfolder + '_' if subfolder else ''
    with open(os.path.join(inputpath, 'dictionary.pkl'), 'rb') as f:
        codedict = pickle.load(f)

    train_X, train_y, feature_map = process_features(train_X_df, train_y, codedict, daylabel, feature_map={})
    test_X, test_y, _ = process_features(test_X_df, test_y, codedict, daylabel, feature_map=feature_map)
    model = LogisticRegression(solver='newton-cg')
    model.fit(train_X, train_y.target)
    predicted_probs = model.predict_proba(test_X)
    predicted_labels = model.predict(test_X)
    actual = test_y.target

    coef = format_coefficients(model, feature_map)
    coef_folder = make_path(resultspath, 'coefficients')
    coef_file = '{}coefs_{}_labelthru={}.txt'.format(subfolder, event, daylabel)
    with open(os.path.join(coef_folder, coef_file), 'w') as f:
        f.write(coef)

    probs = format_probs(predicted_probs, test_y.PID)
    prob_folder = make_path(resultspath, 'label_probabilities')
    prob_file = '{}probs_{}_labelthru={}.txt'.format(subfolder, event, daylabel)
    with open(os.path.join(prob_folder, prob_file), 'w') as f:
        f.write(probs)

    matrix = format_matrix(actual, predicted_labels)
    matrix_folder = make_path(resultspath, 'matrices')
    matrix_file = '{}confusion_matrix_{}_labelthru={}.txt'.format(subfolder, event, daylabel)
    with open(os.path.join(matrix_folder, matrix_file), 'w') as f:
        f.write(matrix)

    # get receiver operating characteristic curve
    graph_folder = make_path(resultspath, 'graphs')
    no_skill = get_no_skill(actual)
    roc = roc_auc_score(actual, predicted_probs[:, 1])
    fpr, tpr, _ = roc_curve(actual, predicted_probs[:, 1])
    no_skill_fpr, no_skill_tpr, _ = roc_curve(actual, no_skill)
    ROC_file = '{}ROC_{}_labelthru={}.png'.format(subfolder, event, daylabel)
    format_graph(
        graph_folder, fpr, tpr, no_skill_fpr, no_skill_tpr, 'FPR', 'TPR', ROC_file)

    # get precision-recall curve
    no_skill_score = sum([1 for i in range(len(no_skill)) if no_skill[i] == actual[i]])/len(actual)
    precision, recall, _ = precision_recall_curve(actual, predicted_probs[:, 1])
    prc = auc(recall, precision)
    PRC_file = '{}PRC_{}_labelthru={}.png'.format(subfolder, event, daylabel)
    format_graph(
        graph_folder, recall, precision, [0, 1], [no_skill_score, no_skill_score], 'Recall', 'Precision', PRC_file)

    return roc, prc


def main():
    """ Collect and process arguments, run regression per arguments and write results to file. """
    parser = argparse.ArgumentParser()
    parser.add_argument('inputpath', help='absolute path to data folder (e.g. C:/Users/kegan/Downloads/fixeddata/1Match)')
    parser.add_argument('resultspath', help='path to results folder (e.g. C:/Users/kegan/Downloads/my_results)')
    parser.add_argument('--subfolders', nargs='+', help='the folders, if any, within path to process (e.g. dropimaginglastday)')
    parser.add_argument('--events', nargs='+', help='ED and/or IP (default ED and IP)')
    parser.add_argument('--train', nargs=1, help='train on set 1 (train), 2 (dev), or 3 (test), (default 1)')
    parser.add_argument('--test', nargs=1, help='test on set 1 (train), 2 (dev), or 3 (test), (default 3)')
    parser.add_argument('--labelthru',  nargs=1, help=
        '(int >= 0) label all days in sequence by day index (i) thru x starting from last (1=last), ' +\
        'run for each i in 0-x (default 0=no day labels)')
    args = parser.parse_args()

    # process and validate all the args
    events = get_events(args.events)
    inputpaths, subfolders = get_subfolders(args.inputpath, args.subfolders)
    labelthru = get_labelthru(args.labelthru)
    train = get_train_test(args.train, '1')
    test = get_train_test(args.test, '3')

    if os.path.exists(args.resultspath):
        while True:
            answer = input('Delete {}? Y/N\n>>'.format(args.resultspath))
            if answer.strip().upper() in ('Y', 'YES'):
                sys.stderr.write('Deleting files in {}...\n'.format(args.resultspath))
                shutil.rmtree(args.resultspath)
                break
            else:
                sys.stderr.write('Please choose a different results folder and try again.\nExiting...\n')
                sys.exit()
    os.mkdir(args.resultspath)
    sys.stderr.write('inputpath={}\n'.format(args.inputpath))
    sys.stderr.write('resultspath={}\n'.format(args.resultspath))
    train_test = 'train={}\ntest={}\n'.format(train, test)
    sys.stderr.write('{}\n'.format(train_test))
    with open(os.path.join(args.resultspath, 'info.txt'), 'w') as f:
        f.write(train_test)

    with open(os.path.join(args.resultspath, 'results_summary.txt'), 'w') as f:
        f.write('dataset\tevent\tday\tROC AUC\tPRC AUC\n')

        for inputpath, subfolder in zip(inputpaths, subfolders):
            if subfolder:
                sys.stderr.write('subfolder={}\n'.format(subfolder))

            for event in sorted(list(events)):
                train_prefix = 'claims_visits_{}_{}'.format(event, train)
                test_prefix = 'claims_visits_{}_{}'.format(event, test)
                train_X = pd.read_pickle(os.path.join(inputpath, train_prefix + '_data.pkl'))
                train_y = pd.read_pickle(os.path.join(inputpath, train_prefix + '_target.pkl'))
                test_X = pd.read_pickle(os.path.join(inputpath, test_prefix + '_data.pkl'))
                test_y = pd.read_pickle(os.path.join(inputpath, test_prefix + '_target.pkl'))

                for daylabel in range(labelthru + 1):
                    roc, prc = get_results(inputpath, subfolder, args.resultspath, train_X, train_y, test_X, test_y, event, daylabel)
                    sys.stderr.write('Event={}, Day={}...\tROC AUC={:.4f}, PRC AUC={:.4f}\n'.format(event, daylabel, roc, prc))
                    line = '\t'.join([subfolder, event, str(daylabel), '{:.4f}'.format(roc), '{:.4f}'.format(prc)])
                    f.write(line + '\n')

            sys.stderr.write('\n')


if __name__=='__main__':
    main()
