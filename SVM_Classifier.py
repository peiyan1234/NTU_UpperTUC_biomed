import argparse
import os
import json
import csv

import random
import numpy as np

from sklearn.svm import SVC, NuSVC
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import StratifiedKFold, LeavePOut, RepeatedStratifiedKFold
from sklearn.metrics import accuracy_score, average_precision_score
from sklearn.metrics import plot_confusion_matrix
from sklearn.metrics import roc_curve, det_curve, auc, plot_roc_curve, plot_det_curve
from sklearn.metrics import roc_auc_score
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RandomizedSearchCV

import matplotlib.pyplot as plt

_pwd_ = os.getcwd()


def splitter(Feature_Table, 
             output_folder, 
             train_index, 
             test_index):
    
    Train_Table = {}
    Test_Table = {}
    
    subject_list = list(Feature_Table.keys())
    
    for _index_ in train_index:
    
        Train_Table[subject_list[_index_]] = Feature_Table[subject_list[_index_]]
    
    for _index_ in test_index:
    
        Test_Table[subject_list[_index_]] = Feature_Table[subject_list[_index_]]

    Train_json_path = os.path.join(output_folder, 
                                   'Train_data.txt')

    Train_json_file = open(Train_json_path, 'w')

    Train_json_content = json.dumps(Train_Table, 
                                    indent = 4)

    Train_json_file.writelines(Train_json_content)
    
    Train_json_file.close()
    
    Test_json_path = os.path.join(output_folder, 
                                  'Test_data.txt')

    Test_json_file = open(Test_json_path, 'w')

    Test_json_content = json.dumps(Test_Table, 
                                   indent = 4)

    Test_json_file.writelines(Test_json_content)
    
    Test_json_file.close()

    return Train_json_path, Test_json_path


def read_SelectionResult(csv_path):

    SelectionResult = {}

    N_features = 0

    feature_csv = open(csv_path, 'r')
    csv_reader = csv.reader(feature_csv, delimiter = ',')
    count = 0

    for row in csv_reader:

        if count == 0:

            count += 1

        else:

            _key_ = row[0]

            features = eval(row[1])
            SelectionResult[_key_] = features

            N_features += len(features)

    return SelectionResult, N_features


def get_Data(SelectionResult, 
             N_features, 
             json_path, 
             LabelName):

    Train_table = json.load(open(json_path, 'r'))
    
    X_train = []
    y_train = []
    
    subj_list = list(Train_table.keys())
    filter_list = list(Train_table[subj_list[0]]['Features'].keys())
    count = 0
    
    for _subj_ in subj_list:
    
        label = Train_table[_subj_][LabelName]
    
        if (label == 'HG'):
    
            y_train.append(1)
            count += 1
    
        elif (label == 'LG'):
    
            y_train.append(0)
            count += 1
    
        else:
    
            continue
    
        feature_values = []
    
        for _filter_ in filter_list:
    
            radiomics_list = list(Train_table[_subj_]['Features'][_filter_].keys())
    
            for _radiomics_ in radiomics_list:
    
                if _radiomics_ in list(SelectionResult.keys()):
    
                    if _filter_ in SelectionResult[_radiomics_]:
    
                        feature_values.append(Train_table[_subj_]['Features'][_filter_][_radiomics_])

        X_train.append(feature_values)

    select_map = {}
    col_index = 0
    
    for _filter_ in filter_list:
    
        radiomics_list = list(Train_table[subj_list[0]]['Features'][_filter_].keys())
    
        for _radiomics_ in radiomics_list:
    
            if _radiomics_ in list(SelectionResult.keys()):
    
                if _filter_ in SelectionResult[_radiomics_]:
    
                    select_map[col_index] = [_filter_, _radiomics_]
                    col_index += 1

    return len(y_train), np.array(X_train, dtype=np.double), np.array(y_train, dtype=np.double), select_map


def backward_selection(SelectionResult, 
                       N_features, 
                       Train_json_path, 
                       Test_json_path, 
                       Label, 
                       Feature_ratio, 
                       K, 
                       select_feature_counter):

    N_samples, X_train, y_train, select_map = get_Data(SelectionResult, 
                                                       N_features, 
                                                       Train_json_path, 
                                                       Label)

    #rng = random.randint(0, N_samples)
    
    C = 8
    tol = 1e-3
    nu = 0.15
    
    #clf = SVC(gamma='scale', class_weight='balanced', C=C, tol=tol)
    clf = NuSVC(class_weight='balanced', 
                tol=tol, 
                nu=nu)
    
    #clf = DecisionTreeClassifier(class_weight='balanced', random_state=0)
    #clf = make_pipeline(StandardScaler(),
    #                    SGDClassifier( 
    #                                class_weight='balanced',
    #                                n_jobs = -1,
    #                                random_state = 0,
    #                                validation_fraction=0.25))
    
    #ss = KFold(n_splits = int(args.Kfold))
    #skf = StratifiedKFold(n_splits=K)
    skf = RepeatedStratifiedKFold(n_splits=K, 
                                  n_repeats=30, 
                                  random_state=0)

    F = list(range(N_features))
    
    target_number = int(N_samples*float(Feature_ratio))

    count = N_features
    
    while (count > target_number):
    
        max_acc = 0
    
        for _index_ in range(N_features):
    
            if _index_ in F:
    
                F.remove(_index_)
                X_tmp = X_train[:, F]
                acc = 0
                
                for train, test in skf.split(X_train, y_train):

                    clf.fit(X_tmp[train], 
                            y_train[train])

                    y_predict = clf.predict(X_tmp[test])
                    
                    acc_tmp = accuracy_score(y_train[test], 
                                             y_predict)

                    acc += acc_tmp
                
                acc = float(acc) / (skf.get_n_splits(X=X_train, 
                                                     y=y_train))

                F.append(_index_)

                if acc > max_acc:
                
                    max_acc = acc
                    idx = _index_
        
        F.remove(idx)
        count = count - 1
    
    F = np.array(F, dtype = np.int)
    
    print(f"Backward select {F.shape} features")
    
    if len(list(select_feature_counter)) == 0:
    
        for _key_ in select_map.keys():
    
            select_feature_counter[f"{select_map[_key_][0]}, {select_map[_key_][1]}"] = 0
    
    else:
    
        for _f_ in F:
    
            if f"{select_map[_f_][0]}, {select_map[_f_][1]}" not in select_feature_counter.keys():
    
                select_feature_counter[f"{select_map[_f_][0]}, {select_map[_f_][1]}"] = 1
    
            else:
    
                select_feature_counter[f"{select_map[_f_][0]}, {select_map[_f_][1]}"] += 1

    _, X_test, y_test,_ = get_Data(SelectionResult, 
                                   N_features, 
                                   Test_json_path, 
                                   Label)
    
    print(f"Testing data: {dict(zip(*np.unique(y_test, return_counts=True)))}")
    print(f"Test:    {y_test}")

    X_selected = X_train[:, F]
    
    #sol = SVC(gamma='scale', class_weight='balanced', C=C, tol=tol)
    sol = NuSVC(class_weight='balanced', 
                tol=tol, 
                nu=nu)
    #sol = make_pipeline(StandardScaler(),
    #                    SGDClassifier(
    #                                class_weight='balanced',
    #                                n_jobs = -1,
    #                                random_state = 0,
    #                                validation_fraction=0.25))
    #sol = DecisionTreeClassifier(class_weight='balanced', random_state=0)
    
    sol.fit(X_selected, 
            y_train)

    y_inference = sol.predict(X_test[:, F])

    print(f"Predict: {y_inference}")
    
    y_score = sol.fit(X_selected, 
                      y_train).decision_function(X_test[:, F])
    
    return accuracy_score(y_test, y_inference), X_test[:, F], y_test, y_inference, y_score, sol, select_feature_counter


def forward_selection(SelectionResult, 
                      N_features, 
                      Train_json_path, 
                      Test_json_path, 
                      Label):
    
    forward_result = {}
    
    selection_ratio = args.Feature_ratio

    return forward_result


def get_args():

    API_description = """
***** Radiomics Analysis Platform  *****
API Name: Radiomics SVM Classifier 
Version:    1.0
Developer: Alvin Li
Email:     alvinli@gorilla-technology.com / d05548014@ntu.edu.tw
****************************************

"""

    parser = argparse.ArgumentParser(prog='SVM_Classifier.py',
                                     formatter_class=argparse.RawDescriptionHelpFormatter,
                                     description=API_description)
    
    parser.add_argument('-Table', 
                        action = 'store', 
                        type = str, 
                        help = 'The absolute path to the Feature table (*.txt).')

    parser.add_argument('-LeavePout', 
                        action = 'store', 
                        help = 'an integer, P for leave-P-out cross validation')

    parser.add_argument('-output_folder', 
                        action = 'store', 
                        help = 'The absolute path to the output folder used to store splitted data')

    parser.add_argument('-method', 
                        action = 'store', 
                        help = 'fisher-intra, LASSO')

    parser.add_argument('-Feature_ratio', 
                        action = 'store', 
                        type = str, 
                        help = 'The ratio for selecting features from the all radiomics features by either backward selection or forward selection)')

    parser.add_argument('-Label', 
                        action = 'store', 
                        type = str, 
                        help = 'The target label name, e.g., \"Histological grade\", \"T stage\", and so on')

    #parser.add_argument('-Validate_ratio', action = 'store', type = str, help = 'The ratio for splitting validate data from Train Table for SVM classifier')
    parser.add_argument('-Kfold', 
                        action = 'store', 
                        type = str, 
                        help = 'The number for Stratified k-fold cross validation')
    
    args = parser.parse_args()
    
    return args


def RemoveMissingData(Feature_Table, 
                      Label):

    subject_list = list(Feature_Table.keys())
    
    assert (Label in list(Feature_Table[subject_list[0]].keys())), "The target label is not in Feature Table"
    
    for _subj_ in list(Feature_Table.keys()):
    
        if Feature_Table[_subj_][Label] == 'None':
    
            del Feature_Table[_subj_]
    
    return Feature_Table


def SVM_Classifier(args):

    json_file = open(args.Table, 'r')

    Feature_Table = RemoveMissingData(json.load(json_file), 
                                      args.Label)

    N_subjects = len(list(Feature_Table.keys()))
    
    print(f"{N_subjects} with valid records of {args.Label} will be analyzed ...")
    
    K = int(args.Kfold)
    #rng = random.randint(0, N_subjects)
    
    if args.LeavePout != None:
    
        lpo = LeavePOut(p=int(args.LeavePout))
    
    else:
    
        #skf = StratifiedKFold(n_splits=K, shuffle=True)
        skf = RepeatedStratifiedKFold(n_splits=K, 
                                      n_repeats=30)

    index_arr = np.array(list(range(N_subjects)))
    
    index_y = []
    
    subj_list = list(Feature_Table.keys())
    
    for _subj_ in subj_list:
    
        label = Feature_Table[_subj_][args.Label]
    
        if (label == 'HG'):
    
            index_y.append(1)
    
        elif (label == 'LG'):
    
            index_y.append(0)
    
        else:
    
            continue
    
    index_y = np.array(index_y)
    
    print(f"Data Label:{dict(zip(*np.unique(index_y, return_counts=True)))}")
    
    if args.LeavePout != None:
    
        print(f"Total iteration number:{lpo.get_n_splits(index_arr)}")
    
    else:
    
        print(f"Total iteration number: {skf.get_n_splits(index_arr, index_y)}")

    acc = 0
    
    y_gt = np.array([])
    
    y_pred = np.array([])
    y_score = np.array([])
    
    y_test = np.array([])
    X_test = np.array([[],[]])
    
    if args.LeavePout != None:
    
        ss = lpo.split(index_arr)
    
    else:
    
        ss = skf.split(index_arr, 
                       index_y)
    
    select_feature_counter = {}
    
    #_, ax = plt.subplots()
    
    fig_roc, ax_roc = plt.subplots()
    #fig_det, ax_det = plt.subplots()
    
    tprs = []
    aucs = []
    
    mean_fpr = np.linspace(0, 1, 100)
    
    #fnrs = []
    
    for i, (train_index, test_index) in enumerate(ss):
    
        if args.LeavePout != None:
    
            print(f"Iteration {i+1} ")
    
        else:
    
            print(f"Iteration {i+1} in Stratified {K}-fold cross validation")
    
        Train_json_path, Test_json_path = splitter(Feature_Table, 
                                                   args.output_folder, 
                                                   train_index, 
                                                   test_index)

        os.system(f"python3 {os.path.join(_pwd_, 'Feature_Selection.py')} -Table {Train_json_path} -Label \"{args.Label}\" -output_folder {args.output_folder} -method {args.method}")
        
        if args.method == "fisher-intra":
        
            csv_path = os.path.join(args.output_folder, 
                                    args.Label.split()[1] + "_FisherFeatureSelectionResult.csv")

        if args.method == "LASSO":
            
            csv_path = os.path.join(args.output_folder, 
                                    args.Label.split()[1] + "_LassoFeatureSelectionResult.csv")
        
        SelectionResult, N_features = read_SelectionResult(csv_path)
        
        if N_features <= 30:
        
            print("*** Apply backward selection for SVM ***")
        
            acc_tmp, Xtest, ytest, y_inference, score, classifier, select_feature_counter = backward_selection(SelectionResult, 
                                                                                                               N_features, 
                                                                                                               Train_json_path, 
                                                                                                               Test_json_path, 
                                                                                                               args.Label, 
                                                                                                               args.Feature_ratio, 
                                                                                                               K, 
                                                                                                               select_feature_counter)
        
            print(f"Iteration {i+1}: get accuracy {acc_tmp*100} %" + '\n\n')
            
            #plot_det_curve(classifier, Xtest, ytest, alpha=0.3, lw=1,
            #             ax=ax_det, name=f'DET fold {i+1}')
            #fpr, fnr,_ = det_curve(ytest, score)
            #interp_fnr = np.interp(mean_fpr, fpr, fnr)
            #fnrs.append(interp_fnr)
            #_ = plot_roc_curve(classifier, Xtest, ytest,
            #             name=f'ROC fold {i+1}',
            #             alpha=0.3, lw=1, ax=ax_roc)
            
            fpr, tpr,_ = roc_curve(ytest, 
                                   score)
            
            interp_tpr = np.interp(mean_fpr, 
                                   fpr, 
                                   tpr)

            roc_auc = auc(fpr, 
                          tpr)

            interp_tpr[0] = 0.0
            
            tprs.append(interp_tpr)
            aucs.append(roc_auc)
            
            y_gt = np.concatenate([y_gt, 
                                   ytest])
            
            y_pred = np.concatenate([y_pred, 
                                     y_inference])
            
            y_score = np.concatenate([y_score, 
                                      score])

            y_test = np.concatenate([y_test, 
                                     ytest])
            
            if i==0:
            
                X_test = Xtest
            
            else:
                X_test = np.concatenate([X_test, 
                                         Xtest], axis=0)
            
            acc += acc_tmp
        
        else:
        
            print("\n\n*** Apply forward selection for SVM ***\n")
        
            #final_selection = forward_selection(SelectionResult, N_features, Train_json_path, Test_json_path, args.Label, args.Feature_ratio, args.Validate_ratio)
    
    if args.LeavePout != None:
        
        mean_acc = acc / lpo.get_n_splits(index_arr)
    
    else:
    
        mean_acc = acc / (skf.get_n_splits(X=index_arr, y=index_y))
    
    print(f"averaged accuracy = {mean_acc}")

    print(classification_report(np.array(y_gt), 
                                np.array(y_pred), 
                                target_names=["LG", 
                                              "HG"]))
    
    average_precision = average_precision_score(y_test, 
                                                y_score)
    
    print('Average precision-recall score: {0:0.2f}'.format(average_precision))

    #ax_roc.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r', label='Chance', alpha=.8)
    
    mean_tpr = np.mean(tprs, 
                       axis=0)
                       
    mean_tpr[-1] = 1.0

    mean_auc = auc(mean_fpr, 
                   mean_tpr)

    std_auc = np.std(aucs)

    ax_roc.plot(mean_fpr, 
                mean_tpr, 
                color='b', 
                label=r'Mean ROC (AUC = %0.2f $\pm$ %0.2f)' % (mean_auc, std_auc), 
                lw=2, 
                alpha=.8)

    std_tpr = np.std(tprs, axis=0)
    
    tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
    tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
    
    ax_roc.fill_between(mean_fpr, 
                        tprs_lower, 
                        tprs_upper, 
                        color='grey', 
                        alpha=.2, 
                        label=r'$\pm$ 1 std. dev.')

    ax_roc.set(xlim=[-0.05, 1.05], 
               ylim=[-0.05, 1.05], 
               title="Receiver Operating Characteristic (ROC) curves")
    
    #ax_roc.legend(loc='center left', bbox_to_anchor=(1, 0.5)) 
    ax_roc.legend(loc= "lower right")
    
    ax_roc.set_xlabel('False Positive Rate (Positive label: HG)')
    ax_roc.set_ylabel('True Positive Rate (Positive label: HG)')   
    
    fig_roc.tight_layout()

    #mean_fnr = np.mean(fnrs, axis=0)
    #ax_det.plot(mean_fpr, mean_fnr, color='b', label='Mean DET', lw=2, alpha=.8)
    #std_fnr = np.std(fnrs, axis=0)
    #fnrs_upper = np.minimum(mean_fnr + std_fnr, 1)
    #fnrs_lower = np.maximum(mean_fnr - std_fnr, 0)
    #ax_det.fill_between(mean_fpr, fnrs_lower, fnrs_upper, color='grey', alpha=.2, label=r'$\pm$ 1 std. dev.')
    #ax_det.set(xlim=[-0.05, 1.05], ylim=[-0.05, 1.05], title="Detection Error Tradeoff (DET) curves")
    #ax_det.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    #ax_det.set_xlabel('False Positive Rate (Positive label: HG)')
    #ax_det.set_ylabel('False Negative Rate (Negative label: LG)') 
    #fig_det.tight_layout()

    fig_bar, ax_bar = plt.subplots()
    
    courses = list(select_feature_counter.keys()) 
    values = list(select_feature_counter.values()) 
    value_sum = 0
    
    for _value_ in values:
    
        value_sum += _value_
    
    for _index_ in range(len(values)):
    
        values[_index_] = 100.0 * values[_index_] / value_sum
    
    ax_bar.barh(courses, 
                values, 
                height = 1, 
                color ='grey', 
                edgecolor='b') 

    for s in ['top', 'bottom', 'left', 'right']: 
        
        ax_bar.spines[s].set_visible(False) 
    
    ax_bar.xaxis.set_ticks_position('none') 
    ax_bar.yaxis.set_ticks_position('none') 
    
    ax_bar.xaxis.set_tick_params(pad = 5) 
    ax_bar.yaxis.set_tick_params(pad = 10) 
    
    ax_bar.grid(b = True, 
                color ='grey', 
                linestyle ='-.', 
                linewidth = 0.5, 
                alpha = 0.2) 

    ax_bar.invert_yaxis() 
    
    for i in ax_bar.patches: 
    
        plt.text(i.get_width()+0.2, 
                 i.get_y()+0.5,  
                 str(round((i.get_width()), 2)), 
                 fontsize = 8, 
                 fontweight ='bold', 
                 color ='grey') 

    ax_bar.set_xlabel("Probability (%)") 
    
    fig_bar.tight_layout()

    cm = confusion_matrix(y_test, 
                          y_pred, 
                          normalize='true')

    disp = ConfusionMatrixDisplay(confusion_matrix=cm, 
                                  display_labels=np.array(['LG', 
                                                           'HG']))
    
    disp.plot()
    
    """
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    fpr[0], tpr[0],_ = roc_curve(y_test, y_score)
    roc_auc[0] = auc(fpr[0], tpr[0])
    fpr["micro"], tpr["micro"], _ = roc_curve(y_test.ravel(), y_score.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
    plt.figure()
    plt.plot(fpr[0], tpr[0], color='darkorange', label='ROC curve (area = %0.2f)' % roc_auc[0])
    plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic')
    plt.legend(loc="lower right")
    """
    
    plt.show()
    
    return mean_acc


def main():

    args = get_args()

    mean_acc = SVM_Classifier(args) 


if __name__ == '__main__':

    main()
