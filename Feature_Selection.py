import argparse
import os
import glob
import copy

import json
import csv

import numbers
import numpy as np

from scipy.sparse import *
from sklearn.linear_model import Lasso
from sklearn.metrics.pairwise import pairwise_distances
from sklearn.base import BaseEstimator, clone, MetaEstimatorMixin
from sklearn.feature_selection._base import SelectorMixin, _get_feature_importances
from sklearn.utils._tags import _safe_tags
from sklearn.utils.validation import check_is_fitted, _deprecate_positional_args
from sklearn.utils.metaestimators import if_delegate_has_method
from sklearn.exceptions import NotFittedError

_pwd_ = os.getcwd()

Feature_Table = {}
Features = []
selected_Features = {}
IntraScores = {}


def _calculate_threshold(estimator, 
                         importances, 
                        threshold):

    if threshold is None:
    
        est_name = estimator.__class__.__name__
    
        if ((hasattr(estimator, "penalty") and 
            estimator.penalty == "l1") or 
            "Lasso" in est_name):

            threshold = 1e-5
        
        else:
        
            threshold = "mean"
    
    if isinstance(threshold, str):
    
        if "*" in threshold:
    
            scale, reference = threshold.split("*")
    
            scale = float(scale.strip())
            reference = reference.strip()
    
            if reference == "median":
    
                reference = np.median(importances)
    
            elif reference == "mean":
    
                reference = np.mean(importances)
    
            else:
    
                raise ValueError("Unknown reference: " + reference)
    
            threshold = scale * reference
    
        elif threshold == "median":
    
            threshold = np.median(importances)
    
        elif threshold == "mean":
    
            threshold = np.mean(importances)
    
        else:
    
            raise ValueError("Expected threshold='mean' or threshold='median' ""got %s" % threshold)
    
    else:
    
        threshold = float(threshold)
    
    return threshold


class SelectFromModel(MetaEstimatorMixin, 
                      SelectorMixin, 
                      BaseEstimator):

    @_deprecate_positional_args
    def __init__(self, 
                 estimator, 
                 *, 
                 threshold=None, 
                 prefit=False, 
                 norm_order=1, 
                 max_features=None, 
                 importance_getter='auto'):

        self.estimator = estimator
        self.threshold = threshold
        self.prefit = prefit
        self.importance_getter = importance_getter
        self.norm_order = norm_order
        self.max_features = max_features
    
    def _get_support_mask(self):
    
        if self.prefit:
    
            estimator = self.estimator
    
        elif hasattr(self, 'estimator_'):
    
            estimator = self.estimator_
    
        else:
    
            raise ValueError('Either fit the model before transform or set'' "prefit=True" while passing the fitted'' estimator to the constructor.')
    
        scores = _get_feature_importances(estimator=estimator, 
                                          getter=self.importance_getter, 
                                          transform_func='norm', 
                                          norm_order=self.norm_order)
    
        threshold = _calculate_threshold(estimator, 
                                         scores, 
                                         self.threshold)

        if self.max_features is not None:
            
            mask = np.zeros_like(scores, dtype=bool)
            
            candidate_indices = np.argsort(-scores, kind='mergesort')[:self.max_features]
            
            mask[candidate_indices] = True
        
        else:
        
            mask = np.ones_like(scores, dtype=bool)
        
        mask[scores < threshold] = False
        
        return mask
    
    
    def fit(self, 
            X, 
            y=None, 
            **fit_params):
    
        if self.max_features is not None:
            
            if not isinstance(self.max_features, numbers.Integral):
            
                raise TypeError("'max_features' should be an integer between"" 0 and {} features. Got {!r} instead.".format(X.shape[1], self.max_features))
            
            elif (self.max_features < 0 or 
                  self.max_features > X.shape[1]):

                raise ValueError("'max_features' should be 0 and {} features.""Got {} instead.".format(X.shape[1], self.max_features))
        
        if self.prefit:
        
            raise NotFittedError("Since 'prefit=True', call transform directly")
        
        self.estimator_ = clone(self.estimator)
        self.estimator_.fit(X, 
                            y, 
                            **fit_params)

        return self


    @property
    def threshold_(self):

        scores = _get_feature_importances(estimator=self.estimator_, 
                                          getter=self.importance_getter, 
                                          transform_func='norm', 
                                          norm_order=self.norm_order)

        return _calculate_threshold(self.estimator, 
                                    scores, 
                                    self.threshold)

    @if_delegate_has_method('estimator')
    def partial_fit(self, 
                    X, 
                    y=None, 
                    **fit_params):

        """Fit the SelectFromModel meta-transformer only once."""
        
        if self.prefit:
        
            raise NotFittedError("Since 'prefit=True', call transform directly")
        
        if not hasattr(self, "estimator_"):
        
            self.estimator_ = clone(self.estimator)
            self.estimator_.partial_fit(X, 
                                        y, 
                                        **fit_params)
        
            return self
    
    
    @property
    def n_features_in_(self):
        
        try:
        
            check_is_fitted(self)
        
        except NotFittedError as nfe:
        
            raise AttributeError("{} object has no n_features_in_ attribute.".format(self.__class__.__name__)) from nfe
        
        return self.estimator_.n_features_in_


    def _more_tags(self):

        return {'allow_nan': _safe_tags(self.estimator, key="allow_nan")}


def assert_paser_valid(args):

    assert (os.path.exists(args.Table)), "The data table cannot be found"

    if (os.path.exists(args.output_folder) == False):

        os.mkdir(args.output_folder)


def Load_Features(Table):

    global Feature_Table

    json_file = open(Table, 'r')

    Feature_Table = json.load(json_file)


def GetIntraFeatureArray(LabelName):

    global IntraScores

    Label = []
    Features = []

    subj_list = list(Feature_Table.keys())
    filter_list = list(Feature_Table[subj_list[0]]['Features'].keys())
    Original_radiomics_list = list(Feature_Table[subj_list[0]]['Features'][filter_list[0]].keys())

    shape_feature_count = 0
    other_feature_count = 0

    for _original_radiomics_ in Original_radiomics_list:

        if "shape2D" in _original_radiomics_:

            shape_feature_count += 1

            continue

        else:

            other_feature_count += 1
            IntraScores[_original_radiomics_] = {}

            for _filter_ in filter_list:

                IntraScores[_original_radiomics_][_filter_] = []

    assert (LabelName in Feature_Table[subj_list[0]].keys()), "Input Label cannot be found in the given Feature Table"

    print(f"{shape_feature_count} shape features")
    print(f"{other_feature_count} other features")
    count = 0 

    for _subj_ in subj_list:

        label = Feature_Table[_subj_][LabelName]

        if (label == 'HG'):

            Label.append(1)
            count += 1

        elif (label == 'LG'):
        
            Label.append(0)
            count += 1
        
        else:
        
            continue
        
        feature_values = []
        
        for _filter_ in filter_list:
        
            radiomics_list = list(Feature_Table[_subj_]['Features'][_filter_].keys())
        
            for _radiomics_ in radiomics_list:
        
                if "shape2D" in _radiomics_:
        
                    continue
        
                else:
        
                    feature_values.append(Feature_Table[_subj_]['Features'][_filter_][_radiomics_])

        Features.append(feature_values)
    
    print(f"We analyze {count} participants")
    
    LabelVector = np.array(Label, dtype=np.int)
    
    IntraFeatureArray = np.array(Features, dtype=np.double)
    
    return IntraFeatureArray, LabelVector


def GetFeatureArray(LabelName):
    
    Label = []
    Features = []
    
    subj_list = list(Feature_Table.keys())
    filter_list = list(Feature_Table[subj_list[0]]['Features'].keys())
    Original_radiomics_list = list(Feature_Table[subj_list[0]]['Features'][filter_list[0]].keys())
    
    shape_feature_count = 0
    other_feature_count = 0
    
    for _original_radiomics_ in Original_radiomics_list:
    
        if "shape2D" in _original_radiomics_:
    
            shape_feature_count += 1
    
        else:
    
            other_feature_count += 1            
    
    assert (LabelName in Feature_Table[subj_list[0]].keys()), "Input Label cannot be found in the given Feature Table"
    
    print(f"{shape_feature_count} shape features")
    print(f"{other_feature_count} other features")
    count = 0 
    
    for _subj_ in subj_list:
    
        label = Feature_Table[_subj_][LabelName]
    
        if (label == 'HG'):
    
            Label.append(1)
            count += 1
    
        elif (label == 'LG'):
    
            Label.append(0)
            count += 1
    
        else:
    
            continue
    
        feature_values = []
    
        for _filter_ in filter_list:
    
            radiomics_list = list(Feature_Table[_subj_]['Features'][_filter_].keys())
    
            for _radiomics_ in radiomics_list:
    
                feature_values.append(Feature_Table[_subj_]['Features'][_filter_][_radiomics_])

        Features.append(feature_values)
    
    print(f"We analyze {count} participants")
    
    LabelVector = np.array(Label, dtype=np.int)
    FeatureArray = np.array(Features, dtype=np.double)
    
    return FeatureArray, LabelVector


def get_Weight_matrix(FeatureArray, 
                      LabelVector):

    n_samples, _ = np.shape(FeatureArray)
    
    label = np.unique(LabelVector)
    
    n_classes = np.unique(label).size
    
    W = lil_matrix((n_samples, n_samples))
    
    for i in range(n_classes):
    
        class_idx = (LabelVector == label[i])
        class_idx_all = (class_idx[:, np.newaxis] & class_idx[np.newaxis, :])
        
        W[class_idx_all] = 1.0/np.sum(np.sum(class_idx))
    
    return W


def IntraFeatureScoring(FeatureArray, 
                        LabelVector): 
    
    #use Fisher Score
    
    W = get_Weight_matrix(FeatureArray, 
                          LabelVector)

    D = np.array(W.sum(axis=1))
    
    tmp = np.dot(np.transpose(D), 
                 FeatureArray)
    
    D = diags(np.transpose(D), [0])
    
    Xt = np.transpose(FeatureArray)
    
    t1 = np.transpose(np.dot(Xt, D.todense()))
    t2 = np.transpose(np.dot(Xt, W.todense()))
    
    D_prime = np.sum( np.multiply(t1, FeatureArray), 0 ) - np.multiply(tmp, tmp) / D.sum()
    L_prime = np.sum( np.multiply(t2, FeatureArray), 0 ) - np.multiply(tmp, tmp) / D.sum()
    
    D_prime[D_prime < 1e-12] = 10000
    
    lap_score = 1 - np.array( np.multiply(L_prime, 1/D_prime) )[0, :]
    
    scores = 1.0/lap_score - 1
    
    return scores


def GetIntraScores_Table(FeatureArray, 
                         LabelVector):

    global IntraScores

    scores = IntraFeatureScoring(FeatureArray, 
                                 LabelVector)

    subj_list = list(Feature_Table.keys())
    filter_list = list(Feature_Table[subj_list[0]]['Features'].keys())

    Original_radiomics_list = list(Feature_Table[subj_list[0]]['Features'][filter_list[0]].keys())
    count = 0

    for _filter_ in filter_list:

        radiomics_list = list(Feature_Table[subj_list[0]]['Features'][_filter_].keys())

        for _index_ in range(len(radiomics_list)):

            _radiomics_ = radiomics_list[_index_]

            if "shape2D" in _radiomics_:

                continue

            else:

                if _filter_ == 'Wavelet':

                    for _original_radiomics_ in Original_radiomics_list:

                        if _radiomics_ == f"H_{_original_radiomics_}":

                            IntraScores[_original_radiomics_][_filter_].append(scores[count])

                        elif _radiomics_ == f"L_{_original_radiomics_}":

                            IntraScores[_original_radiomics_][_filter_].append(scores[count])

                else:

                    IntraScores[_radiomics_][_filter_].append(scores[count])

                count += 1


def save_IntraScores_Table(LabelName, 
                           output_folder):

    GetIntraScores_Table(*GetIntraFeatureArray(LabelName))
    
    csv_path = os.path.join(output_folder, 
                            LabelName.split()[1] + "_IntraFeatureScores.csv")

    csv_file = open(csv_path, 'w')
    
    writer = csv.writer(csv_file, dialect='excel')
    
    headers = []
    headers.append(' ')
    
    subj_list = list(Feature_Table.keys())
    filter_list = list(Feature_Table[subj_list[0]]['Features'].keys())
    
    for _filter_ in filter_list:
    
        headers.append(_filter_)
    
    writer.writerow(headers)
    
    for _key_ in list(IntraScores.keys()):
    
        _line_ = []
    
        _line_.append(_key_)
    
        for _filter_ in filter_list:
    
            _line_.append(IntraScores[_key_][_filter_])
    
        writer.writerow(_line_)
    
    csv_file.close()


def GetIntraFeatureSelectionResult():
    
    key_list = list(IntraScores.keys())
    inner_key_list = list(IntraScores[key_list[0]].keys())
    
    Select_Table = {}
    
    for _key_ in key_list:
    
        Select_Table[_key_] = ""
        fisher_score = 0
    
        for _inner_key_ in inner_key_list:
    
            if _inner_key_ == 'Wavelet':
    
                score1 = IntraScores[_key_][_inner_key_][0]
                score2 = IntraScores[_key_][_inner_key_][1]
    
                if (score1 > score2 and 
                    score1 > fisher_score):
    
                    Select_Table[_key_] = f"{_inner_key_} High Pass"
                    fisher_score = score1
    
                if (score2 > score1 and 
                    score2 > fisher_score):
    
                    Select_Table[_key_] = f"{_inner_key_} Low Pass"
                    fisher_score = score2
    
            else:
    
                if IntraScores[_key_][_inner_key_][0] > fisher_score:
    
                    Select_Table[_key_] = _inner_key_    
                    fisher_score = IntraScores[_key_][_inner_key_][0]
    
    return Select_Table


def save_IntraFeatureSelectionResult(LabelName, 
                                     output_folder):
    
    Select_Table = GetIntraFeatureSelectionResult()
    
    csv_path = os.path.join(output_folder, 
                            LabelName.split()[1] + "_IntraFeatureSelectionResult.csv")

    csv_file = open(csv_path, 'w')
    
    writer = csv.writer(csv_file, dialect='excel')
    
    headers = ['Feature Name', 
               'Selected Filter']
    
    writer.writerow(headers)
    
    for _key_ in list(Select_Table.keys()):
    
        _line_ = []
    
        _line_.append(_key_)
        _line_.append(Select_Table[_key_])
    
        writer.writerow(_line_)
    
    csv_file.close()


def GetFeatureSelectionResult(selected_msk):
    
    Select_Table = {}
    
    Feature_Selected_Table = copy.deepcopy(Feature_Table)
    
    subj_list = list(Feature_Table.keys())
    filter_list = list(Feature_Table[subj_list[0]]['Features'].keys())
    
    for _subj_ in subj_list:
    
        count = 0
    
        for _filter_ in filter_list:
    
            radiomics_list = list(Feature_Selected_Table[_subj_]['Features'][_filter_].keys())
    
            for _index_ in range(len(radiomics_list)):
    
                _radiomics_ = radiomics_list[_index_]
    
                if selected_msk[count]:
    
                    pass
    
                else:
    
                    del Feature_Selected_Table[_subj_]['Features'][_filter_][_radiomics_]
    
                count += 1

    subj_list = list(Feature_Table.keys())
    filter_list = list(Feature_Table[subj_list[0]]['Features'].keys())
    
    count = 0
    
    for _filter_ in filter_list:
    
        radiomics_list = list(Feature_Table[subj_list[0]]['Features'][_filter_].keys())
    
        for _index_ in range(len(radiomics_list)):
    
            _radiomics_ = radiomics_list[_index_]
    
            if selected_msk[count]:
    
                if _radiomics_ in Select_Table.keys():
    
                    Select_Table[_radiomics_].append(_filter_)
    
                else:
    
                    Select_Table[_radiomics_] = [_filter_]
    
            count += 1
    
    return Select_Table, Feature_Selected_Table


def GetInterFeatureSelectionResult(selected_msk):
    
    IntraSelect_Table = GetIntraFeatureSelectionResult()
    Select_Table = {}
    
    subj_list = list(Feature_Table.keys())
    filter_list = list(Feature_Table[subj_list[0]]['Features'].keys())
    
    count = 0
    
    for _filter_ in filter_list:
    
        radiomics_list = list(Feature_Table[subj_list[0]]['Features'][_filter_].keys())
    
        for _index_ in range(len(radiomics_list)):
    
            _radiomics_ = radiomics_list[_index_]
    
            if "shape2D" in _radiomics_:
    
                continue
    
            else:
    
                if _filter_ == 'Wavelet':
    
                    _key_ = '_'.join(_radiomics_.split('_')[1:])
    
                    check_radiomics_H = f"{_filter_} High Pass"
                    check_radiomics_L = f"{_filter_} Low Pass"
    
                    if (check_radiomics_H == IntraSelect_Table[_key_] and 
                        _radiomics_.split('_')[0] == 'H'):
    
                        if selected_msk[count]:
    
                            Select_Table[_radiomics_] = [_filter_]
    
                        count += 1
    
                    if (check_radiomics_L == IntraSelect_Table[_key_] and 
                        _radiomics_.split('_')[0] == 'L'):
    
                        if selected_msk[count]:
    
                            Select_Table[_radiomics_] = [_filter_]
    
                        count += 1
    
                else:
    
                    if _filter_ == IntraSelect_Table[_radiomics_]:
    
                        if selected_msk[count]:
    
                            Select_Table[_radiomics_] = [_filter_]
    
                        count += 1
    
    return Select_Table


def InterFeatureReduction(args):
    
    Select_Table = GetIntraFeatureSelectionResult()
    
    Label = []
    Features = []
    
    subj_list = list(Feature_Table.keys())
    filter_list = list(Feature_Table[subj_list[0]]['Features'].keys())    
    
    count = 0 
    
    for _subj_ in subj_list:
    
        label = Feature_Table[_subj_][args.Label]
    
        if (label == 'HG'):
    
            Label.append(1)
            count += 1
    
        elif (label == 'LG'):
    
            Label.append(0)
            count += 1
    
        else:
    
            continue
    
        feature_values = []

        for _filter_ in filter_list:

            radiomics_list = list(Feature_Table[_subj_]['Features'][_filter_].keys())

            for _radiomics_ in radiomics_list:

                if "shape2D" in _radiomics_:

                    continue

                else:

                    if _filter_ == 'Wavelet':

                        _key_ = '_'.join(_radiomics_.split('_')[1:])

                        check_radiomics_H = f"{_filter_} High Pass"
                        check_radiomics_L = f"{_filter_} Low Pass"
                        if (check_radiomics_H == Select_Table[_key_] and 
                            _radiomics_.split('_')[0] == 'H'):

                            feature_values.append(Feature_Table[_subj_]['Features'][_filter_][_radiomics_])
                        
                        if (check_radiomics_L == Select_Table[_key_] and 
                            _radiomics_.split('_')[0] == 'L'):
                        
                            feature_values.append(Feature_Table[_subj_]['Features'][_filter_][_radiomics_])
                    else:
                        
                        if _filter_ == Select_Table[_radiomics_]:
                        
                            feature_values.append(Feature_Table[_subj_]['Features'][_filter_][_radiomics_])

        Features.append(feature_values)
    
    #print(f"We analyze {count} participants")
    LabelVector = np.array(Label, dtype=np.int)
    FeatureArray = np.array(Features, dtype=np.double)
    print(f"FeatureArray shape: {FeatureArray.shape}, LabelVector shape: {LabelVector.shape}")
    lasso_selector = SelectFromModel(Lasso(alpha=0.002))
    lasso_selector.fit(FeatureArray, LabelVector)
    selected_msk = lasso_selector.get_support()
    print(f"selected mask shape: {selected_msk.shape}")
    print(f"{np.count_nonzero(selected_msk)} features selected")
    Selected_Table = GetInterFeatureSelectionResult(selected_msk)
    csv_path = os.path.join(args.output_folder, args.Label.split()[1] + "_FisherFeatureSelectionResult.csv")
    csv_file = open(csv_path, 'w')
    writer = csv.writer(csv_file, dialect='excel')
    headers = ['Feature Name', 'Selected Filter']
    writer.writerow(headers)
    for _key_ in list(Selected_Table.keys()):
        _line_ = []
        _line_.append(_key_)
        _line_.append(Selected_Table[_key_])
        writer.writerow(_line_)
    csv_file.close()


def Fisher_Feature_Selection(args):
    
    print("Initializing Fisher Selector ...")
    
    save_IntraScores_Table(args.Label, 
                           args.output_folder)
    
    save_IntraFeatureSelectionResult(args.Label, 
                                     args.output_folder)
    
    InterFeatureReduction(args)


def save_LASSO_FeatureSelectionResult(LabelName, 
                                      selected_msk, 
                                      output_folder):

    Select_Table, Feature_Selected_Table = GetFeatureSelectionResult(selected_msk)
    
    csv_path = os.path.join(output_folder, 
                            LabelName.split()[1] + "_LassoFeatureSelectionResult.csv")

    csv_file = open(csv_path, 'w')
    writer = csv.writer(csv_file, dialect='excel')
    
    headers = ['Feature Name', 
               'Selected Filter']
    
    writer.writerow(headers)
    
    for _key_ in list(Select_Table.keys()):
    
        _line_ = []
    
        _line_.append(_key_)
        _line_.append(Select_Table[_key_])
    
        writer.writerow(_line_)
    
    csv_file.close()

    json_path = os.path.join(output_folder, 
                             LabelName.split()[1] + "_LassoFeatureSelectionResult.txt")

    json_file = open(json_path, 'w')
    
    json_content = json.dumps(Feature_Selected_Table, 
                              indent = 4)

    json_file.writelines(json_content)
    
    json_file.close()


def LASSO_Feature_Selection(args):
    
    print("Initializing Lasso Feature Selector ...")
    
    FeatureArray, LabelVector = GetFeatureArray(args.Label)
    
    print(f"FeatureArray shape: {FeatureArray.shape}, LabelVector shape: {LabelVector.shape}")
    
    lasso_selector = SelectFromModel(Lasso(alpha=0.005))
    lasso_selector.fit(FeatureArray, LabelVector)
    
    selected_msk = lasso_selector.get_support()
    
    print(f"selected mask shape: {selected_msk.shape}")
    print(f"{np.count_nonzero(selected_msk)} features selected by LASSO")
    
    save_LASSO_FeatureSelectionResult(args.Label, 
                                      selected_msk, 
                                      args.output_folder)


def main():

    API_description = """
***** Radiomics Analysis Platform  *****
API Name: Radiomics Feature Selection
Version:    1.0
Developer: Alvin Li
Email:     alvinli@gorilla-technology.com / d05548014@ntu.edu.tw
****************************************

"""

    parser = argparse.ArgumentParser(prog='Feature_Selection.py',
                                     formatter_class=argparse.RawDescriptionHelpFormatter,
                                     description=API_description)

    parser.add_argument('-Table', 
                        action = 'store', 
                        type = str, 
                        help = 'The absolute path to the DATA TABLE (*.json).')

    parser.add_argument('-Label', 
                        action = 'store', 
                        type = str, 
                        help = 'The target label name, e.g., \"Histological grade\", \"T stage\", and so on')

    parser.add_argument('-output_folder', 
                        action = 'store', 
                        help = 'The absolute path to the output folder used to store selected Feature Table')

    parser.add_argument('-method', 
                        action = 'store', 
                        help = 'fisher-intra, LASSO')

    args = parser.parse_args()

    assert_paser_valid(args)
    
    Load_Features(args.Table)
    
    if args.method == 'fisher-intra':
    
        Fisher_Feature_Selection(args)
    
    elif args.method == 'LASSO':
    
        LASSO_Feature_Selection(args)


if __name__ == '__main__':
    
    main()
