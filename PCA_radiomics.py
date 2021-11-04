import argparse
import copy
import os

import csv
import json

import numpy as np
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA
from sklearn.decomposition import FastICA
from sklearn.decomposition import KernelPCA
from sklearn.decomposition import SparsePCA
from sklearn.decomposition import DictionaryLearning
from sklearn.decomposition import LatentDirichletAllocation
from mpl_toolkits.mplot3d import Axes3D


_pwd_ = os.getcwd()

Feature_Table = {}


def LDAnalysis(FeatureArray, 
               LabelVector):

    lda = LatentDirichletAllocation()

    return lda.fit_transform(FeatureArray)


def Dict_LearningAnalysis(FeatureArray, 
                          LabelVector):

    dict_learning = DictionaryLearning()

    return dict_learning.fit_transform(FeatureArray)


def SPCAnalysis(FeatureArray, 
                LabelVector):

    spca = SparsePCA()

    return spca.fit_transform(FeatureArray)


def KPCAnalysis(FeatureArray, 
                LabelVector):

    kpca = KernelPCA()

    return kpca.fit_transform(FeatureArray)


def ICAnalysis(FeatureArray, 
               LabelVector):

    ica = FastICA()

    return ica.fit_transform(FeatureArray)


def PCAnalysis(FeatureArray, 
               LabelVector):

    LG_row_indice = []

    for idx in range(LabelVector.shape[0]):
        
        if LabelVector[idx] == 0:

            LG_row_indice.append(idx)

    HG_FeatureArray = np.delete(FeatureArray, LG_row_indice, 0)
    
    pca = PCA()
    #pca.fit(FeatureArray)
    pca.fit(HG_FeatureArray)

    return pca


def LDA_ScatterPlot(LDA_results, 
                    FeatureArray, 
                    LabelVector):

    projection_matrix = LDA_results

    colors = ['navy', 'darkorange']
    target_names = ['L-grade', 'H-grade']

    plt.scatter(projection_matrix[LabelVector==0, 0], 
                projection_matrix[LabelVector==0, 1], 
                color = colors[0], 
                label=target_names[0], 
                lw=2)

    plt.scatter(projection_matrix[LabelVector==1, 0], 
                projection_matrix[LabelVector==1, 1], 
                color = colors[1], 
                label=target_names[1], 
                lw=2)

    plt.gca().set(aspect='equal')

    plt.legend()

    fig = plt.figure()
    plt.clf()
    
    ax = Axes3D(fig, 
                rect=[0, 0, .95, 1], 
                elev=48, 
                azim=134)
    
    plt.cla()
    
    ax.scatter(projection_matrix[LabelVector==0, 0], 
               projection_matrix[LabelVector==0, 1], 
               projection_matrix[LabelVector==0, 2], 
               color = colors[0], 
               label = target_names[0], 
               lw=2)

    ax.scatter(projection_matrix[LabelVector==1, 0], 
               projection_matrix[LabelVector==1, 1], 
               projection_matrix[LabelVector==1, 2], 
               color = colors[1], 
               label=target_names[1], 
               lw=2)

    plt.legend()

    plt.gca().set(aspect='auto')
    
    plt.show()


def DictionaryLearning_ScatterPlot(DictionaryLearning_results, 
                                   FeatureArray, 
                                   LabelVector):

    projection_matrix = DictionaryLearning_results

    colors = ['navy', 'darkorange']
    target_names = ['L-grade', 'H-grade']

    plt.scatter(projection_matrix[LabelVector==0, 0], 
                projection_matrix[LabelVector==0, 1], 
                color = colors[0], 
                label=target_names[0], 
                lw=2)

    plt.scatter(projection_matrix[LabelVector==1, 0], 
                projection_matrix[LabelVector==1, 1], 
                color = colors[1], 
                label=target_names[1], 
                lw=2)

    plt.gca().set(aspect='equal')

    plt.legend()

    fig = plt.figure()
    plt.clf()
    
    ax = Axes3D(fig, 
                rect=[0, 0, .95, 1], 
                elev=48, 
                azim=134)
    
    plt.cla()
    
    ax.scatter(projection_matrix[LabelVector==0, 0], 
               projection_matrix[LabelVector==0, 1], 
               projection_matrix[LabelVector==0, 2], 
               color = colors[0], 
               label = target_names[0], 
               lw=2)

    ax.scatter(projection_matrix[LabelVector==1, 0], 
               projection_matrix[LabelVector==1, 1], 
               projection_matrix[LabelVector==1, 2], 
               color = colors[1], 
               label=target_names[1], 
               lw=2)

    plt.legend()

    plt.gca().set(aspect='auto')
    
    plt.show()


def SPCA_ScatterPlot(SPCA_results, 
                    FeatureArray, 
                    LabelVector):

    projection_matrix = SPCA_results

    colors = ['navy', 'darkorange']
    target_names = ['L-grade', 'H-grade']

    plt.scatter(projection_matrix[LabelVector==0, 0], 
                projection_matrix[LabelVector==0, 1], 
                color = colors[0], 
                label=target_names[0], 
                lw=2)

    plt.scatter(projection_matrix[LabelVector==1, 0], 
                projection_matrix[LabelVector==1, 1], 
                color = colors[1], 
                label=target_names[1], 
                lw=2)

    plt.gca().set(aspect='equal')

    plt.legend()

    fig = plt.figure()
    plt.clf()
    
    ax = Axes3D(fig, 
                rect=[0, 0, .95, 1], 
                elev=48, 
                azim=134)
    
    plt.cla()
    
    ax.scatter(projection_matrix[LabelVector==0, 0], 
               projection_matrix[LabelVector==0, 1], 
               projection_matrix[LabelVector==0, 2], 
               color = colors[0], 
               label = target_names[0], 
               lw=2)

    ax.scatter(projection_matrix[LabelVector==1, 0], 
               projection_matrix[LabelVector==1, 1], 
               projection_matrix[LabelVector==1, 2], 
               color = colors[1], 
               label=target_names[1], 
               lw=2)

    plt.legend()

    plt.gca().set(aspect='auto')
    
    plt.show()


def KPCA_ScatterPlot(KPCA_results, 
                     FeatureArray, 
                     LabelVector):

    projection_matrix = KPCA_results

    colors = ['navy', 'darkorange']
    target_names = ['L-grade', 'H-grade']

    plt.scatter(projection_matrix[LabelVector==0, 0], 
                projection_matrix[LabelVector==0, 1], 
                color = colors[0], 
                label=target_names[0], 
                lw=2)

    plt.scatter(projection_matrix[LabelVector==1, 0], 
                projection_matrix[LabelVector==1, 1], 
                color = colors[1], 
                label=target_names[1], 
                lw=2)

    plt.gca().set(aspect='equal')

    plt.legend()

    fig = plt.figure()
    plt.clf()
    
    ax = Axes3D(fig, 
                rect=[0, 0, .95, 1], 
                elev=48, 
                azim=134)
    
    plt.cla()
    
    ax.scatter(projection_matrix[LabelVector==0, 0], 
               projection_matrix[LabelVector==0, 1], 
               projection_matrix[LabelVector==0, 2], 
               color = colors[0], 
               label = target_names[0], 
               lw=2)

    ax.scatter(projection_matrix[LabelVector==1, 0], 
               projection_matrix[LabelVector==1, 1], 
               projection_matrix[LabelVector==1, 2], 
               color = colors[1], 
               label=target_names[1], 
               lw=2)

    plt.legend()

    plt.gca().set(aspect='auto')
    
    plt.show()


def IC_ScatterPlot(ICA_results, 
                   FeatureArray, 
                   LabelVector):

    projection_matrix = ICA_results

    colors = ['navy', 'darkorange']
    target_names = ['L-grade', 'H-grade']

    plt.scatter(projection_matrix[LabelVector==0, 0], 
                projection_matrix[LabelVector==0, 1], 
                color = colors[0], 
                label=target_names[0], 
                lw=2)

    plt.scatter(projection_matrix[LabelVector==1, 0], 
                projection_matrix[LabelVector==1, 1], 
                color = colors[1], 
                label=target_names[1], 
                lw=2)

    plt.gca().set(aspect='equal')

    plt.legend()

    fig = plt.figure()
    plt.clf()
    
    ax = Axes3D(fig, 
                rect=[0, 0, .95, 1], 
                elev=48, 
                azim=134)
    
    plt.cla()
    
    ax.scatter(projection_matrix[LabelVector==0, 0], 
               projection_matrix[LabelVector==0, 1], 
               projection_matrix[LabelVector==0, 2], 
               color = colors[0], 
               label = target_names[0], 
               lw=2)

    ax.scatter(projection_matrix[LabelVector==1, 0], 
               projection_matrix[LabelVector==1, 1], 
               projection_matrix[LabelVector==1, 2], 
               color = colors[1], 
               label=target_names[1], 
               lw=2)

    plt.legend()

    plt.gca().set(aspect='auto')
    
    plt.show()


def PC_ScatterPlot(PCA_results, 
                   FeatureArray, 
                   LabelVector):
    
    print(f"{PCA_results.explained_variance_ratio_ * 100}")

    projection_matrix = np.matmul(FeatureArray, PCA_results.components_.T)

    colors = ['navy', 'darkorange']
    target_names = ['L-grade', 'H-grade']

    plt.scatter(projection_matrix[LabelVector==0, 0], 
                projection_matrix[LabelVector==0, 1], 
                color = colors[0], 
                label=target_names[0], 
                lw=2)

    plt.scatter(projection_matrix[LabelVector==1, 0], 
                projection_matrix[LabelVector==1, 1], 
                color = colors[1], 
                label=target_names[1], 
                lw=2)

    plt.gca().set(aspect='equal',
                  xlabel=f'first feature ({PCA_results.explained_variance_ratio_[0] * 100:.2f} %)', 
                  ylabel=f'second feature ({PCA_results.explained_variance_ratio_[1] * 100:.2f} %)')

    plt.legend()

    fig = plt.figure()
    plt.clf()
    
    ax = Axes3D(fig, 
                rect=[0, 0, .95, 1], 
                elev=48, 
                azim=134)
    
    plt.cla()
    
    ax.scatter(projection_matrix[LabelVector==0, 0], 
               projection_matrix[LabelVector==0, 1], 
               projection_matrix[LabelVector==0, 2], 
               color = colors[0], 
               label = target_names[0], 
               lw=2)

    ax.scatter(projection_matrix[LabelVector==1, 0], 
               projection_matrix[LabelVector==1, 1], 
               projection_matrix[LabelVector==1, 2], 
               color = colors[1], 
               label=target_names[1], 
               lw=2)

    plt.legend()

    plt.gca().set(aspect='auto',
                  xlabel=f'first feature ({PCA_results.explained_variance_ratio_[0] * 100:.2f} %)', 
                  ylabel=f'second feature ({PCA_results.explained_variance_ratio_[1] * 100:.2f} %)', 
                  zlabel=f'third feature ({PCA_results.explained_variance_ratio_[2] * 100:.2f} %)')
    
    plt.show()


def Load_Features(Table):

    global Feature_Table

    json_file = open(Table, 'r')
    Feature_Table = json.load(json_file)


def RemoveMissingData(LabelName):

    global Feature_Table

    subject_list = list(Feature_Table.keys())

    assert (LabelName in list(Feature_Table[subject_list[0]].keys())), "The target label is not in Feature Table"

    for _subj_ in list(Feature_Table.keys()):

        if Feature_Table[_subj_][LabelName] == 'None':

            del Feature_Table[_subj_]


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
    
                if "shape2D" in _radiomics_:
    
                    pass
    
                else:
    
                    feature_values.append(Feature_Table[_subj_]['Features'][_filter_][_radiomics_])

        Features.append(feature_values)
    
    print(f"We analyze {count} participants")
    
    LabelVector = np.array(Label, dtype=np.int)
    FeatureArray = np.array(Features, dtype=np.double)
    
    print(f"Feature Array shape: {FeatureArray.shape}")
    print(f"Label Vector shape: {LabelVector.shape}")
    
    return FeatureArray, LabelVector


def main():
    
    API_description = """
***** Radiomics Analysis Platform  *****
API Name: Principle Component Analysis for Radiomics Features
Version:    1.0
Developer: Alvin Li
Email:     d05548014@ntu.edu.tw
****************************************

"""

    parser = argparse.ArgumentParser(prog = 'PCA_radiomics.py',
                                     formatter_class = argparse.RawDescriptionHelpFormatter,
                                     description = API_description)

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
                        help = 'The absolute path to the output folder')
    
    parser.add_argument('-method', 
                        action = 'store', 
                        help = 'LDA, Dict-Learn, PCA, KPCA, SPCA, or ICA')
    
    args = parser.parse_args()

    Load_Features(args.Table)
    RemoveMissingData(args.Label)

    if args.method == 'PCA':

        PC_ScatterPlot(PCAnalysis(*GetFeatureArray(args.Label)), 
                       *GetFeatureArray(args.Label))

    elif args.method == 'ICA':

        IC_ScatterPlot(ICAnalysis(*GetFeatureArray(args.Label)), 
                       *GetFeatureArray(args.Label))

    elif args.method == 'KPCA':

        KPCA_ScatterPlot(KPCAnalysis(*GetFeatureArray(args.Label)), 
                         *GetFeatureArray(args.Label))

    elif args.method == 'SPCA':

        SPCA_ScatterPlot(SPCAnalysis(*GetFeatureArray(args.Label)), 
                         *GetFeatureArray(args.Label))

    elif args.method == 'Dict-Learn':

        DictionaryLearning_ScatterPlot(Dict_LearningAnalysis(*GetFeatureArray(args.Label)), 
                                       *GetFeatureArray(args.Label))

    elif args.method == 'LDA':

        LDA_ScatterPlot(LDAnalysis(*GetFeatureArray(args.Label)), 
                                   *GetFeatureArray(args.Label))


if __name__ == '__main__':

    main()
