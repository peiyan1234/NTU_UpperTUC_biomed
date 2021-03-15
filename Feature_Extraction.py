import argparse
import os
import glob
import copy

import csv
import json

import numpy as np

from PIL import Image

import nrrd
import radiomics
from radiomics import featureextractor
import SimpleITK as sitk

_pwd_ = os.getcwd()

data_Table = {}
Feature_Table = {}

hyperparameters = {}

hyperparameters['setting'] = {}
hyperparameters['force2D'] = True
hyperparameters['force2Ddimension'] = 0


def assert_paser_valid(args):

    assert (os.path.exists(args.input_root)), "The image root folder cannot be found"

    if args.Table != None:

        assert (os.path.exists(args.Table)), "The data table cannot be found"

    assert (len(args.Volume) != 0), "Input volume cannot be found"
    assert (len(args.Mask) != 0), "Input Mask cannot be found"
    assert (len(args.Mask) == len(args.Volume)), "The number of Masks is not consistent with the number of Volumes."

    if os.path.exists(args.output_folder) == False:

        os.mkdir(args.output_folder)

    if args.Volume[0] == 'all':

        assert (args.Mask[0]) == 'all', "-Mask: should be \'all\'"

    assert (isinstance(eval(args.width), float) or 
            isinstance(eval(args.width), int)), "-width: should be a float/int number"

    assert (isinstance(eval(args.level), float) or 
            isinstance(eval(args.level), int)), "-level: should be a float/int number"


def read_data_Table(Table_path):

    global data_Table

    data_csv = open(Table_path, 'r')
    csv_reader = csv.reader(data_csv, delimiter = ',')

    for row in csv_reader:

        ID = row[0]
        data_Table[ID] = row

    data_csv.close()


def read_data(args):

    global Feature_Table

    Vols = []
    Segs = []

    Folder_Vol = os.path.join(args.input_root, 
                              'crop_vol')

    Folder_Seg = os.path.join(args.input_root, 
                              'crop_msk')

    if args.Volume[0] == 'all':

        Vols = sorted( glob.glob( os.path.join(Folder_Vol, 
                                               'UC*')))

        Segs = sorted( glob.glob( os.path.join(Folder_Seg, 
                                               'UC*')))
        
        for _index_ in range(len(Vols)):
            
            ID = os.path.basename(Vols[_index_]).split('_')[0]
            
            Feature_Table[ID] = {}
            
            Feature_Table[ID]['Type'] = 'UTUC'
            Feature_Table[ID]['Sex'] = data_Table[ID][2]
            
            Grade_info = data_Table[ID][4]
            
            if ('High' in Grade_info or 
                'high' in Grade_info):
            
                Feature_Table[ID]['Histological grade'] = 'HG'
            
            elif ('Low' in Grade_info or 
                  'low' in Grade_info):

                Feature_Table[ID]['Histological grade'] = 'LG'
            
            else:
            
                Feature_Table[ID]['Histological grade'] = 'None'
            
            if (data_Table[ID][6] == '' or 
                data_Table[ID][6] == None):
            
                Feature_Table[ID]['T stage'] = 'None'
            
            elif data_Table[ID][6] == 'A':
            
                Feature_Table[ID]['T stage'] = 'a'
            
            else:
            
                Feature_Table[ID]['T stage'] = data_Table[ID][6]
            
            Feature_Table[ID]['Lymph-Invasion'] = data_Table[ID][9]
            
            Feature_Table[ID]['tumor'] = glob.glob( os.path.join(Vols[_index_], 
                                                                 '*.tif'))[0]

            Feature_Table[ID]['mask'] = glob.glob( os.path.join(Segs[_index_], 
                                                                '*.png'))[0]

    else:
    
        N = len(args.Volume)
    
        for _index_ in range(N):
    
            Vol = glob.glob( os.path.join(Folder_Vol, 
                                          f'{args.Volume[_index_]}*'))[0]

            Seg = glob.glob( os.path.join(Folder_Seg, 
                                          f'{args.Mask[_index_]}*'))[0]
    
            ID = os.path.basename(Vol).split('_')[0]
            
            Feature_Table[ID] = {}
            
            Feature_Table[ID]['Type'] = 'UTUC'
            Feature_Table[ID]['Sex'] = data_Table[ID][2]
            
            Grade_info = data_Table[ID][4]
            
            if ('High' in Grade_info or 
                'high' in Grade_info):
            
                Feature_Table[ID]['Histological grade'] = 'HG'
            
            elif ('Low' in Grade_info or 
                  'low' in Grade_info):
            
                Feature_Table[ID]['Histological grade'] = 'LG'
            
            else:
            
                Feature_Table[ID]['Histological grade'] = 'None'
            
            if (data_Table[ID][6] == '' or 
                data_Table[ID][6] == None):
            
                Feature_Table[ID]['T stage'] = 'None'
            
            else:
            
                Feature_Table[ID]['T stage'] = data_Table[ID][6]
            
            Feature_Table[ID]['Lymph-Invasion'] = data_Table[ID][9]
            Feature_Table[ID]['tumor'] = glob.glob( os.path.join(Vol, 
                                                                 '*.tif'))[0]

            Feature_Table[ID]['mask'] = glob.glob( os.path.join(Seg, 
                                                                '*.png'))[0]


def Extract_features(args):
    
    import matplotlib.pyplot as plt
    
    global Feature_Table
    global hyperparameters
    
    args.width = eval(args.width)
    args.level = eval(args.level)
    
    Lower_bound = (args.level - (args.width/2))

    hyperparameters['setting']['voxelArrayShift'] = Lower_bound

    extractor = featureextractor.RadiomicsFeatureExtractor(**hyperparameters)
    
    extractor.enableImageTypeByName('Wavelet', 
                                    customArgs={'level':1})
    
    extractor.enableImageTypeByName('Square')
    extractor.enableImageTypeByName('SquareRoot')
    extractor.enableImageTypeByName('Logarithm')
    extractor.enableImageTypeByName('Exponential')
    
    extractor.enableImageTypeByName('Gradient', 
                                    customArgs={'gradientUseSpacing':False})
    
    extractor.enableImageTypeByName('LBP2D', 
                                    customArgs={'lbp2Dmethod':'default', 
                                                'lbp2DRadius':3, 
                                                'lbp2DSamples':36})
    
    extractor.enableAllFeatures()
    
    for ID in Feature_Table.keys():
        
        imageFilepath = Feature_Table[ID]['tumor']
        maskFilepath = Feature_Table[ID]['mask']
        
        img = sitk.ReadImage(imageFilepath)
        
        np_img = sitk.GetArrayFromImage(img)
        
        np_img = np_img * (args.width/65535) + Lower_bound
        
        np_img = np_img.astype(np.int)
        #plt.imshow(np_img, cmap='gray') 
        #plt.show()
        
        IMG = sitk.GetImageFromArray(np_img)
        
        features = extractor.execute(IMG, 
                                     maskFilepath, 
                                     255)

        F = {}

        print(f'analyzing {ID}')
        
        F['Original'] = {}
        F['Wavelet'] = {}
        F['Square'] = {}
        F['SquareRoot'] = {}
        F['Logarithm'] = {}
        F['Exponential'] = {}
        F['Gradient'] = {}
        F['LBP2D'] = {}
        
        for key in features.keys():
        
            #print(f"Compute {key} : {features[key]}")
        
            if 'diagnostics' in key:
        
                continue
        
            if 'original' in key:
        
                F['Original'][key.split('original_')[1]] = float(features[key])
        
                continue
        
            if 'wavelet' in key:
        
                F['Wavelet'][key.split('wavelet-')[1]] = float(features[key])
        
                continue
        
            if 'square_' in key:
        
                F['Square'][key.split('square_')[1]] = float(features[key])
        
                continue
        
            if 'squareroot_' in key:
        
                F['SquareRoot'][key.split('squareroot_')[1]] = float(features[key])
        
                continue
        
            if 'logarithm_' in key:
        
                F['Logarithm'][key.split('logarithm_')[1]] = float(features[key])
        
            if 'exponential' in key:
        
                F['Exponential'][key.split('exponential_')[1]] = float(features[key])
        
                continue
        
            if 'gradient' in key:
        
                F['Gradient'][key.split('gradient_')[1]] = float(features[key])
        
                continue
        
            if 'lbp-2D_' in key:
        
                F['LBP2D'][key.split('lbp-2D_')[1]] = float(features[key])
        
                continue
        
        Feature_Table[ID]['Features'] = F


def normalization():

    NumberOfpatients = len(list(Feature_Table.keys()))
    base_ID = list(Feature_Table.keys())[0]

    F = Feature_Table[base_ID]['Features']

    buffer_list = [0.0] * NumberOfpatients

    for _filter_ in list(F.keys()):

        feature_types = list(F[_filter_].keys())

        for _feature_ in feature_types:

            _index_ = 0

            _Max_ = Feature_Table[base_ID]['Features'][_filter_][_feature_]
            _Min_ = Feature_Table[base_ID]['Features'][_filter_][_feature_]

            for ID in list(Feature_Table.keys()):

                feature_value = Feature_Table[ID]['Features'][_filter_][_feature_]
                buffer_list[_index_] = feature_value

                print(_filter_, 
                      _feature_, 
                      feature_value, 
                      _Max_, 
                      _Min_)

                if feature_value > _Max_:

                    _Max_ = feature_value

                if feature_value < _Min_:

                    _Min_ = feature_value

                _index_ += 1
            
            #Normalize to the range of [0, 1]
            offset = 0.0

            if (_Max_ - _Min_) == 0:

                continue

            scale_factor = (1.0 - 0.0)/(_Max_ - _Min_)

            _index_ = 0

            for ID in list(Feature_Table.keys()):

                Feature_Table[ID]['Features'][_filter_][_feature_] = (offset + 
                                                                      scale_factor*(buffer_list[_index_] - 
                                                                                    _Min_))

                _index_ += 1


def save_results(args):

    json_path = os.path.join(args.output_folder, 
                             'Features.txt')

    json_file = open(json_path, 'w')
    
    json_content = json.dumps(Feature_Table, 
                              indent = 4)

    json_file.writelines(json_content)
    json_file.close()

    csv_path = os.path.join(args.output_folder, 
                            'Features.csv')

    csv_file = open(csv_path, 'w')
    
    writer = csv.writer(csv_file, dialect='excel')
    
    headers = []
    
    headers.append('Subject')
    
    first_key = list(Feature_Table.keys())[0]
    inner_keys = list(Feature_Table[first_key].keys())
    
    for inner_key in inner_keys:
    
        if inner_key == 'Features':
    
            Feature_keys = list(Feature_Table[first_key][inner_key].keys())
    
            for Feature_key in Feature_keys:
    
                _features_ = list(Feature_Table[first_key][inner_key][Feature_key].keys())
    
                for _feature_ in _features_:
    
                    headers.append(f'{Feature_key}: ' + _feature_)
    
        else:
    
            headers.append(inner_key)
    
    writer.writerow(headers)
    
    _line_ = []
    
    print(f"We totally analyze {len(list(Feature_Table.keys()))} participants")
    
    for key in sorted(list(Feature_Table.keys())):
    
        _line_ = []
    
        _line_.append(key)
    
        inner_keys = list(Feature_Table[key].keys())
    
        for inner_key in inner_keys:
    
            if inner_key == 'Features':
    
                Feature_keys = list(Feature_Table[key][inner_key].keys())
    
                for Feature_key in Feature_keys:
    
                    _features_ = list(Feature_Table[first_key][inner_key][Feature_key].keys())
    
                    for _feature_ in _features_:
    
                        _line_.append(Feature_Table[key][inner_key][Feature_key][_feature_])
    
            else:
    
                _line_.append(Feature_Table[key][inner_key])
    
        writer.writerow(_line_)
    
    csv_file.close()
    
    a = zip(*csv.reader(open(csv_path, "r")))
    
    csv.writer(open(csv_path, "w")).writerows(a)


def main():

    API_description = """
***** Radiomics Analysis Platform  *****
API Name: Radiomics Feature Analysis
Version:    1.0
Developer: Alvin Li
Email:     alvinli@gorilla-technology.com / d05548014@ntu.edu.tw
****************************************

"""

    parser = argparse.ArgumentParser(prog='Feature_Extraction.py',
                                     formatter_class=argparse.RawDescriptionHelpFormatter,
                                     description=API_description)

    parser.add_argument('-input_root', 
                        action = 'store', 
                        type = str, 
                        help = 'The absolute path to input root.')

    parser.add_argument('-Table', 
                        action = 'store', 
                        type = str, 
                        help = 'The absolute path to the DATA TABLE (*.csv).')

    parser.add_argument('-Volume', 
                        nargs = '+', 
                        help = 'ex: -Volume Vol1.tif Vol2.tif ...')

    parser.add_argument('-Mask', 
                        nargs = '+', 
                        help = 'ex: -Mask Msk1.png Msk2.png ...')

    parser.add_argument('-output_folder', 
                        action = 'store', 
                        help = 'The absolute path to the output folder used to store extracted Feature Table')

    parser.add_argument('-width', 
                        action = 'store', 
                        type = str, 
                        help = 'window width')

    parser.add_argument('-level', 
                        action = 'store', 
                        type = str, 
                        help = 'window level')

    parser.add_argument('-normalize', 
                        action = 'store', 
                        type = str, 
                        help = 'True/False')
    
    args = parser.parse_args()

    assert_paser_valid(args)
    
    read_data_Table(args.Table)
    read_data(args)
    
    Extract_features(args)
    
    if args.normalize == 'True':
    
        normalization()
    
    save_results(args)


if __name__ == '__main__':

    main()
