import argparse
import os

import json

import numpy as np

from PIL import Image
import SimpleITK as sitk

import radiomics
from radiomics import featureextractor

import matplotlib.pyplot as plt


class radiomics_analyzer():

    def __init__(self, 
                 lower_bound):

        hyperparameters = {}

        hyperparameters['setting'] = {}
        
        hyperparameters['force2D'] = True
        hyperparameters['force2Ddimension'] = 0

        hyperparameters['setting']['voxelArrayShift'] = lower_bound

        self.hyp = hyperparameters


    def get_tumor_patch(self, tumor_patch):

        self.tumor_patch = tumor_patch


    def get_tumor_binary_mask(self, binary_mask):

        self.bmsk = binary_mask

    
    def feature_ext(self):

        extractor = featureextractor.RadiomicsFeatureExtractor(**self.hyp)

        extractor.enableAllFeatures()

        return extractor.execute(self.tumor_patch, 
                                 self.bmsk, 
                                 255)


    def save_results(self, Feature_Table):

        stat_data = {}

        for patch_number in Feature_Table.keys():

            Features = Feature_Table[patch_number]
            Patch_features = {}

            for key in Features.keys():

                if 'diagnostics' in key:

                    continue

                if 'original' in key:

                    if opt.ftype == 'all':
                        
                        Patch_features[key] = float(Features[key])

                    else:

                        if opt.ftype in key:

                            Patch_features[key] = float(Features[key])
            
            for radiomics_type in Patch_features.keys():

                if radiomics_type not in stat_data:

                    stat_data[radiomics_type] = []

                stat_data[radiomics_type].append(Patch_features[radiomics_type])


        self.patch_num = len(Feature_Table)
        self.ftable = stat_data
                
        json_path = os.path.join(opt.output, 
                                 'stat_data.json')

        json_file = open(json_path, 'w')
    
        json_content = json.dumps(stat_data, 
                                  indent = 4)

        json_file.writelines(json_content)
        json_file.close()
    
        
    def plt_hist(self):

        radiomics_types = list(self.ftable)
        print(self.patch_num)

        for rtype in radiomics_types:

            data = np.array(self.ftable[rtype], dtype=np.double)
            min = np.min(data)
            max = np.max(data)
            data = (data - min) / (max - min)
            fig = plt.figure()
            plt.hist(data,  
                     bins=40,
                     edgecolor='none',
                     histtype='stepfilled',
                     density=True)

            plt.title(rtype)
            plt.xlabel("normalized feature value")
            plt.ylabel("density (bins = 40)")
            fig.tight_layout()
            plt.savefig(os.path.join(opt.output,
                                     f"{rtype}.png"))
            plt.close()


def feature_analyze(analyzer,
                    tumor,
                    bmsk,
                    sliding_box):

    results = {}

    t_height, t_width = tumor.shape

    print(f"Tumor image shape: {tumor.shape}")
    print(f"Tumor patch shape: {sliding_box.shape}")

    if (t_height < sliding_box.shape[0] or 
        t_width < sliding_box.shape[1]):

        print("The tumor is too small to be separated as patches.")
        print("Done !")
    
    else:

        w_iter = range(0, t_width + 1 -  
                          sliding_box.shape[1])

        h_iter = range(0, t_height + 1 -
                          sliding_box.shape[0])
        
        
        bmsk_patch_path = os.path.join(opt.output,
                                       'bmsk.png')

        Image.fromarray(255 * np.ones(sliding_box.shape,
                                      dtype=np.uint8)).save(bmsk_patch_path)
        
        count = 0
        
        for x in w_iter:

            for y in h_iter:

                bmsk_patch = np.copy(bmsk[y : y+sliding_box.shape[0],
                                          x : x+sliding_box.shape[1]])
                
                if (np.count_nonzero(bmsk_patch) == bmsk_patch.size):

                    tumor_patch = np.copy(tumor[y : y+sliding_box.shape[0],
                                                x : x+sliding_box.shape[1]])

                    Image.fromarray(tumor_patch.astype(np.uint16)).save(os.path.join(opt.output,
                                                                                     'tumor_patch.tif'))
                    
                    tumor_patch = tumor_patch * (opt.CT_width / 65535.0) + Lower_bound
                    
                    analyzer.get_tumor_patch(tumor_patch = sitk.GetImageFromArray(tumor_patch))
                    
                    analyzer.get_tumor_binary_mask(binary_mask = bmsk_patch_path)

                    results[count] = analyzer.feature_ext()

                    count += 1

    return results


def homogeneous_validation_main():

    opt.patch_size

    sliding_box = np.zeros((opt.patch_size, 
                           opt.patch_size),
                           dtype = np.double)

    print(f"Loading image: {opt.tumor}")

    assert (os.path.isfile(opt.tumor)), "The tumor image not found"

    sitk_tumor = sitk.ReadImage(opt.tumor)
    np_tumor = sitk.GetArrayFromImage(sitk_tumor)

    sitk_bmsk = sitk.ReadImage(opt.binary_msk)
    np_bmsk = sitk.GetArrayFromImage(sitk_bmsk)

    analyzer = radiomics_analyzer(lower_bound = Lower_bound)

    results = feature_analyze(analyzer = analyzer,
                              tumor = np_tumor,
                              bmsk = np_bmsk,
                              sliding_box = sliding_box)

    analyzer.save_results(Feature_Table = results)

    if results:

        analyzer.plt_hist()


if __name__ == '__main__':

    API_description = """
***** Radiomics Analysis Platform  *****
API Name: Homogeneity Analysis
Version:    1.0
Developer: Alvin Li
Email:     d05548014@ntu.edu.tw
****************************************

"""
    
    _pwd_ = os.getcwd()

    parser = argparse.ArgumentParser(prog='homogeneous_validation.py',
                                     formatter_class=argparse.RawDescriptionHelpFormatter,
                                     description=API_description)
    
    parser.add_argument('-tumor', 
                        action = 'store', 
                        type = str, 
                        help = 'The path of a segmented tunmor image.')

    parser.add_argument('-binary_msk', 
                        action = 'store', 
                        type = str, 
                        help = 'The path of the binary mask for the tumor.')

    parser.add_argument('-CT_level', 
                        action = 'store', 
                        type = float,
                        default = 40.0,
                        help = 'CT level (HU).')

    parser.add_argument('-CT_width', 
                        action = 'store', 
                        type = float,
                        default = 350.0,
                        help = 'CT width (HU).')
    
    parser.add_argument('-patch_size', 
                        action = 'store', 
                        type = int,
                        default = 9,
                        help = 'for sliding window size, e.g. 7 x 7, 9 x 9.')

    parser.add_argument('-ftype',
                        action = 'store',
                        type = str,
                        default = 'all',
                        help= 'all: all features, else, firstorder, glcm, ...')
    
    parser.add_argument('-output', 
                        action = 'store', 
                        type = str,
                        default = os.path.join(_pwd_, 
                                               'output'),
                        help = 'The path for output folder')
    
    opt = parser.parse_args()

    if (os.path.exists(opt.output)):

        os.system(f"rm -rf {opt.output}")

    os.mkdir(opt.output)

    Lower_bound = (opt.CT_level - (opt.CT_width/2))

    homogeneous_validation_main()