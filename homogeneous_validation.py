import argparse
import os
import glob

import json

import numpy as np

from PIL import Image
import SimpleITK as sitk

import radiomics
from radiomics import featureextractor

import matplotlib.pyplot as plt
from matplotlib.path import Path
from matplotlib.patches import PathPatch


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

        if opt.ftype == 'all':

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

        return extractor.execute(self.tumor_patch, 
                                 self.bmsk, 
                                 255)


    def save_results(self, 
                     Feature_Table, 
                     output = None):

        if opt.ftype == 'all':

            original_stat_data = {}
            wavelet_stat_data = {}
            square_stat_data = {}
            squareroot_stat_data = {}
            logarithm_stat_data = {}
            exponential_stat_data = {}
            gradient_stat_data = {}
            lbp2D_stat_data = {}

        else:

            stat_data = {}

        if output == None:

            output = opt.output


        Patch_features = {}
            
        for patch_number in Feature_Table.keys():

            Features = Feature_Table[patch_number]
            
            for key in Features.keys():

                if 'diagnostics' in key:

                    continue

                if 'original' in key:

                    if opt.ftype == 'all' and 'shape' not in key:
                        
                        Patch_features[key] = float(Features[key])

                    else:

                        if opt.ftype in key:

                            Patch_features[key] = float(Features[key])

                else:

                    if opt.ftype == 'all' and 'shape' not in key:

                        Patch_features[key] = float(Features[key])
            
            
            if opt.ftype == 'all':
                
                for radiomics_type in Patch_features.keys():
                    
                    if 'original_' in radiomics_type and 'shape' not in radiomics_type:

                        if radiomics_type not in original_stat_data:
                            original_stat_data[radiomics_type] = []
                        
                        original_stat_data[radiomics_type].append(Patch_features[radiomics_type])

                    elif 'wavelet' in radiomics_type:

                        if radiomics_type not in wavelet_stat_data:
                            wavelet_stat_data[radiomics_type] = []
                        
                        wavelet_stat_data[radiomics_type].append(Patch_features[radiomics_type])

                    elif 'square_' in radiomics_type:

                        if radiomics_type not in square_stat_data:
                            square_stat_data[radiomics_type] = []
                       
                        square_stat_data[radiomics_type].append(Patch_features[radiomics_type])

                    elif 'squareroot_' in radiomics_type:

                        if radiomics_type not in squareroot_stat_data:
                            squareroot_stat_data[radiomics_type] = []
                        
                        squareroot_stat_data[radiomics_type].append(Patch_features[radiomics_type])

                    elif 'logarithm_' in radiomics_type:

                        if radiomics_type not in logarithm_stat_data:
                            logarithm_stat_data[radiomics_type] = []
                        
                        logarithm_stat_data[radiomics_type].append(Patch_features[radiomics_type])

                    elif 'exponential' in radiomics_type:

                        if radiomics_type not in exponential_stat_data:
                            exponential_stat_data[radiomics_type] = []
                        
                        exponential_stat_data[radiomics_type].append(Patch_features[radiomics_type])

                    elif 'gradient' in radiomics_type:

                        if radiomics_type not in gradient_stat_data:
                            gradient_stat_data[radiomics_type] = []
                        
                        gradient_stat_data[radiomics_type].append(Patch_features[radiomics_type])

                    elif 'lbp-2D_' in radiomics_type:

                        if radiomics_type not in lbp2D_stat_data:
                            lbp2D_stat_data[radiomics_type] = []
                        
                        lbp2D_stat_data[radiomics_type].append(Patch_features[radiomics_type])

            else:
                
                for radiomics_type in Patch_features.keys():

                    if radiomics_type not in stat_data:

                        stat_data[radiomics_type] = []

                    stat_data[radiomics_type].append(Patch_features[radiomics_type])

        
        if len(list(Patch_features.keys())) == 0:

            return
        
        if opt.ftype == 'all':

            Patch_features.clear()

            json_path = os.path.join(output, 
                                     'original_stat_data.json')
            json_file = open(json_path, 'w')
    
            json_content = json.dumps(original_stat_data, 
                                      indent = 4)

            json_file.writelines(json_content)
            json_file.close()

            original_stat_data.clear()


            json_path = os.path.join(output, 
                                     'wavelet_stat_data.json')
            json_file = open(json_path, 'w')
    
            json_content = json.dumps(wavelet_stat_data, 
                                      indent = 4)

            json_file.writelines(json_content)
            json_file.close()

            wavelet_stat_data.clear()

            
            json_path = os.path.join(output, 
                                     'square_stat_data.json')
            json_file = open(json_path, 'w')
    
            json_content = json.dumps(square_stat_data, 
                                      indent = 4)

            json_file.writelines(json_content)
            json_file.close()

            square_stat_data.clear()


            json_path = os.path.join(output, 
                                     'squareroot_stat_data.json')
            json_file = open(json_path, 'w')
    
            json_content = json.dumps(squareroot_stat_data, 
                                      indent = 4)

            json_file.writelines(json_content)
            json_file.close()

            squareroot_stat_data.clear()


            json_path = os.path.join(output, 
                                     'logarithm_stat_data.json')
            json_file = open(json_path, 'w')
    
            json_content = json.dumps(logarithm_stat_data, 
                                      indent = 4)

            json_file.writelines(json_content)
            json_file.close()

            logarithm_stat_data.clear()

            json_path = os.path.join(output, 
                                     'exponential_stat_data.json')
            json_file = open(json_path, 'w')
    
            json_content = json.dumps(exponential_stat_data, 
                                      indent = 4)

            json_file.writelines(json_content)
            json_file.close()

            exponential_stat_data.clear()

            json_path = os.path.join(output, 
                                     'gradient_stat_data.json')
            json_file = open(json_path, 'w')
    
            json_content = json.dumps(gradient_stat_data, 
                                      indent = 4)

            json_file.writelines(json_content)
            json_file.close()

            gradient_stat_data.clear()

            json_path = os.path.join(output, 
                                     'lbp2D_stat_data.json')
            json_file = open(json_path, 'w')
    
            json_content = json.dumps(lbp2D_stat_data, 
                                      indent = 4)

            json_file.writelines(json_content)
            json_file.close()

            lbp2D_stat_data.clear()


        else:

            self.patch_num = len(Feature_Table)
            self.ftable = stat_data
                
            json_path = os.path.join(output, 
                                     'stat_data.json')

            json_file = open(json_path, 'w')
    
            json_content = json.dumps(stat_data, 
                                      indent = 4)

            json_file.writelines(json_content)
            json_file.close()
    
        
    def plt_hist(self,
                 output = None):

        radiomics_types = list(self.ftable)
        print(self.patch_num)

        if output == None:

            output = opt.output
        
        for rtype in radiomics_types:

            data = np.array(self.ftable[rtype], dtype=np.double)
            
            divider = np.max(data) - np.min(data)
            
            if divider == 0:

                divider = 1

            data = (data - np.min(data)) / divider
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
            plt.savefig(os.path.join(output,
                                     f"{rtype}.png"))
            plt.close()


def feature_analyze(analyzer,
                    tumor,
                    bmsk,
                    sliding_box,
                    output = None):

    results = {}

    t_height, t_width = tumor.shape

    print(f"Tumor image shape: {tumor.shape}")
    print(f"Tumor patch shape: {sliding_box.shape}")

    if output == None:

        output = opt.output
    
    if (t_height < sliding_box.shape[0] or 
        t_width < sliding_box.shape[1]):

        print("The tumor is too small to be separated as patches.")
        print("Done !")
    
    else:

        w_iter = range(0, t_width + 1 -  
                          sliding_box.shape[1])

        h_iter = range(0, t_height + 1 -
                          sliding_box.shape[0])
        
        
        bmsk_patch_path = os.path.join(output,
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

                    Image.fromarray(tumor_patch.astype(np.uint16)).save(os.path.join(output,
                                                                                     'tumor_patch.tif'))
                    
                    tumor_patch = tumor_patch * (opt.CT_width / 65535.0) + Lower_bound
                    
                    analyzer.get_tumor_patch(tumor_patch = sitk.GetImageFromArray(tumor_patch))
                    
                    analyzer.get_tumor_binary_mask(binary_mask = bmsk_patch_path)

                    results[count] = analyzer.feature_ext()

                    count += 1

    return results


def get_vols():

    crop_vol_folder = os.path.join(opt.tumor_folder,
                                   'crop_vol')

    vols = []
    tumor_names = os.listdir(crop_vol_folder)

    for _folder_ in tumor_names:

        local_vols = glob.glob(os.path.join(crop_vol_folder,
                               os.path.join(_folder_,
                                            '*.tif')))

        for local_vol in local_vols:

            vols.append(local_vol)

    return vols, tumor_names


def get_msks():

    crop_msk_folder = os.path.join(opt.tumor_folder,
                                   'crop_msk')

    msks = []

    for _folder_ in os.listdir(crop_msk_folder):

        local_msks = glob.glob(os.path.join(crop_msk_folder,
                               os.path.join(_folder_, 
                                            '*.png')))
        
        for local_msk in local_msks:

            msks.append(local_msk)
    
    return msks


def batch_mode_analysis():

    sliding_box = np.zeros((opt.patch_size, 
                           opt.patch_size),
                           dtype = np.double)

    vols, tumor_names = get_vols()
    msks = get_msks()

    
    for _index_ in range(len(tumor_names)):

        tumor_name = tumor_names[_index_]
        tumor_img = vols[_index_]
        tumor_msk = msks[_index_]
        
        result_folder = os.path.join(opt.output,
                                     tumor_name)

        if (os.path.isdir(result_folder)):

            os.system(f"rm -rf {result_folder}")

        os.mkdir(result_folder)

        sitk_tumor = sitk.ReadImage(tumor_img)
        np_tumor = sitk.GetArrayFromImage(sitk_tumor)

        sitk_bmsk = sitk.ReadImage(tumor_msk)
        np_bmsk = sitk.GetArrayFromImage(sitk_bmsk)

        analyzer = radiomics_analyzer(lower_bound = Lower_bound)
        
        
        results = feature_analyze(analyzer = analyzer,
                                  tumor = np_tumor,
                                  bmsk = np_bmsk,
                                  sliding_box = sliding_box,
                                  output = result_folder)

        analyzer.save_results(Feature_Table = results,
                              output = result_folder)

        if results and opt.ftype != 'all':

            analyzer.plt_hist(output = result_folder)


def homogeneous_validation_main():

    # main()

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


def get_hist_data(ftable, 
                  batch_result):

    feature_json_file = open(ftable, 'r')
    feature_json = json.load(feature_json_file)
    feature_json_file.close()
    
    hist_data = {}
    
    for _folder_ in os.listdir(batch_result):

        data_stat = os.path.join(batch_result,
                    os.path.join(_folder_, 
                                 'stat_data.json'))

        data_stat_json = open(data_stat, 'r')
        
        patient = _folder_.split("_")[0]

        grade_type = feature_json[patient]['Histological grade']
        
        if grade_type not in hist_data:

            hist_data[grade_type] = {}
        
        hist_data[grade_type][patient] = {}
        hist_data[grade_type][patient] = json.load(data_stat_json)

        data_stat_json.close()

    return hist_data


def plt2dhist():

    hist_data = get_hist_data(ftable = opt.ftable, 
                              batch_result = opt.batch_result)

    for grade_type in hist_data:

        if grade_type != "None":

            output_folder = os.path.join(opt.output, grade_type)
            if (os.path.isdir(output_folder)):

                os.system(f"rm -rf {output_folder}")

            os.mkdir(output_folder)
            
            normalized_data = {}
            
            for patient in hist_data[grade_type]:

                for feature in hist_data[grade_type][patient]:

                    if feature not in normalized_data:

                        normalized_data[feature] = {}
                        normalized_data[feature]["data"] = []
                        normalized_data[feature]["std"] = []
                        normalized_data[feature]["mean"] = []

                    fdata = np.array(hist_data[grade_type][patient][feature], dtype = np.double)
                    dsize = fdata.size

                    divider = np.max(fdata) - np.min(fdata)
            
                    if divider == 0:
                        
                        divider = 1
                        
                    fdata = (fdata - np.min(fdata)) / divider
                    fstd  = np.std(fdata, dtype = np.double)
                    fmean = np.mean(fdata, dtype = np.double)

                    normalized_data[feature]["data"].append(fdata)
                    normalized_data[feature]["std"].append(fstd)
                    normalized_data[feature]["mean"].append(fmean)

            
            for feature in normalized_data:
                
                ftitle = f"{grade_type}: {feature}"
                fig = plt.figure()

                x = np.arange(len(normalized_data[feature]["mean"]))
                y = np.array(normalized_data[feature]["mean"])
                e = np.array(normalized_data[feature]["std"])
                plt.plot(x, y, 'k-', label='mean')
                plt.fill_between(x, y-e, y+e, alpha=0.3, label='1std', color='green')
                #plt.fill_between(x, y-2*e, y+2*e, alpha=0.2, label='2std', interpolate="bicubic", cmap=plt.cm.Greens)
           
                plt.title(ftitle)
                plt.xlabel("subject")
                plt.ylabel("normalized feature value")   
                plt.xlim([np.min(x), np.max(x)])
                plt.ylim([0, 1])
                plt.grid(True)
                plt.legend()
                
                #plt.imshow(xx.reshape(yy.size,1),  cmap=plt.cm.Reds,interpolation="bicubic",
                #origin='lower',extent=[0,10,-0.0,0.40],aspect="auto", clip_path=patch, clip_on=True)
                
                #for _index_ in range(x.size):
                #    _data_ = normalized_data[feature]["data"][_index_]
                #    x_coor = np.ones(_data_.shape) * x[_index_]
                #    plt.scatter(x_coor, _data_, alpha=0.1, color='k')

                plt.savefig(os.path.join(output_folder,
                                         f"{feature}.png"))
                fig.tight_layout()
                plt.close()


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
    
    parser.add_argument('-tumor_folder', 
                        action = 'store', 
                        type = str, 
                        help = 'The folder contains tumor images and segmentation labels.')
    
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
    
    parser.add_argument('-plt_hist2d',
                        action = 'store_true',
                        default = False,
                        help = 'plot 2D histogram for batch mode results of each type of features')

    parser.add_argument('-batch_result',
                        action = 'store',
                        help = 'The folder of batch mode analzed results.')

    parser.add_argument('-ftable',
                        action = 'store',
                        help = 'Feature json file')
    
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

    if (opt.tumor_folder != None and 
        os.path.isdir(opt.tumor_folder)):

        batch_mode_analysis()
    
    if (opt.plt_hist2d):

        assert (os.path.isdir(opt.batch_result)), "The batch result folder not found."
        assert (os.path.isfile(opt.ftable)), "The feature json file not found."

        plt2dhist()
        
    else:

        homogeneous_validation_main()