import argparse
import glob
import os
import csv
import cv2

import numpy as np
from PIL import Image

import nrrd

csv_dict = {}

_pwd_ = os.getcwd()

Vols = []
Segs = []

API_description = """
***** Radiomics Analysis Platform  *****
API Name: 3D Segmentation Extractor
Version:    1.0
Developer: Alvin Li
Email:     d05548014@ntu.edu.tw
****************************************

"""

parser = argparse.ArgumentParser(prog = '3DSegExtractor.py',
                                 formatter_class = argparse.RawDescriptionHelpFormatter,
                                 description = API_description)

parser.add_argument('-LabelType', 
                    action = 'store', 
                    type = str, 
                    help = 'BB or SegMsk')

parser.add_argument('-Root', 
                    action = 'store', 
                    type = str, 
                    help = 'The absolute path to the image root folder.')

parser.add_argument('-Table', 
                    action = 'store', 
                    type = str, 
                    help = 'The absolute path to the DATA TABLE (*.csv).')

parser.add_argument('-Volume', 
                    nargs = '+', 
                    help = 'ex: -Volume Vol1.nrrd Vol2.nrrd ...')

parser.add_argument('-Mask', 
                    nargs = '+', 
                    help = 'ex: -Mask Msk1.seg.nrrd Msk2.seg.nrrd ...')

parser.add_argument('-width', 
                    action = 'store', 
                    type = str, 
                    help = 'window width')

parser.add_argument('-level', 
                    action = 'store', 
                    type = str, 
                    help = 'window level')

parser.add_argument('-output_folder', 
                    action = 'store', 
                    help = 'The absolute path to the output folder used to store extracted images')

args = parser.parse_args()


def assert_paser_valid():

    global args

    assert (args.LabelType == 'BB' or 
            args.LabelType == 'SegMsk'), "Invalid input argument. BB or SegMsk?"

    assert (os.path.exists(args.Root)), "The image root folder cannot be found"

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

    args.width = eval(args.width)
    args.level = eval(args.level)


def read_data():

    global Vols, Segs

    Folder_Vol = os.path.join(args.Root, 'Vol')
    Folder_BB = os.path.join(args.Root, 'BB')

    if args.Volume[0] == 'all':

        Vols = sorted( glob.glob( os.path.join(Folder_Vol, 
                                               'UC*.nrrd') ))

        Segs = sorted( glob.glob( os.path.join(Folder_BB, 
                                               'UC*.seg.nrrd')))
        
    else:

        N = len(args.Volume)

        for _index_ in range(N):

            Vols.append( glob.glob( f'{os.path.join(Folder_Vol, args.Volume[_index_])}*.nrrd' )[0] )
            Segs.append( glob.glob( f'{os.path.join(Folder_BB, args.Mask[_index_])}*.seg.nrrd' )[0] )


def extract_data():

    if args.LabelType == 'BB':

        print('Extract Lession region by bounding box (BB mode)')

        output_folder = args.output_folder
        
        output_vol_folder = os.path.join(output_folder, 'vol')
        output_msk_folder = os.path.join(output_folder, 'msk')
        
        output_crop_image_folder = os.path.join(output_folder, 'crop_vol')
        output_crop_msk_folder = os.path.join(output_folder, 'crop_msk')

        if os.path.exists(output_vol_folder) == False:
            
            os.mkdir(output_vol_folder)
        
        if os.path.exists(output_msk_folder) == False:
        
            os.mkdir(output_msk_folder)
        
        if os.path.exists(output_crop_image_folder) == False:
        
            os.mkdir(output_crop_image_folder)
        
        if os.path.exists(output_crop_msk_folder) == False:
        
            os.mkdir(output_crop_msk_folder)

        for _index_ in range(len(Vols)):
        
            file_name = os.path.split(Vols[_index_])[1].split('.')[0]
        
            print(f'Extract {file_name}')
        
            sub_vol_folder = os.path.join(output_vol_folder, file_name)
            sub_msk_folder = os.path.join(output_msk_folder, file_name)
        
            sub_crop_image_folder = os.path.join(output_crop_image_folder, file_name)
            sub_crop_msk_folder = os.path.join(output_crop_msk_folder, file_name)

            if os.path.exists(sub_vol_folder) == False:
        
                os.mkdir(sub_vol_folder)

            if os.path.exists(sub_msk_folder) == False:

                os.mkdir(sub_msk_folder)

            if os.path.exists(sub_crop_image_folder) == False:

                os.mkdir(sub_crop_image_folder)

            if os.path.exists(sub_crop_msk_folder) == False:

                os.mkdir(sub_crop_msk_folder)

            vol_data, vol_meta = nrrd.read(Vols[_index_]) #x, y, z
            seg_data, seg_meta = nrrd.read(Segs[_index_])

            x_size, y_size, z_size = seg_data.shape

            xyz_offsets = seg_meta['Segmentation_ReferenceImageExtentOffset'].split()

            x_offset = int(xyz_offsets[0])
            y_offset = int(xyz_offsets[1])
            z_offset = int(xyz_offsets[2])

            x_slice = x_offset + x_size
            y_slice = y_offset + y_size
            z_slice = z_offset + z_size

            #normalized to grayscale 0 - 65535
            #air HU = -1000
            #bone max HU = 3000

            Lower_bound = (args.level - (args.width/2))
            Upper_bound = (args.level + (args.width/2))

            vol_window_low = vol_data < Lower_bound
            vol_window_upp = vol_data > Upper_bound
            
            vol_data[vol_window_low] = Lower_bound
            vol_data[vol_window_upp] = Upper_bound
            
            vol_data = 65535 * (vol_data - Lower_bound) / args.width

            img_block = vol_data[x_offset : x_slice, 
                                 y_offset : y_slice, 
                                 z_offset : z_slice]

            msk = np.zeros(vol_data.shape, order='F')

            vol_xsize, vol_yslize, vol_zslice = vol_data.shape
            
            for slice_index in range(vol_zslice):
            
                vol_slice = vol_data[:, :, slice_index].transpose() 
                vol_file_path = os.path.join(sub_vol_folder, 
                                             f'{slice_index}'.zfill(5)) + '.tif'
            
                Image.fromarray(vol_slice.astype(np.uint16)).save(vol_file_path)
            
            for slice_index in range(z_size):

                vol_slice = vol_data[:, :, z_offset+slice_index]
                msk_slice = msk[:, :, z_offset+slice_index]
                seg_slice = seg_data[:, :, slice_index]

                BB_x_offset = 0
                BB_y_offset = 0

                x_coordinates = []
                y_coordinates = []
                
                for x in range(x_size):
                    for y in range(y_size):
                        if seg_slice[x, y] > 0:
                    
                            x_coordinates.append(x)
                            y_coordinates.append(y)

                if len(x_coordinates) == 0:

                    BB_x_offset = 0

                else:

                    BB_x_offset = min(x_coordinates)

                if len(y_coordinates) == 0:

                    BB_y_offset = 0

                else:

                    BB_y_offset = min(y_coordinates)

                BB_xsize = np.amax( np.count_nonzero(seg_slice, 
                                                     axis=1) )

                BB_ysize = np.amax( np.count_nonzero(seg_slice, 
                                                     axis=0) )

                if (BB_xsize == 0 and 
                    BB_ysize == 0):

                    pass

                else:

                    seg_slice = np.zeros(seg_slice.shape, order='F') 
                    
                    seg_slice[BB_x_offset : BB_x_offset+BB_xsize, 
                              BB_y_offset : BB_y_offset+BB_ysize] = 255

                    msk_slice[x_offset : x_slice, 
                              y_offset : y_slice] = seg_slice
                    
                    image_slice = img_block[:, :, slice_index]
                    image_slice = image_slice * (seg_slice / 255)
                    image_slice = image_slice.transpose()

                    seg_slice = seg_slice.transpose() 
                    msk_slice = msk_slice.transpose()
                    vol_slice = vol_slice.transpose()
                    
                    crop_image_file_path = os.path.join(sub_crop_image_folder, 
                                                        f'{slice_index}'.zfill(5)) + '.tif'

                    crop_seg_file_path = os.path.join(sub_crop_msk_folder, 
                                                      f'{slice_index}'.zfill(5)) + '.png'

                    msk_file_path = os.path.join(sub_msk_folder, 
                                                 f'{slice_index}'.zfill(5)) + '.png'

                    vol_file_path = os.path.join(sub_msk_folder, 
                                                 f'{slice_index}'.zfill(5)) + '.tif'
                    
                    Image.fromarray(image_slice.astype(np.uint16)).save(crop_image_file_path)
                    Image.fromarray(seg_slice.astype(np.uint8)).save(crop_seg_file_path)
                    
                    Image.fromarray(msk_slice.astype(np.uint8)).save(msk_file_path)
                    Image.fromarray(vol_slice.astype(np.uint16)).save(vol_file_path)

    if args.LabelType == 'SegMsk':

        print('Extract Tumor by segmentation (SegMsk mode)')
        
        output_folder = args.output_folder
        
        output_vol_folder = os.path.join(output_folder, 'vol')
        output_msk_folder = os.path.join(output_folder, 'msk')
        
        output_crop_image_folder = os.path.join(output_folder, 'crop_vol')
        output_crop_msk_folder = os.path.join(output_folder, 'crop_msk')

        
        if os.path.exists(output_vol_folder) == False:
        
            os.mkdir(output_vol_folder)
        
        if os.path.exists(output_msk_folder) == False:
        
            os.mkdir(output_msk_folder)
        
        if os.path.exists(output_crop_image_folder) == False:
        
            os.mkdir(output_crop_image_folder)
        
        if os.path.exists(output_crop_msk_folder) == False:
        
            os.mkdir(output_crop_msk_folder)

        for _index_ in range(len(Vols)):

            file_name = os.path.split(Vols[_index_])[1].split('.')[0]

            print(f'Extract {file_name}')

            sub_vol_folder = os.path.join(output_vol_folder, file_name)
            sub_msk_folder = os.path.join(output_msk_folder, file_name)

            sub_crop_image_folder = os.path.join(output_crop_image_folder, file_name)
            sub_crop_msk_folder = os.path.join(output_crop_msk_folder, file_name)


            if os.path.exists(sub_vol_folder) == False:

                os.mkdir(sub_vol_folder)

            if os.path.exists(sub_msk_folder) == False:

                os.mkdir(sub_msk_folder)

            if os.path.exists(sub_crop_image_folder) == False:

                os.mkdir(sub_crop_image_folder)

            if os.path.exists(sub_crop_msk_folder) == False:

                os.mkdir(sub_crop_msk_folder)

            vol_data, vol_meta = nrrd.read(Vols[_index_]) #x, y, z
            seg_data, seg_meta = nrrd.read(Segs[_index_])

            x_size, y_size, z_size = seg_data.shape

            xyz_offsets = seg_meta['Segmentation_ReferenceImageExtentOffset'].split()

            x_offset = int(xyz_offsets[0])
            y_offset = int(xyz_offsets[1])
            z_offset = int(xyz_offsets[2])

            x_slice = x_offset + x_size
            y_slice = y_offset + y_size
            z_slice = z_offset + z_size

            #normalized to grayscale 0 - 65535
            #air HU = -1000
            #bone max HU = 3000

            Lower_bound = (args.level - (args.width/2))
            Upper_bound = (args.level + (args.width/2))

            vol_window_low = vol_data < Lower_bound
            vol_window_upp = vol_data > Upper_bound

            vol_data[vol_window_low] = Lower_bound
            vol_data[vol_window_upp] = Upper_bound

            vol_data = 65535 * (vol_data - Lower_bound) / args.width

            img_block = vol_data[x_offset : x_slice, 
                                 y_offset : y_slice, 
                                z_offset : z_slice]

            msk = np.zeros(vol_data.shape, order='F')

            vol_xsize, vol_yslize, vol_zslice = vol_data.shape
            
            for slice_index in range(vol_zslice):
            
                vol_slice = vol_data[:, :, slice_index].transpose()
                vol_file_path = os.path.join(sub_vol_folder, 
                                             f'{slice_index}'.zfill(5)) + '.tif'

                Image.fromarray(vol_slice.astype(np.uint16)).save(vol_file_path)

            for slice_index in range(z_size):

                vol_slice = vol_data[:, :, z_offset+slice_index]
                msk_slice = msk[:, :, z_offset+slice_index]
                seg_slice = seg_data[:, :, slice_index] * 255

                mask = np.zeros((x_size+4, y_size+4), np.uint8)

                im_floodfill = np.zeros((x_size+2, y_size+2), np.uint8)
                im_floodfill[1 : 1+x_size, 
                             1 : 1+y_size] = seg_slice
                
                cv2.floodFill(im_floodfill, mask, (0,0), 255)
                
                im_floodfill_inv = cv2.bitwise_not(im_floodfill)
                
                defects = im_floodfill_inv[1 : 1+x_size, 
                                           1 : 1+y_size] > 0
                
                seg_slice[defects] = 255

                msk_slice[x_offset : x_slice, 
                          y_offset : y_slice] = seg_slice

                image_slice = img_block[:, :, slice_index]
                image_slice = image_slice * (seg_slice / 255)
                image_slice = image_slice.transpose()

                seg_slice = seg_slice.transpose()
                msk_slice = msk_slice.transpose()
                vol_slice = vol_slice.transpose()

                crop_image_file_path = os.path.join(sub_crop_image_folder, 
                                                    f'{slice_index}'.zfill(5)) + '.tif'

                crop_seg_file_path = os.path.join(sub_crop_msk_folder, 
                                                  f'{slice_index}'.zfill(5)) + '.png'

                msk_file_path = os.path.join(sub_msk_folder, 
                                             f'{slice_index}'.zfill(5)) + '.png'

                vol_file_path = os.path.join(sub_msk_folder, 
                                             f'{slice_index}'.zfill(5)) + '.tif'

                Image.fromarray(image_slice.astype(np.uint16)).save(crop_image_file_path)
                Image.fromarray(seg_slice.astype(np.uint8)).save(crop_seg_file_path)
                
                Image.fromarray(msk_slice.astype(np.uint8)).save(msk_file_path)
                Image.fromarray(vol_slice.astype(np.uint16)).save(vol_file_path)


if __name__ == '__main__':

    assert_paser_valid()
    read_data()
    extract_data()

