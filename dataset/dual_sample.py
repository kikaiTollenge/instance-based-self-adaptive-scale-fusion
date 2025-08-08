import os
import sys
os.add_dll_directory('D:\\openslide-win64-20231011\\bin')
import openslide
from openslide.deepzoom import DeepZoomGenerator
from PIL import Image, ImageFilter, ImageStat
import numpy as np
import multiprocessing as mp
import time
import argparse
def find_target_level(dzg,magnification):
    high_level = None
    low_level = None
    if abs(magnification - 40.0) < 5.0:
        high_level = dzg.level_count - 3 
        low_level = dzg.level_count - 4 
    if abs(magnification - 20.0) < 5.0:
        high_level = dzg.level_count - 2 
        low_level = dzg.level_count - 3 
    
    if abs(magnification - 80.0) < 5.0:
        high_level = dzg.level_count - 4 
        low_level = dzg.level_count - 5 
    return high_level,low_level 


def get_edge(image_object,tile_size,threshold):
    edge = image_object.filter(ImageFilter.FIND_EDGES)
    edge = ImageStat.Stat(edge).sum
    edge = np.mean(edge) / tile_size**2
    # print(edge)
    if edge > threshold:
        return True
    else:
        return False

def crop_tile_by_dzg(slide_path,output_dir,tile_size,magnification_40_level,magnification_20_level,threshold,args):
    start_time = time.time()
    quality = 70
    os.makedirs(output_dir,exist_ok=True)
    try:
        slide = openslide.OpenSlide(slide_path)
        dzg = DeepZoomGenerator(slide,tile_size,overlap=0,limit_bounds=True)

        dz_level_40 = magnification_40_level
        dz_level_20 = magnification_20_level

        cols_20,rows_20 = dzg.level_tiles[dz_level_20]
        scale_factor = dzg.level_dimensions[dz_level_40][0] / dzg.level_dimensions[dz_level_20][0]
        if abs(scale_factor - 2.0) < 0.2:
            scale_factor = 2 

        for row in range(rows_20):
            for col in range(cols_20):
                tile_20 = dzg.get_tile(dz_level_20,(col,row))
                tile_20_path = os.path.join(output_dir,f'{col}_{row}.jpeg')
                if get_edge(tile_20,tile_size,threshold=threshold):
                    tile_20.convert('RGB').save(tile_20_path,quality = quality)
                    tile_40_fold = os.path.join(output_dir, f'{col}_{row}')
                else:
                    continue

                base_col_40 = col * scale_factor
                base_row_40 = row * scale_factor
                for i in range(scale_factor):
                    for j in range(scale_factor):
                        current_col_40 = int(base_col_40) + i
                        current_row_40 = int(base_row_40) + j

                        if (current_col_40 >= dzg.level_tiles[dz_level_40][0] or
                            current_row_40 >= dzg.level_tiles[dz_level_40][1]):
                            continue

                        tile_40 = dzg.get_tile(dz_level_40,(current_col_40,current_row_40))
                        tile_40_path = os.path.join(tile_40_fold,f'{current_col_40}_{current_row_40}.jpeg')
                        if get_edge(tile_40,tile_size,threshold):
                            os.makedirs(tile_40_fold, exist_ok=True)
                            tile_40.convert('RGB').save(tile_40_path,quality = quality)
    
    except Exception as e:
        print(f"process {slide_path} failed: {str(e)}")
        return slide_path+'Error'+str(e)
    finally:
        end_time = time.time()
        print(f'{slide_path} processed successfully, {end_time-start_time} seconds')
        return slide_path

def process_slide(slide_path, output_dir_root, tile_size, threshold, target_scale,args):
    try:
        slide_image = openslide.OpenSlide(slide_path)
        dzg = DeepZoomGenerator(slide_image,tile_size=tile_size,overlap=0,limit_bounds=True)
        magnification = 30.0 

        if 'aperio.AppMag' in slide_image.properties:
            magnification = float(slide_image.properties['aperio.AppMag'])
            if magnification < target_scale:
                target_scale = magnification


        elif openslide.PROPERTY_NAME_MPP_X in slide_image.properties:
            mppx = slide_image.properties.get(openslide.PROPERTY_NAME_MPP_X)
            magnification = 0.25 / float(mppx) * 40.0
            print(f"MAX manification is : {magnification}")
            if magnification < target_scale:
                target_scale = magnification     
        else:
            return slide_path+'Error'+'magnification'
        
        level_40,level_20 = find_target_level(dzg,magnification)
        if level_20 is None or level_40 is None:
            return slide_path+'Error'+'level'
        output_dir = os.path.join(output_dir_root, os.path.splitext(os.path.basename(slide_path))[0])
        return crop_tile_by_dzg(slide_path, output_dir, tile_size, level_40, level_20, threshold,args)
    except Exception as e:
        print(f"process {slide_path} failed: {str(e)}")
        return slide_path+'Error'+str(e)
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset',type=str,help='luad or lucs or c16 or colon')
    args = parser.parse_args()

    config = {
        'c16': {
            'svs_root': "E:\BaiduNetDiskDownload\CAMELYON16\\testing\images", # your wsi root
            'output_dir_root': "E:\\WSI\\C16\\test",
            'threshold': 35,
            'target_scale': 20.0 # high scale
        },
        'luad': {
            'svs_root': "E:\\BaiduNetDiskDownload\\TCGA-LUAD",
            'output_dir_root': "E:\\WSI\\TCGA-LUAD",
            'threshold': 15,
            'target_scale': 10.0
        },
        'lusc': {
            'svs_root': "E:\\BaiduNetDiskDownload\\TCGA-LUSC",
            'output_dir_root': "E:\\WSI\\TCGA-LUSC",
            'threshold': 15,
            'target_scale': 10.0
        },
        'default': {
            'svs_root': "E:\\BaiduNetDiskDownload\\hospital",
            'output_dir_root': "E:\\WSI\\hospital",
            'threshold': 9,
            'target_scale': 20.0
        }
    }

    cfg = config.get(args.dataset, config['default'])
    svs_root = cfg['svs_root']
    output_dir_root = cfg['output_dir_root']
    threshold = cfg['threshold']
    target_scale = cfg['target_scale']
    tile_size = 224
    slide_paths = []
    os.makedirs(output_dir_root, exist_ok=True)
    if args.dataset in ['luad','lusc']:
        for item in os.listdir(svs_root):
            folder = os.path.join(svs_root,item)
            for img in os.listdir(folder):
                if img.endswith('.svs' ) or img.endswith('.tif'):
                    img_path = os.path.join(folder,img)
                    slide_paths.append(img_path)
    elif args.dataset == 'c16':
        slide_paths = [os.path.join(svs_root, slide_path) for slide_path in os.listdir(svs_root) if slide_path.endswith('.tif') or slide_path.endswith('.svs')]

    processed_file = os.path.join(output_dir_root, 'processed.txt')
    processed = set()
    if os.path.exists(processed_file):
        with open(processed_file, 'r') as f:
            processed = set(line.strip() for line in f.readlines())
    remaining_paths = [p for p in slide_paths if p not in processed]
    print(f"total files: {len(slide_paths)}, remaining : {len(remaining_paths)}")
    
    def write_processed(slide_path):
        with open(processed_file, 'a') as f:
            f.write(f"{slide_path}\n")


    with mp.Pool(int(mp.cpu_count() * 0.75)) as pool:
        results = []
        for path in remaining_paths:
            res = pool.apply_async(
                process_slide,
                args=(path, output_dir_root, tile_size, threshold, target_scale, args),
                callback=write_processed
            )
            results.append(res)
        

        for res in results:
            try:
                res.get()
            except Exception as e:
                print(f"Error: {str(e)}")
