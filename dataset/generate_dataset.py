"""
Creates an augmented version of the Labeled Faces in the Wild dataset.
Run with:
    python generate_dataset.py --path="/foo/bar/lfw"
"""
from __future__ import print_function, division
import os
import random
import re
import numpy as np
from scipy import misc
from ImageAugmenter import create_aug_matrices
from skimage import transform as tf
import argparse

random.seed(43)
np.random.seed(43)

# specs from http://conradsanderson.id.au/lfwcrop/
CROP_UPPER_LEFT_CORNER_X = 83
CROP_UPPER_LEFT_CORNER_Y = 92
CROP_LOWER_RIGHT_CORNER_X = 166
CROP_LOWER_RIGHT_CORNER_Y = 175

WRITE_AUG = True
WRITE_UNAUG = True
WRITE_AUG_TO = "out_aug_64x64"
WRITE_UNAUG_TO = "out_unaug_64x64"
SCALE = 64
AUGMENTATIONS = 19

def main():
    """Main method that reads the images, augments and saves them."""
    parser = argparse.ArgumentParser(description="Create augmented version of LFW.")
    parser.add_argument("--path", required=True, help="Path to your LFW dataset directory")
    args = parser.parse_args()
    
    ds = Dataset([args.path])
    print("Found %d images total." % (len(ds.fps),))
    
    for img_idx, image in enumerate(ds.get_images()):
        print("Image %d..." % (img_idx,))
        augmentations = augment(image, n=AUGMENTATIONS, hflip=True, vflip=False,
                                #scale_to_percent=(0.85, 1.1), scale_axis_equally=True,
                                scale_to_percent=(0.82, 1.10), scale_axis_equally=True,
                                rotation_deg=8, shear_deg=0,
                                translation_x_px=5, translation_y_px=5,
                                brightness_change=0.1, noise_mean=0.0, noise_std=0.00)
        faces = [image]
        faces.extend(augmentations)
        
        for aug_idx, face in enumerate(faces):
            crop = face[CROP_UPPER_LEFT_CORNER_Y:CROP_LOWER_RIGHT_CORNER_Y+1,
                        CROP_UPPER_LEFT_CORNER_X:CROP_LOWER_RIGHT_CORNER_X+1,
                        ...]
            
            #misc.imshow(face)
            #misc.imshow(crop)
            
            filename = "{:0>6}_{:0>3}.jpg".format(img_idx, aug_idx)
            if WRITE_UNAUG and aug_idx == 0:
                face_scaled = misc.imresize(crop, (SCALE, SCALE))
                misc.imsave(os.path.join(WRITE_UNAUG_TO, filename), face_scaled)
            if WRITE_AUG:
                face_scaled = misc.imresize(crop, (SCALE, SCALE))
                misc.imsave(os.path.join(WRITE_AUG_TO, filename), face_scaled)

    print("Finished.")

def augment(image, n,
            hflip=False, vflip=False, scale_to_percent=1.0, scale_axis_equally=True,
            rotation_deg=0, shear_deg=0, translation_x_px=0, translation_y_px=0,
            brightness_change=0.0, noise_mean=0.0, noise_std=0.0):
    """Augment an image n times.
    Args:
            n                   Number of augmentations to generate.
            hflip               Allow horizontal flipping (yes/no).
            vflip               Allow vertical flipping (yes/no)
            scale_to_percent    How much scaling/zooming to allow. Values are around 1.0.
                                E.g. 1.1 is -10% to +10%
                                E.g. (0.7, 1.05) is -30% to 5%.
            scale_axis_equally  Whether to enforce equal scaling of x and y axis.
            rotation_deg        How much rotation to allow. E.g. 5 is -5 degrees to +5 degrees.
            shear_deg           How much shearing to allow.
            translation_x_px    How many pixels of translation along the x axis to allow.
            translation_y_px    How many pixels of translation along the y axis to allow.
            brightness_change   How much change in brightness to allow. Values are around 0.0.
                                E.g. 0.2 is -20% to +20%.
            noise_mean          Mean value of gaussian noise to add.
            noise_std           Standard deviation of gaussian noise to add.
    Returns:
        List of numpy arrays
    """
    assert n >= 0
    result = []
    if n == 0:
        return result
    
    width = image.shape[0]
    height = image.shape[1]
    matrices = create_aug_matrices(n, img_width_px=width, img_height_px=height,
                                   scale_to_percent=scale_to_percent,
                                   scale_axis_equally=scale_axis_equally,
                                   rotation_deg=rotation_deg,
                                   shear_deg=shear_deg,
                                   translation_x_px=translation_x_px,
                                   translation_y_px=translation_y_px)
    for i in range(n):
        img = np.copy(image)
        matrix = matrices[i]
        
        # random horizontal / vertical flip
        if hflip and random.random() > 0.5:
            img = np.fliplr(img)
        if vflip and random.random() > 0.5:
            img = np.flipud(img)
        
        # random brightness adjustment
        by_percent = random.uniform(1.0 - brightness_change, 1.0 + brightness_change)
        img = img * by_percent
        
        # gaussian noise
        # numpy requires a std above 0
        if noise_std > 0:
            img = img + (255 * np.random.normal(noise_mean, noise_std, (img.shape)))
        
        # clip to 0-255
        img = np.clip(img, 0, 255).astype(np.uint8)
        
        arr = tf.warp(img, matrix, mode="nearest") # projects to float 0-1
        img = np.array(arr * 255, dtype=np.uint8)
        result.append(img)
        
    return result

class Dataset(object):
    """Helper class to handle the loading of the LFW dataset dataset."""
    def __init__(self, dirs):
        """Instantiate a dataset object.
        Args:
            dirs    List of filepaths to directories. Direct subdirectories will be read.
        """
        self.dirs = dirs
        self.fps = self.get_filepaths(self.get_direct_subdirectories(dirs))
    
    def get_direct_subdirectories(self, dirs):
        """Find all direct subdirectories of a list of directories.
        Args:
            dirs    List of directories to search in.
        Returns:
            Set of paths to directories
        """
        result = []
        result.extend(dirs)
        for fp_dir in dirs:
            subdirs = [name for name in os.listdir(fp_dir) if os.path.isdir(os.path.join(fp_dir, name))]
            subdirs = [os.path.join(fp_dir, name) for name in subdirs]
            result.extend(subdirs)
        return set(result)
    
    def get_filepaths(self, dirs):
        """Find all jpg-images in provided filepaths.
        Args:
            dirs    List of paths to directories to search in.
        Returns:
            List of filepaths
        """
        result = []
        for fp_dir in dirs:
            fps = [f for f in os.listdir(fp_dir) if os.path.isfile(os.path.join(fp_dir, f))]
            fps = [os.path.join(fp_dir, f) for f in fps]
            fps_img = [fp for fp in fps if re.match(r".*\.jpg$", fp)]
            if len(fps) != len(fps_img):
                print("[Warning] directory '%s' contained %d files with extension differing from 'jpg'" % (fp_dir, len(fps)-len(fps_img)))
            result.extend(fps_img)
        if len(result) < 1:
            print("[Warning] [Dataset] No images of extension *.ppm found in given directories.")
        return result
    
    def get_images(self, start_at=None, count=None):
        """Returns a generator of images.
        Args:
            start_at    Index of first image to return or None.
            count       Maximum number of images to return or None.
        Returns:
            Generator of images (numpy arrays).
        """
        start_at = 0 if start_at is None else start_at
        end_at = len(self.fps) if count is None else start_at+count
        for fp in self.fps[start_at:end_at]:
            image = misc.imread(fp)
            yield image

if __name__ == "__main__":
    main()
