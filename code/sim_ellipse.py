#%% Import libraries

import sympy
import contextlib

from skimage import draw
import random
from random import randint
import matplotlib.pyplot as plt
from sympy import geometry

import os
import pathlib
import shutil
import sys
from datetime import datetime

import numpy as np
import pandas as pd
import requests
from tqdm import tqdm

import joblib

from joblib import Parallel, delayed

#%% Import functions

@contextlib.contextmanager
def tqdm_joblib(tqdm_object):
    """Context manager to patch joblib to report into tqdm progress bar given as argument"""
    class TqdmBatchCompletionCallback(joblib.parallel.BatchCompletionCallBack):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)

        def __call__(self, *args, **kwargs):
            tqdm_object.update(n=self.batch_size)
            return super().__call__(*args, **kwargs)

    old_batch_callback = joblib.parallel.BatchCompletionCallBack
    joblib.parallel.BatchCompletionCallBack = TqdmBatchCompletionCallback
    try:
        yield tqdm_object
    finally:
        joblib.parallel.BatchCompletionCallBack = old_batch_callback
        tqdm_object.close()


def make_subdirectory(directory, append_name=""):
    """Makes subdirectories.
    Parameters
    ----------
    directory : str or pathlib object
        A string with the path of directory where subdirectories should be created.
    append_name : str
        A string to be appended to the directory path (name of the subdirectory created).
    Returns
    -------
    pathlib object
        A pathlib object for the subdirectory created.
    """
    
    # Define the path of the directory within which this function will make a subdirectory.
    directory = pathlib.Path(directory)
    # The name of the subdirectory.
    append_name = str(append_name)
    # Define the output path by the initial directory and join (i.e. "+") the appropriate text.
    output_path = pathlib.Path(directory).joinpath(str(append_name))
    
    # Use pathlib to see if the output path exists, if it is there it returns True
    if pathlib.Path(output_path).exists() == False:
        
        # Prints a status method to the console using the format option, which fills in the {} with whatever
        # is in the ().
        print(
            "This output path doesn't exist:\n            {} \n Creating...".format(
                output_path))
        
        # Use pathlib to create the folder.
        pathlib.Path.mkdir(output_path, parents=True, exist_ok=True)
        
        # Prints a status to let you know that the folder has been created
        print("Output path has been created")
    
    # Since it's a boolean return, and True is the only other option we will simply print the output.
    else:
        # This will print exactly what you tell it, including the space. The backslash n means new line.
        print("Output path already exists:\n               {}".format(
                output_path))
    return output_path


def sim_ellipse(output_directory, im_width_px, im_height_px, min_diam_um, max_diam_um, px_per_um, angle_deg):
    # conversions
    um_per_inch = 25400
    dpi = int(px_per_um * um_per_inch)
    min_rad_um = min_diam_um / 2
    max_rad_um = max_diam_um / 2
    
    # image size in inches
    im_width_inch = (im_width_px / px_per_um) / um_per_inch
    im_height_inch = (im_height_px / px_per_um) / um_per_inch
    
    imsize_inch = im_height_inch, im_width_inch
    imsize_px = im_height_px, im_width_px
    
    min_rad_px = min_rad_um * px_per_um
    max_rad_px = max_rad_um * px_per_um
    
    # generate array of ones (will show up as white background)
    img = np.ones(imsize_px, dtype=np.uint8)
    
    # generate ellipse in center of image
    rr, cc = draw.ellipse(im_height_px / 2, im_width_px / 2, min_rad_px, max_rad_px, shape=img.shape,
                          rotation=np.deg2rad(angle_deg))
    img[rr, cc] = 0
    
    fig = plt.figure(frameon=False)
    fig.set_size_inches(im_width_inch, im_height_inch)
    ax = plt.Axes(fig, [0, 0, 1, 1])
    ax.set_axis_off()
    fig.add_axes(ax)
    
    p1 = geometry.Point((im_height_px / px_per_um) / 2, (im_width_px / px_per_um) / 2)
    e1 = geometry.Ellipse(p1, hradius=max_rad_um, vradius=min_rad_um)
    area = sympy.N(e1.area)
    eccentricity = e1.eccentricity
    ax.imshow(img, cmap="gray", aspect='auto')
    
    jetzt = datetime.now()
    timestamp = jetzt.strftime("%b%d_%H%M_%S_%f")
    
    name = "sim_ellipse_" + str(timestamp)
    
    im_path = pathlib.Path(output_directory).joinpath(name + ".tiff")
    df_path = pathlib.Path(output_directory).joinpath(name + ".csv")

    data = {'ID': [name], 'area': [area], 'eccentricity': [eccentricity], 'ref_min_diam': [min_diam_um],
            'ref_max_diam': [max_diam_um]}

    df = pd.DataFrame(data)
    
    df.to_csv(df_path)
    
    plt.ioff()
    fig.savefig(fname=im_path, dpi=dpi)
    plt.cla()
    plt.close()
    
    return df


def validation_section(output_location, repeats, jobs=2):
    
    jetzt = datetime.now()
    timestamp = jetzt.strftime("%b%d_%H%M_")
    testname = str(timestamp + "ValidationTest_Section")

    main_output_path = make_subdirectory(output_location, append_name=testname)

    dummy_dir = make_subdirectory(main_output_path, append_name="ValidationData")
    
    # create list of random variables from range
    def gen_ellipse_data():
        min_diam_um = random.uniform(30, 120)
        ecc = random.uniform(0.0, 1.0)
        # min_diam_um = random.uniform(30, max_diam_um)
        max_diam_um = geometry.Ellipse(geometry.Point(0,0), vradius=min_diam_um, eccentricity=ecc).hradius
        angle_deg = random.randint(0, 360)
        list = [max_diam_um, min_diam_um, angle_deg]
        return list
    
    tempdf = [gen_ellipse_data() for i in range(repeats)]
    
    gen_ellipse_df = pd.DataFrame(tempdf, columns=['max_diam_um', 'min_diam_um', 'angle_deg'])
    
    df_list = []
    # for index, row in tqdm(gen_ellipse_df.iterrows(), desc="Generating ellipses", position=0, unit="datasets", leave=True):
    #     df = sim_ellipse(dummy_dir, 5200, 3900, row['min_diam_um'], row['max_diam_um'], 4.25, row['angle_deg'])
    #     df_list.append(df)

    with tqdm_joblib(tqdm(desc="Generating ellipses", position=0, unit="datasets", leave=True, total=len(gen_ellipse_df), miniters=1)) as progress_bar:
        progress_bar.monitor_interval = 1
        df_list = Parallel(n_jobs=jobs, verbose=0)(delayed(sim_ellipse)(dummy_dir, 5200, 3900, row['min_diam_um'], row['max_diam_um'], 4.25, row['angle_deg']) for index, row in gen_ellipse_df.iterrows())
    
    sim_ellipse_sum_df = pd.concat(df_list)
    sim_ellipse_sum_df.set_index('ID', inplace=True)

    with pathlib.Path(main_output_path).joinpath("summary_" + testname + ".csv") as savename:
        sim_ellipse_sum_df.to_csv(savename)
    
    return main_output_path


#%% Run command
# Run with the following command in Python console:
output_location = pathlib.Path("[insert output path]")
validation_section(output_location, repeats=10, jobs=2)

# Consider if the number of jobs is appropriate for your computer's CPU.
# May run slowly due to the large images being produced

