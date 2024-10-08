'''
Define a function to split data in a given dir for fitting a model, uncertainty quantification, and calibration when the data is not split yet
'''

import numpy as np
import pandas as pd
import os

def split_data_for_PTC_DRO(dir_name,train_ratio,UQ_ratio,max_cal_num):
    train_sample_dir = dir_name
    has_split = False
    # judge if the "LUQ" dir exists, if not, create it
    if not os.path.exists(train_sample_dir+"PTC_DRO/"):
        os.makedirs(train_sample_dir+"PTC_DRO/")
    else:
        has_split = True
    LUQ_dir = train_sample_dir+"PTC_DRO/"
    if not has_split:
        # load the data in train_sample_dir
        covs = pd.read_csv(train_sample_dir+"covs.csv",header=None).to_numpy()

        true_f = pd.read_csv(train_sample_dir+"true_f.csv",header=None).to_numpy()

        half_width = pd.read_csv(train_sample_dir+"half_width.csv",header=None).to_numpy()

        c = pd.read_csv(train_sample_dir+"c.csv",header=None).to_numpy()

        #split the data into three parts, the first part with 3/5 samples it used to fit a model , the second part with 1/5 for uncertainty quantification, and the third part with 1/5 for calibration
        num_samples = covs.shape[0]
        # if the number of samples for UQ is larger than max_UQ_num, then set the number of samples for UQ to max_UQ_num, and the number of samples for fitting to num_samples - max_UQ_num
        if num_samples*(1-train_ratio-UQ_ratio) > max_cal_num:
            num_samples_cal = max_cal_num
            num_samples_UQ = int(num_samples*UQ_ratio)
            num_samples_fit = num_samples - num_samples_UQ - num_samples_cal
        else:
            num_samples_fit = int(num_samples*train_ratio)
            num_samples_UQ = int(num_samples*UQ_ratio)
            num_samples_cal = num_samples - num_samples_UQ - num_samples_fit
        

        # save the data into the "fit" dir, a sub dir of LUQ. If it does not exist, create it
        if not os.path.exists(LUQ_dir+"fit/"):
            os.makedirs(LUQ_dir+"fit/")
        fit_dir = LUQ_dir+"fit/"
        pd.DataFrame(covs[0:num_samples_fit,:]).to_csv(fit_dir+"covs_fit.csv",header=False,index=False)
        pd.DataFrame(true_f[0:num_samples_fit,:]).to_csv(fit_dir+"true_f_fit.csv",header=False,index=False)
        pd.DataFrame(half_width[0:num_samples_fit,:]).to_csv(fit_dir+"half_width_fit.csv",header=False,index=False)
        pd.DataFrame(c[0:num_samples_fit,:]).to_csv(fit_dir+"c_fit.csv",header=False,index=False)

        # save the UQ data to the UQ dir, if it does not exist, create it
        if not os.path.exists(LUQ_dir+"UQ/"):
            os.makedirs(LUQ_dir+"UQ/")
        UQ_dir = LUQ_dir+"UQ/"
        pd.DataFrame(covs[num_samples_fit:num_samples_fit+num_samples_UQ,:]).to_csv(UQ_dir+"covs_UQ.csv",header=False,index=False)
        pd.DataFrame(true_f[num_samples_fit:num_samples_fit+num_samples_UQ,:]).to_csv(UQ_dir+"true_f_UQ.csv",header=False,index=False)
        pd.DataFrame(half_width[num_samples_fit:num_samples_fit+num_samples_UQ,:]).to_csv(UQ_dir+"half_width_UQ.csv",header=False,index=False)
        pd.DataFrame(c[num_samples_fit:num_samples_fit+num_samples_UQ,:]).to_csv(UQ_dir+"c_UQ.csv",header=False,index=False)

        # save the cal data to the cal dir, if it does not exist, create it
        if not os.path.exists(LUQ_dir+"cal/"):
            os.makedirs(LUQ_dir+"cal/")
        cal_dir = LUQ_dir+"cal/"
        pd.DataFrame(covs[num_samples_fit+num_samples_UQ:num_samples_fit+num_samples_UQ+num_samples_cal,:]).to_csv(cal_dir+"covs_cal.csv",header=False,index=False)
        pd.DataFrame(true_f[num_samples_fit+num_samples_UQ:num_samples_fit+num_samples_UQ+num_samples_cal,:]).to_csv(cal_dir+"true_f_cal.csv",header=False,index=False)
        pd.DataFrame(half_width[num_samples_fit+num_samples_UQ:num_samples_fit+num_samples_UQ+num_samples_cal,:]).to_csv(cal_dir+"half_width_cal.csv",header=False,index=False)
        pd.DataFrame(c[num_samples_fit+num_samples_UQ:num_samples_fit+num_samples_UQ+num_samples_cal,:]).to_csv(cal_dir+"c_cal.csv",header=False,index=False)
        

def split_data_for_LUQ(dir_name):
    train_sample_dir = dir_name
    has_split = False
    # judge if the "LUQ" dir exists, if not, create it
    if not os.path.exists(train_sample_dir+"LUQ/"):
        os.makedirs(train_sample_dir+"LUQ/")

    if os.path.exists(train_sample_dir+"LUQ/fit/covs_fit.csv"):
        has_split = True

    LUQ_dir = train_sample_dir+"LUQ/"
    if not has_split:
        # load the data in train_sample_dir
        covs = pd.read_csv(train_sample_dir+"covs.csv",header=None).to_numpy()

        c = pd.read_csv(train_sample_dir+"c.csv",header=None).to_numpy()

        #split the data into three parts, the first part with 3/5 samples it used to fit a model , the second part with 1/5 for uncertainty quantification, and the third part with 1/5 for calibration
        num_samples = covs.shape[0]
        num_samples_fit = int(num_samples*3/5)
        num_samples_UQ = int(num_samples*1/5)
        num_samples_cal = num_samples-num_samples_fit-num_samples_UQ

        # save the data into the "fit" dir, a sub dir of LUQ. If it does not exist, create it
        if not os.path.exists(LUQ_dir+"fit/"):
            os.makedirs(LUQ_dir+"fit/")
        fit_dir = LUQ_dir+"fit/"
        pd.DataFrame(covs[0:num_samples_fit,:]).to_csv(fit_dir+"covs_fit.csv",header=False,index=False)
        pd.DataFrame(c[0:num_samples_fit,:]).to_csv(fit_dir+"c_fit.csv",header=False,index=False)


        # save the UQ data to the UQ dir, if it does not exist, create it
        if not os.path.exists(LUQ_dir+"UQ/"):
            os.makedirs(LUQ_dir+"UQ/")
        UQ_dir = LUQ_dir+"UQ/"
        pd.DataFrame(covs[num_samples_fit:num_samples_fit+num_samples_UQ,:]).to_csv(UQ_dir+"covs_UQ.csv",header=False,index=False)
        
        pd.DataFrame(c[num_samples_fit:num_samples_fit+num_samples_UQ,:]).to_csv(UQ_dir+"c_UQ.csv",header=False,index=False)
        # save the calibration data to the cal dir, if it does not exist, create it
        if not os.path.exists(LUQ_dir+"cal/"):
            os.makedirs(LUQ_dir+"cal/")
        cal_dir = LUQ_dir+"cal/"
        pd.DataFrame(covs[num_samples_fit+num_samples_UQ:num_samples_fit+num_samples_UQ+num_samples_cal,:]).to_csv(cal_dir+"covs_cal.csv",header=False,index=False)
        pd.DataFrame(c[num_samples_fit+num_samples_UQ:num_samples_fit+num_samples_UQ+num_samples_cal,:]).to_csv(cal_dir+"c_cal.csv",header=False,index=False)

        # check if the true_f and half_width exists, if not, skip
        if os.path.exists(train_sample_dir+"true_f.csv") and os.path.exists(train_sample_dir+"half_width.csv"):
            # save the UQ data to the UQ dir, if it does not exist, create it
            true_f = pd.read_csv(train_sample_dir+"true_f.csv",header=None).to_numpy()
            half_width = pd.read_csv(train_sample_dir+"half_width.csv",header=None).to_numpy()

            pd.DataFrame(true_f[0:num_samples_fit,:]).to_csv(fit_dir+"true_f_fit.csv",header=False,index=False)
            pd.DataFrame(half_width[0:num_samples_fit,:]).to_csv(fit_dir+"half_width_fit.csv",header=False,index=False)
            pd.DataFrame(true_f[num_samples_fit:num_samples_fit+num_samples_UQ,:]).to_csv(UQ_dir+"true_f_UQ.csv",header=False,index=False)
            pd.DataFrame(half_width[num_samples_fit:num_samples_fit+num_samples_UQ,:]).to_csv(UQ_dir+"half_width_UQ.csv",header=False,index=False)
            pd.DataFrame(true_f[num_samples_fit+num_samples_UQ:num_samples_fit+num_samples_UQ+num_samples_cal,:]).to_csv(cal_dir+"true_f_cal.csv",header=False,index=False)
            pd.DataFrame(half_width[num_samples_fit+num_samples_UQ:num_samples_fit+num_samples_UQ+num_samples_cal,:]).to_csv(cal_dir+"half_width_cal.csv",header=False,index=False)
        

def split_data_for_HetRes(dir_name,save_dir,train_ratio,max_UQ_num = 100):
    train_sample_dir = dir_name
    has_split = False
    # judge if the "LUQ" dir exists, if not, create it
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    else:
        has_split = True
    LUQ_dir = save_dir
    if not has_split:
        # load the data in train_sample_dir
        covs = pd.read_csv(train_sample_dir+"covs.csv",header=None).to_numpy()

        true_f = pd.read_csv(train_sample_dir+"true_f.csv",header=None).to_numpy()

        half_width = pd.read_csv(train_sample_dir+"half_width.csv",header=None).to_numpy()

        c = pd.read_csv(train_sample_dir+"c.csv",header=None).to_numpy()

        #split the data into three parts, the first part with 3/5 samples it used to fit a model , the second part with 1/5 for uncertainty quantification, and the third part with 1/5 for calibration
        num_samples = covs.shape[0]
        # if the number of samples for UQ is larger than max_UQ_num, then set the number of samples for UQ to max_UQ_num, and the number of samples for fitting to num_samples - max_UQ_num
        if num_samples*(1-train_ratio) > max_UQ_num:
            num_samples_UQ = max_UQ_num
            num_samples_fit = num_samples - num_samples_UQ
        else:
            num_samples_fit = int(num_samples*train_ratio)
            num_samples_UQ = num_samples - num_samples_fit
        

        # save the data into the "fit" dir, a sub dir of LUQ. If it does not exist, create it
        if not os.path.exists(LUQ_dir+"fit/"):
            os.makedirs(LUQ_dir+"fit/")
        fit_dir = LUQ_dir+"fit/"
        pd.DataFrame(covs[0:num_samples_fit,:]).to_csv(fit_dir+"covs_fit.csv",header=False,index=False)
        pd.DataFrame(true_f[0:num_samples_fit,:]).to_csv(fit_dir+"true_f_fit.csv",header=False,index=False)
        pd.DataFrame(half_width[0:num_samples_fit,:]).to_csv(fit_dir+"half_width_fit.csv",header=False,index=False)
        pd.DataFrame(c[0:num_samples_fit,:]).to_csv(fit_dir+"c_fit.csv",header=False,index=False)

        # save the UQ data to the UQ dir, if it does not exist, create it
        if not os.path.exists(LUQ_dir+"UQ/"):
            os.makedirs(LUQ_dir+"UQ/")
        UQ_dir = LUQ_dir+"UQ/"
        pd.DataFrame(covs[num_samples_fit:num_samples_fit+num_samples_UQ,:]).to_csv(UQ_dir+"covs_UQ.csv",header=False,index=False)
        pd.DataFrame(true_f[num_samples_fit:num_samples_fit+num_samples_UQ,:]).to_csv(UQ_dir+"true_f_UQ.csv",header=False,index=False)
        pd.DataFrame(half_width[num_samples_fit:num_samples_fit+num_samples_UQ,:]).to_csv(UQ_dir+"half_width_UQ.csv",header=False,index=False)
        pd.DataFrame(c[num_samples_fit:num_samples_fit+num_samples_UQ,:]).to_csv(UQ_dir+"c_UQ.csv",header=False,index=False)
        