from ctypes.wintypes import PINT
import h5py
import os
from numpy import concatenate
import torch
import numpy as np
import torch.optim as optim
from torch.utils import data
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import pandas as pd
import datetime
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import re
from io import StringIO
import ffmpeg
import pdb
from sklearn.metrics import (
    classification_report,
    roc_auc_score,
    confusion_matrix,
    roc_curve,
    f1_score,
    auc,
    precision_recall_curve,
)
import pdb

"""
 Function that takes inputs (sample, reconstruction, and label)
 generates reconstruction erros, and then generates performance metrics and saves these in a csv

performance metrics
- AUC ROC and PR for std, and mean of frame error (both day and night)
- AUC ROC and PR for std and mean of window error for different thresholds (both day and night)

"""

ht, wd = 64, 64

def get_performance_metrics(sample, output, labels, window_len):
    recon_data = output.reshape(output.shape[1], window_len, ht * wd)
    sample_data = sample.reshape(sample.shape[1], window_len,  ht * wd)
    labels = shape_labels(labels)
    window_std, window_mean, window_labels = get_window_metrics(sample_data, recon_data, labels, window_len)
    frame_std, frame_mean, frame_labels = get_frame_metrics(sample_data, recon_data, labels, window_len)
    return(frame_std, frame_mean, frame_labels, window_std, window_mean, window_labels)

def get_multimodal_preformance_metrics(sample, output, labels, window_len):
    recon_data = output.reshape(output.shape[0], output.shape[1], window_len, ht * wd)
    sample_data = sample.reshape(sample.shape[0], sample.shape[1], window_len,  ht * wd)
    labels = shape_labels(labels)
    window_std, window_mean, window_labels = get_multimodal_window_metrics(sample_data, recon_data, labels, window_len)
    frame_std, frame_mean, frame_labels = get_multimodal_frame_metrics(sample_data, recon_data, labels, window_len)
    return(frame_std, frame_mean, frame_labels, window_std, window_mean, window_labels)
    
    
    
def get_window_metrics(sample, output, labels, window_len):
    recon_error = np.mean(np.power(sample - output, 2), axis=2)
    mean_window_error = []
    std_window_error = [] 
    window_labels = [] 
    for tolerance in range(1, window_len):
        stride = 1
        windowed_labels = create_windowed_labels(labels, stride, tolerance, window_len)
        windowed_labels = windowed_labels[:,0]
        inwin_mean = np.mean(recon_error, axis =1)
        inwin_std = np.std(recon_error, axis =1)
        mean_window_error.append(inwin_mean)
        std_window_error.append(inwin_std)
        window_labels.append(windowed_labels)
    return(mean_window_error, std_window_error, window_labels)

def get_multimodal_window_metrics(sample, output, labels, window_len):
    recon_error = np.mean(np.mean(np.power(sample - output, 2), axis=3), axis = 0)
    mean_window_error = []
    std_window_error = [] 
    window_labels = [] 
    for tolerance in range(1, window_len+1):
        stride = 1
        windowed_labels = create_windowed_labels(labels, stride, tolerance, window_len)
        windowed_labels = windowed_labels[:,0]
        inwin_mean = np.mean(recon_error, axis =1)
        inwin_std = np.std(recon_error, axis =1)
        mean_window_error.append(inwin_mean)
        std_window_error.append(inwin_std)
        window_labels.append(windowed_labels)
    return(mean_window_error, std_window_error, window_labels)

def get_frame_metrics(sample, output, labels, window_len):

    recon_error = np.mean(np.power(sample - output, 2), axis=2)
    #print(("mse shape",recon_error.shape))
    # ------- Frame Reconstruction Error ---------------
    # create empty matrix w/ orignal number of frames
    mat = np.zeros((len(recon_error) + window_len - 1, len(recon_error)))
    mat[:] = np.NAN
    # dynmaically fill matrix with windows values for each frame
    #print(len(recon_error))
    for i in range(len(recon_error)):
        win = recon_error[i]
        mat[i : len(win) + i, i] = win
    frame_scores = []
    # each row corresponds to a frame across windows
    # so calculate stats for a single frame frame(row)
    for i in range(len(mat)):
        row = mat[i, :]
        mean = np.nanmean(row, axis=0)
        std = np.nanstd(row, axis=0)
        frame_scores.append((mean, std, mean + std * 10 ** 3))
    
    frame_scores = np.array(frame_scores)
    x_std = frame_scores[:, 1]
    x_mean = frame_scores[:, 0]

    return(x_mean, x_std, labels)

def get_multimodal_frame_metrics(sample, output, labels, window_len):

    recon_error = np.mean(np.mean(np.power(sample - output, 2), axis=3), axis = 0)
    #print(("mse shape",recon_error.shape))
    # ------- Frame Reconstruction Error ---------------
    # create empty matrix w/ orignal number of frames
    mat = np.zeros((len(recon_error) + window_len - 1, len(recon_error)))
    mat[:] = np.NAN
    # dynmaically fill matrix with windows values for each frame
    #print(len(recon_error))
    for i in range(len(recon_error)):
        win = recon_error[i]
        mat[i : len(win) + i, i] = win
    frame_scores = []
    # each row corresponds to a frame across windows
    # so calculate stats for a single frame frame(row)
    for i in range(len(mat)):
        row = mat[i, :]
        mean = np.nanmean(row, axis=0)
        std = np.nanstd(row, axis=0)
        frame_scores.append((mean, std, mean + std * 10 ** 3))
    
    frame_scores = np.array(frame_scores)
    x_std = frame_scores[:, 1]
    x_mean = frame_scores[:, 0]

    return(x_mean, x_std, labels)

def get_global_performance_metrics(name, frame_stats, window_stats, window_len):
    
    frame_mean_flat = [] 
    frame_std_flat = []
    frame_labels_flat = []
    for i in range(len(frame_stats)):
        # this a single video metrics
        frame_mean, frame_std, frame_labels = frame_stats[i]
        frame_mean_flat.extend(frame_mean)
        frame_std_flat.extend(frame_std)
        frame_labels_flat.extend(frame_labels[:len(frame_std)])
        
    window_mean_thres = [] 
    window_std_thres = []
    window_labels_thres = []
    for j in range(0, window_len-1):
        window_mean_flat =[]
        window_std_flat = []
        window_label_flat = []
        for i in range(len(window_stats)):
            mean_window_error, std_window_error, window_labels = window_stats[i]
            window_mean_flat.extend(mean_window_error[j])
            window_std_flat.extend(std_window_error[j])
            window_label_flat.extend(window_labels[j])
        window_mean_thres.append(window_mean_flat)
        window_std_thres.append(window_std_flat)
        window_labels_thres.append(window_label_flat)
    
    video_metrics = np.zeros((5, window_len ))
    video_metrics[0, 0], video_metrics[1, 0], video_metrics[2, 0], video_metrics[3, 0] = get_performance_values(frame_mean_flat, frame_std_flat, frame_labels_flat)
    # store each thresholds results for this video 
    for j in range(0, window_len-1):
        vid_labels = window_labels_thres[j]
        
        (unique, counts) = np.unique(vid_labels, return_counts=True)
        frequencies = np.asarray((unique, counts)).T 
        
        if len(np.unique(vid_labels)) != 2:
            continue
        video_metrics[0, j+1], video_metrics[1, j+1], video_metrics[2, j+1], video_metrics[3, j+1] = get_performance_values(window_mean_thres[j], window_std_thres[j], vid_labels)
        video_metrics[4, j+1] = j
    
    pd.DataFrame(video_metrics).to_csv('FinalResults\{}global_cross_context_results.csv'.format(name))    
    return()
    

def get_total_performance_metrics(name, frame_stats, window_stats, window_len):
    get_curves_and_thresholds(name, frame_stats, window_stats, window_len)

    video_metrics = np.zeros((len(frame_stats), 5, window_len ))

    # here i need to get the error and everything and store it in video metrics 

    for i in range(len(frame_stats)):
        # print(i)
        # this a single video metrics
        frame_mean, frame_std, frame_labels = frame_stats[i]
        mean_window_error, std_window_error, window_labels = window_stats[i]
    
        # store frame results for this vidoe
        video_metrics[i, 0, 0], video_metrics[i, 1, 0], video_metrics[i, 2, 0], video_metrics[i, 3, 0] = get_performance_values(frame_mean, frame_std, frame_labels)
        video_metrics[i, 4, 0] = 0 

        # store each thresholds results for this video 
        for j in range(1, window_len):
            vid_labels = window_labels[j-1]
            #print(vid_labels)
            if len(np.unique(vid_labels)) != 2:
                # print("ERROR: no fall for threshold {} in video {}".format(j, i))
                continue
            video_metrics[i, 0, j], video_metrics[i, 1, j], video_metrics[i, 2, j], video_metrics[i, 3, j] = get_performance_values(mean_window_error[j-1], std_window_error[j-1], vid_labels)
            video_metrics[i, 4, j] = j 
    
    video_metrics[video_metrics == 0] = np.nan
    final_performance_mean = np.nanmean(video_metrics, axis=0) # get the mean performance across all videos 
    final_performance_std = np.nanstd(video_metrics, axis=0) #get the standard dev for each mean

    #np.savetxt('results.csv', final_performance, delimiter=',', fmt='%d')

    pd.DataFrame(final_performance_mean).to_csv('FinalResults\{}results_mean.csv'.format(name))
    pd.DataFrame(final_performance_std).to_csv('FinalResults\{}results_std.csv'.format(name))
    
    return(final_performance_mean, final_performance_std)

def late_fusion_performance_metrics(output, labels):
    roc_fpr, roc_tpr, roc_thresholds = roc_curve(y_true=labels[:len(output)], y_score=output, pos_label=1)
    mean_AUROC = auc(roc_fpr, roc_tpr)
    
    precision, recall, pr_thresholds = precision_recall_curve(labels[:len(output)], output, pos_label=1)
    mean_AUPR = auc(recall, precision )
    
    return(mean_AUROC, mean_AUPR)

def get_curves_and_thresholds(name, frame_stats, window_stats, window_len):
    frame_mean_flat = [] 
    frame_std_flat = []
    frame_labels_flat = []
    for i in range(len(frame_stats)):
        # print(i)
        # this a single video metrics
        frame_mean, frame_std, frame_labels = frame_stats[i]
        frame_mean_flat.extend(frame_mean)
        frame_std_flat.extend(frame_std)
        frame_labels_flat.extend(frame_labels[:len(frame_std)])
    
    
    def plot_ROC_AUC(fpr, tpr, roc_auc, data_option):
        plt.figure()
        lw = 2
        plt.plot(fpr, tpr, color='darkorange',
            lw=lw, label='ROC curve (area = %0.4f)' % roc_auc)
        plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic for {}'.format(data_option))
        plt.legend(loc="lower right")
        plt.savefig('FinalResults/{}{}ROC_AUC.png'.format(name, data_option))
        
    def plot_PR_AUC(precision, recall, mean_AUPR, no_skill, data_option):
        plt.figure()
        plt.plot([0,1], [no_skill,no_skill], linestyle='--', label='No Skill (area = %0.4f)' % no_skill)
        plt.plot(recall, precision, marker='.', label='AUPR curve (area = %0.4f)' % mean_AUPR)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, max(precision)/5])
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision Recall Curve for {}'.format(data_option))
        plt.legend(loc="upper right")
        plt.savefig('FinalResults/{}{}PR_AUC.png'.format(name, data_option))
        
    # calculate metrics for Standard Deviation 
    roc_fpr, roc_tpr, roc_thresholds = roc_curve(y_true=frame_labels_flat, y_score=frame_std_flat, pos_label=1)
    std_AUROC = auc(roc_fpr, roc_tpr)
    data_option = 'STD of Reconstruction Error'
    plot_ROC_AUC(roc_fpr, roc_tpr, std_AUROC, data_option) 
    
    precision, recall, pr_thresholds = precision_recall_curve(frame_labels_flat, frame_std_flat, pos_label=1)
    data_option = 'STD of Reconstruction Error'
    no_skill = frame_labels_flat.count(1) / len(frame_labels_flat)
    mean_AUPR = auc(recall, precision )
    plot_PR_AUC(precision, recall, mean_AUPR, no_skill, data_option)
    
    '''
    gmeans = np.sqrt(roc_tpr * (1-roc_fpr))
    ix = np.argmax(gmeans)
    print('Best Threshold=%f, G-Mean=%.3f' % (roc_thresholds[ix], gmeans[ix]))
    optimal_threshold = roc_thresholds[ix]
    '''
    
    optimal_idx = np.argmax(roc_tpr - roc_fpr)
    optimal_threshold = roc_thresholds[optimal_idx]

    for i in range(len(frame_std_flat)):
        if frame_std_flat[i] >= optimal_threshold:
            frame_std_flat[i] = 1
        else:
            frame_std_flat[i] = 0

    std_tn, std_fp, std_fn, std_tp = confusion_matrix(frame_labels_flat, frame_std_flat).ravel()    #, labels=[0,1]
    std_TPR = std_tp/(std_tp+std_fn)
    std_FPR = std_fp/(std_fp+std_tn) 
    std_Precision = std_tp/(std_tp+std_fp)
    std_Recall = std_tp/(std_tp+std_fn)
    print("----------------------------------")
    print("STD Global Classification Results")
    print('TPR {:.3f}, FPR {:.3f}, Precision {:.3f}, Recall {:.3f}'.format(std_TPR, std_FPR, std_Precision, std_Recall))
    print('tn {}, fp {}, fn {}, tp {}'.format(std_tn, std_fp, std_fn, std_tp))
    print("std_AUROC  {:.3f}".format(std_AUROC))
    print("----------------------------------")

    # calculate the Mean AUC
    roc_fpr, roc_tpr, roc_thresholds = roc_curve(y_true=frame_labels_flat, y_score=frame_mean_flat, pos_label=1)
    mean_AUROC = auc(roc_fpr, roc_tpr)
    data_option = 'Mean of Reconstruction Error'
    plot_ROC_AUC(roc_fpr,roc_tpr, mean_AUROC, data_option)
    
    precision, recall, thresholds = precision_recall_curve(frame_labels_flat, frame_mean_flat, pos_label=1)
    data_option = 'Mean of Reconstruction Error'
    no_skill = frame_labels_flat.count(1) / len(frame_labels_flat)
    mean_AUPR = auc(recall, precision )
    plot_PR_AUC(precision, recall, mean_AUPR, no_skill, data_option)
    
    optimal_idx = np.argmax(roc_tpr - roc_fpr)
    optimal_threshold = roc_thresholds[optimal_idx]

    for i in range(len(frame_mean_flat)):
        if frame_mean_flat[i] >= optimal_threshold:
            frame_mean_flat[i] = 1
        else:
            frame_mean_flat[i] = 0

    tn, fp, fn, tp = confusion_matrix(frame_labels_flat, frame_mean_flat).ravel()    
    TPR = tp/(tp+fn)
    FPR = fp/(fp+tn) 
    Precision = tp/(tp+fp)
    Recall = tp/(tp+fn)
    print("----------------------------------")
    print("Mean Global Classification Results")
    print('TPR {:.3f}, FPR {:.3f}, Precision {:.3f}, Recall {:.3f}'.format(TPR, FPR, Precision, Recall))
    print('tn {}, fp {}, fn {}, tp {}'.format(tn, fp, fn, tp))
    print("mean_AUROC {:.3f}".format(mean_AUROC))
    print("----------------------------------")

    d = {'TPR': [TPR, std_TPR], 'FPR': [FPR, std_FPR], 'Precision': [Precision, std_Precision], 'Recall': [Recall, std_Recall], 
        'tn': [tn, std_tn], 'fp': [fp, std_fp], 'fn': [fn, std_fn], 'tp': [tp, std_tp]}
    df = pd.DataFrame(data=d)
    pd.DataFrame(df).to_csv('FinalResults\{}global_cross_context_classification_results.csv'.format(name))
    return()





def get_performance_values(vid_mean, vid_std, vid_labels):

    # calculate metrics for Standard Deviation 
    std_fpr, std_tpr, std_thresholds = roc_curve(y_true=vid_labels[:len(vid_std)], y_score=vid_std, pos_label=1)
    std_AUROC = auc(std_fpr, std_tpr)

    std_precision, std_recall, thresholds = precision_recall_curve(vid_labels[:len(vid_std)], vid_std)
    std_AUPR = auc(std_recall, std_precision)

    # calculate the Mean AUC
    fpr, tpr, thresholds = roc_curve(y_true=vid_labels[:len(vid_std)], y_score = vid_mean, pos_label=1)
    mean_AUROC = auc(fpr, tpr)
    
    mean_precision, mean_recall, thresholds = precision_recall_curve(vid_labels[:len(vid_std)], vid_mean)
    mean_AUPR = auc(mean_recall, mean_precision)
    data_option = 'Mean Err'

    #print('-------------------------------------')
    return std_AUROC, mean_AUROC, std_AUPR, mean_AUPR 




def shape_labels(labels):
    # generate labels
    label = labels[0, :, :]
    windowed_labels = label # shape (window_len, # of windows) 
    frame_labels = un_window(label)
    return frame_labels



def create_windowed_labels(labels, stride, tolerance, window_length):
    output_length = int(np.floor((len(labels) - window_length) / stride))+1
    output_shape = (output_length, 1)
    total = np.zeros(output_shape)
    i=0
    while i < output_length:
        next_chunk = np.array([labels[i+j] for j in range(window_length)])
        num_falls = sum(next_chunk) #number of falls in the window

        if num_falls >= tolerance:
            total[i] = 1
        else:
            total[i] = 0

        i = i+stride
    labels_windowed = total
    return labels_windowed

def un_window(windowed_data):
# Input: Windowed Data with format (window_length, # of windows )
    unwindowed_data = np.zeros(windowed_data.shape[0] + windowed_data.shape[1])
    for i in range(len(unwindowed_data)):
        if i >= windowed_data.shape[1]:
            last_window = windowed_data[:, i - 1]
            unwindowed_data[i:] = last_window
            break
        else:
            unwindowed_data[i] = windowed_data[0, i]

    return unwindowed_data

def animate(test_data, recons_seq, frame_mean, dset, start_time):
    ani_dir = "./Animation/{}/".format(dset)
    ani_dir = ani_dir + "/{}".format(start_time)
    if not os.path.isdir(ani_dir):
        os.makedirs(ani_dir)
    print("saving animation to {}".format(ani_dir))

    animate_fall_detect_present(
        testfall=test_data[:, 0, :].reshape(len(test_data),  ht, wd, 1),
        recons=recons_seq[:, 0, :].reshape(len(recons_seq),  ht, wd, 1),
        win_len=1,
        scores=frame_mean,
        to_save=ani_dir + "/{}.mp4".format(len(test_data)),
    )


def animate_fall_detect_present(
    testfall, recons, scores, win_len, threshold=0, to_save="./test.mp4"
):
    """
    Pass in data for single video, recons is recons frames, scores is x_std or x_mean etc.
    Threshold is RRE, mean, etc..
    """
    import matplotlib.gridspec as gridspec

    Writer = animation.writers["pillow"]
    writer = Writer(fps=15, metadata=dict(artist="Me"), bitrate=1800)
    gs = gridspec.GridSpec(2, 2, height_ratios=[2, 1])


    eps = 0.0001
    # setup figure
    # fig = plt.figure()
    fig, ((ax1, ax3)) = plt.subplots(1, 2, figsize=(6, 6))

    ax1.axis("off")
    ax3.axis("off")
    # ax1=fig.add_subplot(2,2,1)

    ax1 = fig.add_subplot(gs[0, 0])
    ax1.set_title("Original")
    ax1.set_xticks([])
    ax1.set_yticks([])

    # ax2=fig.add_subplot(gs[-1,0])
    ax2 = fig.add_subplot(gs[1, :])

    # ax2.set_yticks([])
    # ax2.set_xticks([])
    ax2.set_ylabel("Score")
    ax2.set_xlabel("Frame")

    if threshold != 0:
        ax2.axhline(y=threshold, color="r", linestyle="dashed", label="RRE")
        ax2.legend()

    # ax3=fig.add_subplot(2,2,2)
    ax3 = fig.add_subplot(gs[0, 1])
    ax3.set_title("Reconstruction")
    ax3.set_xticks([])
    ax3.set_yticks([])

    # set up list of images for animation
    ims = []

    for time in range(1, len(testfall) - (win_len - 1) - 1):
        im1 = ax1.imshow(testfall[time].reshape(ht, wd), cmap="gray", aspect="equal")
        figure = recons[time].reshape(ht, wd)
        im2 = ax3.imshow(figure, cmap="gray", aspect="equal")
        # print(im1.shape)
        # print(im2.shape)
        
        # print("time={} mse={} std={}".format(time,mse_difficult[time],std))
        if time > 0:

            scores_curr = scores[0:time]

            fall_pts_idx = np.argwhere(scores_curr > threshold)
            nonfall_pts_idx = np.argwhere(scores_curr <= threshold)

            fall_pts = scores_curr[fall_pts_idx]
            nonfall_pts = scores_curr[nonfall_pts_idx]

            if fall_pts_idx.shape[0] > 0:
                # pass
                (plot_r,) = ax2.plot(fall_pts_idx, fall_pts, "r.")
                (plot,) = ax2.plot(nonfall_pts_idx, nonfall_pts, "b.")
            else:
                (plot,) = ax2.plot(scores_curr, "b.")
                (plot_r,) = ax2.plot(scores_curr, "r.")

        else:
            (plot,) = ax2.plot(scores[0], "b.")
            (plot_r,) = ax2.plot(scores[0], "r.")

        ims.append([im1, plot, im2, plot_r])  # list of ims

    # run animation
    ani = animation.ArtistAnimation(fig, ims, interval=40, repeat=False)
    # plt.tight_layout()
    #plt.show()
    # gs.tight_layout(fig)
    ani.save(to_save)

    ani.event_source.stop()
    del ani
    plt.close()
    # plt.show()
    # return ani



def create_pytorch_dataset(name, dset, path, window_len, fair_compairson, stride, TOD = "Both"):
    falls = []
    adl = []
    if fair_compairson == True:
        shared_adl_vids = np.loadtxt('shared_adl_vids.txt').astype(int)
        shared_fall_vids = np.loadtxt('shared_fall_vids.txt').astype(int)
        day_fall_vids = np.loadtxt('day_fall_vids.txt').astype(int)
        night_fall_vids = np.loadtxt('night_fall_vids.txt').astype(int)
        if TOD == "Day":
            tod_list = day_fall_vids
        if TOD == "Night":
            tod_list = night_fall_vids   
        if TOD == "Both":
            tod_list = shared_fall_vids
            
        #print(shared_fall_vids)
        # create list of all fall and nonfall folders
        for (root, dirs, files) in os.walk("S:/{}/Fall".format(dset)):
            for dir in dirs:
                x = re.findall('[0-9]+', dir)[0]
                if (int(x) in shared_fall_vids) and (int(x) in tod_list):
                    falls.append(dir)
        
        for (root, dirs, files) in os.walk("S:/{}/NonFall".format(dset)):
            for dir in dirs:
                x = re.findall('[0-9]+', dir)[0]
                if int(x) in shared_adl_vids:
                    adl.append(dir)
        #print(falls)
        #print(adl)
    elif fair_compairson == False:
        # create list of all fall and nonfall folders
        for (root, dirs, files) in os.walk("S:/{}/Fall".format(dset)):
            if len(dirs) > 0:
                falls.extend(dirs)
        for (root, dirs, files) in os.walk("S:/{}/NonFall".format(dset)):
            if len(dirs) > 0:
                adl.extend(dirs)
        #print(falls)
        #print(adl)
        

    x_data_fall = []
    y_data_fall = []
    x_data_adl = []
    y_data_adl = []
    x_info_fall = []
    x_info_adl = []

    # path = "processed_data\data_set-{}-imgdim64x64.h5".format(name)

    # load in images of falls
    with h5py.File(path, "r") as hf:
        data_dict = hf["{}/Processed/Split_by_video".format(name)]
        # print(data_dict.keys())
        for Fall_name in falls:
            try:
                vid_total = data_dict[Fall_name]["Data"][:]
                if len(vid_total) < 10:
                    continue
                x_data_fall.append(vid_total)
                x_info_fall.append(Fall_name) #[4:]
                labels_total = data_dict[Fall_name]["Labels"][:]
                y_data_fall.append(labels_total)
            except:
                print("Skipped", Fall_name)
                
            #if len(x_data_fall) > 5:
                #break

        for adl_name in adl:
            try:
                vid_total = data_dict[adl_name]["Data"][:]
                if len(vid_total) < 10:
                    continue
                x_data_adl.append(vid_total)
                x_info_adl.append(adl_name) #[7:]
                labels_total = data_dict[adl_name]["Labels"][:]
                y_data_adl.append(labels_total)
            except:
                print("Skipped", adl_name)
                
            #if len(x_data_fall) > 5:
                #break
            




    
    # get matching day/night label from falls

        

    # pdb.set_trace()
    # %%    temp_df = my_data.loc[my_data["Video"] == int(fall), "ToD"]

    # ----------------------------------------------------------------------------
    # *** PREPARING DATASET LOADER ***
    # ----------------------------------------------------------------------------

    # 1) Need a ADL loader and a Fall Loader

    class Dataset(data.Dataset):
        "Characterizes a dataset for PyTorch"

        def __init__(self, labels, data, window):
            "Initialization"
            self.labels = labels
            self.data = data
            self.window = window

        def __len__(self):
            "Denotes the total number of samples"
            return len(self.data)

        def __getitem__(self, index):
            "Generates one sample of data"
            # prepare lists to dynamically fill with windows
            X_list = []
            Y_list = []
            # load a single video to chop up into windows
            ind_vid = self.data[index]
            ind_label = self.labels[index]
            # loop through each frame of the video (stopping window length short)
            for i in range(len(ind_vid) - self.window):
                # select the current window of the video
                X = ind_vid[i : i + self.window]
                y = ind_label[i : i + self.window]
                # add the current window the list of windows
                X_list.append(X)
                Y_list.append(y)
            # convert lists into arrays with proper size
            X = np.vstack(X_list)
            X = np.reshape(X_list, (len(ind_vid) - self.window, self.window, ht, wd))
            y = np.vstack(Y_list).T
            # X should be (window-length, 64, 64, # of windows w/in video) array
            # ex. (8, 64, 64, 192) for a 200 frame video and window size of 8
            # y is array (8, # of windows w/in video)
            return X, y

    print(len(x_data_fall))
    print(len(x_data_adl))
    Test_Dataset = Dataset(y_data_fall, x_data_fall, window=window_len)
    test_dataloader = data.DataLoader(Test_Dataset, batch_size=1)

    Train_Dataset = Dataset(y_data_adl, x_data_adl, window=window_len)
    train_dataloader = data.DataLoader(Train_Dataset, batch_size=1)

    return(Test_Dataset, test_dataloader, Train_Dataset, train_dataloader)

def late_fusion_performance_metricsV4(output, original, window_len, labels):
    labels = un_window(labels)
    recon_error = output
    # ------- Frame Reconstruction Error ---------------
    # create empty matrix w/ orignal number of frames
    mat = np.zeros((len(recon_error) + window_len - 1, len(recon_error)))
    mat[:] = np.NAN
    # dynmaically fill matrix with windows values for each frame
    #print(len(recon_error))
    for i in range(len(recon_error)):
        win = recon_error[i]
        #print(len(win))
        #print(i)
        mat[i : len(win) + i, i] = win
    frame_scores = []
    # each row corresponds to a frame across windows
    # so calculate stats for a single frame frame(row)
    for i in range(len(mat)):
        row = mat[i, :]
        mean = np.nanmean(row, axis=0)
        std = np.nanstd(row, axis=0)
        frame_scores.append((mean, std, mean + std * 10 ** 3))
    
    frame_scores = np.array(frame_scores)
    x_std = frame_scores[:, 1]
    x_mean = frame_scores[:, 0]
    
    roc_fpr, roc_tpr, roc_thresholds = roc_curve(y_true=labels[:-1], y_score=x_mean, pos_label=1)
    mean_AUROC = auc(roc_fpr, roc_tpr)
    
    precision, recall, pr_thresholds = precision_recall_curve(labels[:-1], x_mean, pos_label=1)
    mean_AUPR = auc(recall, precision )
    
    roc_fpr, roc_tpr, roc_thresholds = roc_curve(y_true=labels[:-1], y_score=x_std, pos_label=1)
    std_AUROC = auc(roc_fpr, roc_tpr)
    
    precision, recall, pr_thresholds = precision_recall_curve(labels[:-1], x_std, pos_label=1)
    std_AUPR = auc(recall, precision )
    
    return(mean_AUROC, mean_AUPR, std_AUROC, std_AUPR)

def create_multimodal_pytorch_dataset(names, dsets, window_len, fair_compairson, stride):
    
    class Multi_Dataset(data.Dataset):
        "Characterizes a dataset for PyTorch"

        def __init__(self, labels, datas, window):
            "Initialization"
            self.labels = labels
            self.datas = datas
            self.window = window

        def __len__(self):
            "Denotes the total number of modalities"
            print(len(self.datas[0]))
            return(len(self.datas[0]))

        def __getitem__(self, index):
            "Generates one sample of data"
            
            x_data = self.datas
            y_data = self.labels

            vid = index
            X_modality_list = []
            y_modality_list = []
            # loop through each frame of the video (stopping window length short)
            for j in range(len(x_data)):
                X_list = []
                Y_list = []
                #have to make arrays same size 
                max_length = 0 
                for k in range(len(x_data)):
                    if len(x_data[k][vid]) > max_length:
                        max_length = len(x_data[k][vid]) 
                for k in range(len(x_data)):
                    while len(x_data[k][vid]) < max_length:
                        x_data[k][vid] = np.pad(x_data[k][vid], [(0, 1), (0, 0), (0, 0), (0, 0)], 'mean')#, (0,0)
                        y_data[k][vid] = np.append(y_data[k][vid], 0) #, (0,0)
                #create windows

                for i in range(0, len(y_data[j][vid]) - window_len):
                    # select the current window of the video
                    X = x_data[j][vid][i : i + window_len][:]
                    y = y_data[j][vid][i : i + window_len]
                    # add the current window the list of windows
                    X_list.append(X)
                    Y_list.append(y)
                #save videos into list 
                X_modality_list.append(np.asarray(X_list))
                y_modality_list.append(np.asarray(Y_list))
            X = np.squeeze(np.stack(X_modality_list))
            y = np.squeeze(np.stack(y_modality_list))
            return(X, y)
            '''
            mod_vid = [] 
            mod_labels = []
            for j in range(len(self.datas)):
                video = self.datas[j][index]
                label = self.labels[j][index]
                mod_vid.append(np.squeeze(video))
                mod_labels.append(np.squeeze(label))
            X = np.squeeze(np.stack(mod_vid))
            y = np.squeeze(np.stack(mod_labels))
            '''
            # X should be (modalities, window-length, 64, 64, # of windows w/in video) array
            # ex. (3, 8, 64, 64, 192) for a 200 frame video and window size of 8 and 3 modalities
            # y is array (8, # of windows w/in video)
            return X, y
    
    def load_data(name, dset, path, fair_compairson):
        print(name, dset, path)
        falls = []
        adl = []
        if fair_compairson == True:
            shared_adl_vids = np.loadtxt('shared_adl_vids.txt').astype(int)
            shared_fall_vids = np.loadtxt('shared_fall_vids.txt').astype(int)
            # create list of all fall and nonfall folders
            for (root, dirs, files) in os.walk("S:/{}/Fall".format(dset)):
                for dir in dirs:
                    x = re.findall('[0-9]+', dir)[0]
                    if int(x) in shared_fall_vids:
                        falls.append(dir)
            
            for (root, dirs, files) in os.walk("S:/{}/NonFall".format(dset)):
                for dir in dirs:
                    x = re.findall('[0-9]+', dir)[0]
                    if int(x) in shared_adl_vids:
                        adl.append(dir)

        elif fair_compairson == False:
            # create list of all fall and nonfall folders
            for (root, dirs, files) in os.walk("S:/{}/Fall".format(dset)):
                if len(dirs) > 0:
                    falls.extend(dirs)
            for (root, dirs, files) in os.walk("S:/{}/NonFall".format(dset)):
                if len(dirs) > 0:
                    adl.extend(dirs)

        

        x_data_fall = []
        y_data_fall = []
        x_data_adl = []
        y_data_adl = []
        x_info_fall = []
        x_info_adl = []
        

        # path = "processed_data\data_set-{}-imgdim64x64.h5".format(name)

        # load in images of falls
        with h5py.File(path, "r") as hf:
            data_dict = hf["{}/Processed/Split_by_video".format(name)]
            # print(data_dict.keys())

            for Fall_name in falls:
                try:
                    vid_total = data_dict[Fall_name]["Data"][:]
                    if len(vid_total) < 10:
                        continue
                    x_data_fall.append(vid_total)
                    x_info_fall.append(Fall_name) #[4:]
                    labels_total = data_dict[Fall_name]["Labels"][:]
                    y_data_fall.append(labels_total)
                except:
                    print("Skipped", Fall_name)
                
        
            for adl_name in adl:
                try:
                    vid_total = data_dict[adl_name]["Data"][:]
                    if len(vid_total) < 10:
                        print(adl_name)
                        continue
                    x_data_adl.append(vid_total)
                    x_info_adl.append(adl_name) #[7:]
                    labels_total = data_dict[adl_name]["Labels"][:]
                    y_data_adl.append(labels_total)
                except:
                    print("Skipped", adl_name)


        
        ''' 
        # get matching day/night label from falls
        labels_dir = "D:/{}/".format(dset) + "Labels.csv"
        my_data = pd.read_csv(labels_dir)
        # sorting by first name
        my_data.sort_values("Video", inplace=True)
        my_data.drop_duplicates(subset="Video", keep="first", inplace=True)
        print(my_data.head())
        '''
        return(y_data_fall, x_data_fall, x_data_adl, y_data_adl)
    # ----------------------------------------------------------------------------
    # *** PREPARING DATASET LOADER ***
    # ----------------------------------------------------------------------------

    # 1) Need a ADL loader and a Fall Loader


    y_data_falls = []
    x_data_falls = []
    y_data_adls = []
    x_data_adls = []
    for i in range(len(dsets)):
            path = "S:\\H5PY\Data_set-{}-imgdim64x64.h5".format(names[i])
            print('loading', names[i])
            y_data_fall, x_data_fall, x_data_adl, y_data_adl = load_data(names[i], dsets[i], path, fair_compairson)
            y_data_falls.append(y_data_fall)
            x_data_falls.append(x_data_fall)
            y_data_adls.append(y_data_adl)
            x_data_adls.append(x_data_adl)
            print(len(y_data_fall))
            print(len(x_data_fall))
            print(len(y_data_adl))
            print(len(x_data_adl))
            del y_data_fall
            del x_data_fall
            del y_data_adl
            del x_data_adl
    
    '''' 
    def re_arrange_data(x_data, y_data):
        #(3, 8, 64, 64, 192) for a 200 frame video and window size of 8 and 3 modalities
        #loop through each modality 
        X_modality_list = []
        y_modality_list = []
        # loop through each frame of the video (stopping window length short)
        for j in range(0, len(x_data)):
            print(j)
            X_video_list = []
            y_label_list = []
            for vid in range(len(x_data[j])):
                print(vid)
                X_list = []
                Y_list = []
                #find shortest modality
                max_length = 0 
                for k in range(len(x_data)):
                    if len(x_data[k][vid]) > max_length:
                        max_length = len(x_data[k][vid]) 
                for j in range(len(x_data)):
                    while len(x_data[k][vid]) < max_length:

                        x_data[k][vid] = np.pad(x_data[k][vid], [(0, 1), (0, 0), (0, 0), (0, 0)], 'mean')#, (0,0)
                        y_data[k][vid] = np.append(y_data[k][vid], 0) #, (0,0)

                for i in range(0, len(y_data[j][vid]) - window_len):
                    # select the current window of the video
                    X = x_data[j][vid][i : i + window_len][:]
                    y = y_data[j][vid][i : i + window_len]
                    # add the current window the list of windows
                    X_list.append(X)
                    Y_list.append(y)
                X_video_list.append(np.asarray(X_list))
                y_label_list.append(np.asarray(Y_list))
                
            X_modality_list.append(X_video_list)
            y_modality_list.append(y_label_list)
        return(X_modality_list, y_modality_list)

    X_modality_list, y_modality_list = re_arrange_data(x_data_falls, y_data_falls)
    '''

    Test_Dataset = Multi_Dataset(y_data_falls, x_data_falls, window=window_len)
    test_dataloader = data.DataLoader(Test_Dataset, batch_size=1)

    #X_modality_list, y_modality_list = re_arrange_data(x_data_adls, y_data_adls)

    Train_Dataset = Multi_Dataset(y_data_adls, x_data_adls, window=window_len)
    train_dataloader = data.DataLoader(Train_Dataset, batch_size=1)

    return(Test_Dataset, test_dataloader, Train_Dataset, train_dataloader)


