
import numpy as np
import scipy.io as sio
from util import util_dataload as udl



def get_dataInfo(config, subject_name):

    subject_names = config['subjectInfo']['names']
    pycortex_db_names = config['subjectInfo']['pycortex_db_names']
    isid = np.where(np.array(subject_names) == subject_name)[0][0]

    load_items = udl.set_load_items(config, subjectID=isid, movID=0)
    tvoxels = udl.load_mask_data(load_items['mask_path']) + 1 #(!) python --> matlab order
    datasize = udl.get_datasize(config, isid)
    
    dataInfo = dict()
    dataInfo['maindir'] = config['dir']['resp'].replace('{derivative}', config['dir']['derivative'])
    dataInfo['maskdir'] = config['dir']['mask'].replace('{derivative}', config['dir']['derivative'])
    dataInfo['loadInfoSbID'] = isid
    dataInfo['tvoxels'] = tvoxels #(!) matlab order
    dataInfo['datasize'] = datasize
    dataInfo['subjectName'] = subject_names[isid]
    dataInfo['subjectNamePycortex'] = pycortex_db_names[isid]
    dataInfo['dataSetName'] = config['pycortexInfo']['dataSetName']
    dataInfo['dataInfo_path'] = config['path']['dataInfo']
    dataInfo['locNames'] = config['localizerInfo']['locLabels']
    dataInfo['plotTypes'] = config['localizerInfo']['plotTypes']

    return dataInfo

def assign_color_to_each_roi(n_rois, nLaps=1):

    if n_rois > 3:
        colors = list()
        for lap_id in range(nLaps):
            colors.append([0.95, 0.05, 0.2]) # red
            colors.append([0.85, 0.55, 0.1]) # orange
            colors.append([0.9, 0.95, 0.05]) # yellow
            colors.append([0.15, 0.75, 0.1]) # green
            colors.append([0.15, 0.80, 0.9]) # cyan
            colors.append([0.45, 0.45, 1]) # blue
            colors.append([0.75, 0.2, 0.85]) # purple
            colors.append([0.95, 0.05, 0.2]) # red (repeat)
            
        colors = np.array(colors)
    
        n_colors = len(colors)
        nbins = int( np.ceil(50/len(colors)) )
    
        color_map = list()
        for ii in range(n_colors-1):
            for jj in range(nbins):
                t_color = (colors[ii]*(nbins-jj)/nbins) + (colors[ii+1]*jj/nbins)
                color_map.append(list(t_color)) 
        n_color_map = len(color_map)
        interval = int( np.floor(n_color_map/n_rois) )
    
        del colors
        colors = color_map[::interval]

    elif n_rois == 1:
        colors = [[1, 0, 0]]
        
    elif n_rois == 2:
        colors = list()
        colors.append([1,0,0]) # red
        colors.append([0,1,0]) # green

    elif n_rois == 3:
        colors = list()
        colors.append([1,0,0]) # red
        colors.append([0,1,0]) # green
        colors.append([0,0,1]) # blue

    return colors 


def norm_weight(weight):
    
    min_w = weight.min(axis=0)
    weight = weight - min_w
    weight = weight/np.sum(weight, axis=0)
    
    return weight


### Reconstruct dt as data with org epi shape (to show using pycortex)

def get_dt3d_vol(dataInfo, stats_, sigs_, nLaps=1, normData=True, integrationScale=0.5, get_dt3d_sig=True, statBase=np.nan):

    # INPUT
    # dataInfo
    #      'maindir': directory including 'dataInfo.mat'
    #      'maskdir': directory including mask file ('vsetXXX.mat')
    #      'loadInfoSbID': subject #
    # stats_rois: list for np array of stat vec (size: [tvoxels, ])
    # sigs_rois: list for np array of sig vec (size: [tvoxels, ])

    # load brain info
    tvoxels = dataInfo['tvoxels'] #(!)matlab order
    datasize = dataInfo['datasize']

    nStats = len(stats_)
    colors = assign_color_to_each_roi(nStats, nLaps=nLaps)

    ###
    ### Set 3d volumes for stat, roi, and mask
    
    dt3d_stat_ = list()
    dt3d_sig_ = list()
    for roi_id in range(nStats):

        ### Set temporal color/stats/ sigTvoxelsInds (for each roi)
        tcolor = colors[roi_id]
        stats  = stats_[roi_id]
    
        ### Stat
        # Normalize stats & assign color
        if normData == True:
            stats_min = stats[~np.isnan(stats)].min()
            stats_max = stats[~np.isnan(stats)].max()
            d_stats = stats_max - stats_min
            n_stats = (stats - stats_min)/d_stats # ranging from 0 to 1
            stats_color_R = n_stats* tcolor[0]
            stats_color_G = n_stats* tcolor[1]
            stats_color_B = n_stats* tcolor[2]
        else:
            stats_color_R = stats* tcolor[0]
            stats_color_G = stats* tcolor[1]
            stats_color_B = stats* tcolor[2]
    
        # Set 3d result (to show flattened cortical map)
        dt3d_stat_R = statBase* np.ones(np.prod(datasize))
        dt3d_stat_G = statBase* np.ones(np.prod(datasize))
        dt3d_stat_B = statBase* np.ones(np.prod(datasize))
        dt3d_stat_R[tvoxels-1] = stats_color_R
        dt3d_stat_G[tvoxels-1] = stats_color_G
        dt3d_stat_B[tvoxels-1] = stats_color_B
        dt3d_stat_R = dt3d_stat_R.reshape([datasize[2], datasize[1], datasize[0]]) # reshape
        dt3d_stat_G = dt3d_stat_G.reshape([datasize[2], datasize[1], datasize[0]]) 
        dt3d_stat_B = dt3d_stat_B.reshape([datasize[2], datasize[1], datasize[0]]) 
        # Append color channels
        color_stats = []
        color_stats.append(dt3d_stat_R)
        color_stats.append(dt3d_stat_G)
        color_stats.append(dt3d_stat_B)

        ### Sig
        color_sig = list()
        if get_dt3d_sig == True:
            sigTvoxelsInds = sigs_[roi_id] # (!) matlab order
            # Set 3d result (to show flattened cortical map)
            dt3d_sig_R = statBase* np.ones(np.prod(datasize))
            dt3d_sig_G = statBase* np.ones(np.prod(datasize))
            dt3d_sig_B = statBase* np.ones(np.prod(datasize))
            dt3d_sig_R[tvoxels[sigTvoxelsInds-1]-1] = tcolor[0] # tvoxels-1: python order
            dt3d_sig_G[tvoxels[sigTvoxelsInds-1]-1] = tcolor[1]
            dt3d_sig_B[tvoxels[sigTvoxelsInds-1]-1] = tcolor[2]
            dt3d_sig_R = dt3d_sig_R.reshape([datasize[2], datasize[1], datasize[0]]) # reshape
            dt3d_sig_G = dt3d_sig_G.reshape([datasize[2], datasize[1], datasize[0]]) 
            dt3d_sig_B = dt3d_sig_B.reshape([datasize[2], datasize[1], datasize[0]]) 
            # Append color channels
            color_sig.append(dt3d_sig_R)
            color_sig.append(dt3d_sig_G)
            color_sig.append(dt3d_sig_B)

        dt3d_stat_.append(color_stats)
        dt3d_sig_.append(color_sig)
    
    ### Integrating colors of 3d volumes for stat and sig
    # Stats (F value, T value, Corrcoef ...)
    dt3d_stats_show_R = np.zeros(dt3d_stat_[0][0].shape)
    dt3d_stats_show_G = np.zeros(dt3d_stat_[0][1].shape)
    dt3d_stats_show_B = np.zeros(dt3d_stat_[0][2].shape)
    for roicnt in range(nStats):
        dt3d_stats_show_R = np.nansum([dt3d_stats_show_R, integrationScale* dt3d_stat_[roicnt][0]], axis=0)
        dt3d_stats_show_G = np.nansum([dt3d_stats_show_G, integrationScale* dt3d_stat_[roicnt][1]], axis=0)
        dt3d_stats_show_B = np.nansum([dt3d_stats_show_B, integrationScale* dt3d_stat_[roicnt][2]], axis=0)
    dt3d_stats_show = [dt3d_stats_show_R, dt3d_stats_show_G, dt3d_stats_show_B]
    # Significance (based on P values)
    if get_dt3d_sig == True:
        dt3d_sig_show_R = np.zeros(dt3d_sig_[0][0].shape)
        dt3d_sig_show_G = np.zeros(dt3d_sig_[0][1].shape)
        dt3d_sig_show_B = np.zeros(dt3d_sig_[0][2].shape)
        for roicnt in range(nStats):
            dt3d_sig_show_R = np.nansum([dt3d_sig_show_R, integrationScale* dt3d_sig_[roicnt][0]], axis=0)
            dt3d_sig_show_G = np.nansum([dt3d_sig_show_G, integrationScale* dt3d_sig_[roicnt][1]], axis=0)
            dt3d_sig_show_B = np.nansum([dt3d_sig_show_B, integrationScale* dt3d_sig_[roicnt][2]], axis=0)
        dt3d_sig_show   = [dt3d_sig_show_R, dt3d_sig_show_G, dt3d_sig_show_B]
    else:
        dt3d_sig_show = []
    
    return dt3d_stats_show, dt3d_sig_show, colors


