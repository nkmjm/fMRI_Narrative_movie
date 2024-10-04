import pickle
import scipy.io as sio
import scipy.stats
import numpy as np
from util import util_pycortex as utlpy
import cortex

def set_show_params(locName):
    
    if locName == 'Motor':
        showbias = 0.3
        showcoef = 2
    elif locName == 'MiniGame':
        showbias = 0.2
        showcoef = 4
    elif 'Kathleen' in locName:
        showbias = 0.4
        showcoef = 4
    else:
        showbias = 0.1
        showcoef = 2
        
    return showbias, showcoef


def set_stat_from_weight(weight, sig=[], statType='WTA', with_bias=True):
    
    # weight (size: nDim x nVoxels): the last 2 dims corresponding to [bias, linear-trend]
    # sig (size: nSigVoxels, ): voxelIDs showing significantly high stats

    if with_bias == True:
        weight = weight[0:-2,:]

    nDims = weight.shape[0]
    stats_w = []
    sig_w = []

    if statType == 'norm':       
        stats_w_array = norm_weight(weight)
        for i in range(stats_w_array.shape[0]):
            stats_w.append(stats_w_array[i])
        for i in range(nDims):
            if len(sig) > 0:
                sig_w.append(sig)
        
    elif statType == 'WTA':     
        for i in range(nDims):
            base_stat = 0* np.ones(weight.shape[1])
            base_stat[weight.argmax(axis=0)==i] = 1
            stats_w.append(base_stat)
            if len(sig) > 0:
                sig_w.append(sig)

    return stats_w, sig_w

    
def set_show_data_from_weight(locName, sigVoxels, weight):

    if 'Kathleen' in locName or locName == 'Motor':
        stat, sig = set_stat_from_weight(weight, sig=sigVoxels, statType='WTA')
    elif 'MiniGame' in locName:
        stat, sig = set_stat_from_weight(weight, sig=sigVoxels, statType='WTA', with_bias=False)
    else:
        stat, sig = set_stat_from_weight(weight, sig=sigVoxels, statType='norm') 
        
    return stat, sig


def set_max_stat(locName, colorType):
    
    if colorType == 'single':
        if locName == 'PPA_OPA_RSC': max_stat = 0.4
        else: max_stat = 0.3
    else:
        max_stat = 0.5
        
    return max_stat


def get_results_visualization_demo(config, subjectName, resultType):
    
    if resultType == 'banded__obj_speech_story':
            
        featName = 'object_gpt2mjp_l12m__speech_gpt2mjp_l12m__story_gpt2mjp_l12m'
        result_dir = config['dir']['accuracy'].replace('{derivative_dir}', config['dir']['derivative'])
        result_path = '{:s}/accuracy__banded_ridge__sub-{:s}__object_gpt2mjp_l12m__speech_gpt2mjp_l12m__story_gpt2mjp_l12m.pkl'.format(result_dir, subjectName)
        print('Loading file: {:s}'.format(result_path))
        with open(result_path, 'rb') as f:
            dt = pickle.load(f)

        return dt
        
    elif 'loc_' in resultType:
        
        locID = int(resultType.split('_')[1])-1
        locName = config['localizerInfo']['locLabels'][locID]
        result_dir = config['dir']['froi'].replace('{derivative_dir}', config['dir']['derivative'])
        result_file = config['localizerInfo']['fileBase'].format(subjectName, locName)
        result_path = '{:s}/sub-{:s}/{:s}'.format(result_dir, subjectName, result_file)
        
        print('Loading file: {:s}'.format(result_path))
        dt = sio.loadmat(result_path)

        return dt, locName
        
    else:
        print('resultType should be set to "banded__obj_speech_story" or "loc_{no}."')
        
        return None   
        

### Banded ridge
def get_bandedridge_dt3d(dataInfo, dt):
    
    Stats = []
    Sigs = []
    p_thres = 0.01
    featNames = list(dt['results_eachFeat'].keys())
    
    for fi in [0,1,2]:
    
        featname = featNames[fi]
    
        stats_corr = np.squeeze( dt['accs']['corrcoef']['rs'] )
        stats = np.squeeze( dt['accs']['r2']['r2_split'][fi])
        p = np.squeeze( dt['accs']['corrcoef']['result_FDR_correction']['ps_correction'] )
        sigTvoxelsInds = np.array(range(len(p)))[(p<p_thres) * (stats_corr>0) == 1] + 1# matlab order
        
        Stats.append(stats)
        Sigs.append(sigTvoxelsInds)
    
    dt3d_stats_show, dt3d_sig_show, _ = utlpy.get_dt3d_vol(dataInfo, Stats, Sigs, nLaps=1, normData=False, integrationScale=1)
    showType = 'stats' # 'stats' 'sigTvoxels'
    
    sig_mask = np.sum(dt3d_sig_show, axis=0)
    sig_mask[sig_mask>0] = 1
    mask = sig_mask #None

    return dt3d_stats_show, dt3d_sig_show, mask, showType


### functional ROIs
def get_froi_dt3d(dataInfo, dt, locName):

    ### Loading result items
    t_roi_names = dataInfo['locNames']
    t_plot_types = dataInfo['plotTypes']
    isroiid = np.where(np.array(t_roi_names)==locName)[0][0]
    plot_type = t_plot_types[isroiid]
    
    ### Setting stat, sigTvoxelsInds, ...
    p_thres = 0.05
    statName = dt['param'][0][0]['statName'][0]
    stats = np.squeeze( dt['result'][0][0]['stats'] )
    p = np.squeeze( dt['result'][0][0]['p'] )
    sigTvoxelsInds = np.squeeze( dt['result'][0][0]['sigTvoxelsInds'] )
    
    ### Computing sig FDR
    p_new = np.nan* np.ones(p.shape)
    p_new[~np.isnan(p)] = scipy.stats.false_discovery_control(p[~np.isnan(p)], method='by')
    sigTvoxelsInds_FDR = np.array( range(len(p)) )[p_new < p_thres] + 1 # matlab order
    
    ### plot_type 'weight': Setting weight
    if plot_type == 'weight':
        weight = dt['result']['weight'][0][0]
    if locName == 'MiniGame':
        tmpdelay = dt['result'][0][0]['param'][0][0]['tempdelay'][0]
        nDim, nSmp = weight.shape
        weight = weight.reshape([int(nDim/len(tmpdelay)), len(tmpdelay), nSmp])
        weight = np.nanmean(weight, axis=1).squeeze()
    
    ### Setting show items
    if plot_type == 'weight':
        
        showbias, showcoef = set_show_params(locName)
        stat_w, sig = set_show_data_from_weight(locName, sigTvoxelsInds_FDR, weight)
        dt3d_stats_show, _, colors = utlpy.get_dt3d_vol(dataInfo, stat_w, sig, nLaps=1)
        _, dt3d_sig_show, _ = utlpy.get_dt3d_vol(dataInfo, [stats], [sigTvoxelsInds_FDR], nLaps=1)
        
        showType = 'stats'
        mask = np.mean(dt3d_sig_show, axis=0)* showcoef + showbias
        mask[mask<0] = 0
        mask[mask>1] = 1

        colorType = 'rgb'
        
    else:
    
        # reshape to 3d data
        dt3d_stats_show, dt3d_sig_show, colors = utlpy.get_dt3d_vol(dataInfo, [stats], [sigTvoxelsInds_FDR], nLaps=1)
    
        showType = 'stats' # 'stats'    
        mask = None
        colorType = 'single'

    return dt3d_stats_show, dt3d_sig_show, mask, showType, colorType


def pycortex_visualization(dataInfo, dt3d_stats_show, dt3d_sig_show, \
                           showType = 'stats', colorType = 'single', max_stat = 0.5, \
                           mask = [], saveFigName = [], **kwargs):

    subjectNamePycortex = dataInfo['subjectNamePycortex']
    dataSetName = dataInfo['dataSetName']

    ### Show 'vol' using pycortex
    
    if showType == 'stats':
        vol1 = dt3d_stats_show[0] #dt3d_stat_rois[2][0]#
        vol2 = dt3d_stats_show[1] #dt3d_stat_rois[2][1]
        vol3 = dt3d_stats_show[2] #dt3d_stat_rois[2][2]#
    elif showType == 'sigTvoxels':
        vol1 = dt3d_sig_show[0]
        vol2 = dt3d_sig_show[1]
        vol3 = dt3d_sig_show[2]
    
    ### data volume
    
    if colorType == 'single':
        dv = cortex.dataset.Volume(vol1, subject=subjectNamePycortex, xfmname=dataSetName)
        dv.vmax = max_stat
        dv.vmin = 0 
        
    elif colorType == 'rgb':
        if len(mask) != 0: dv = cortex.dataset.VolumeRGB(vol1, vol2, vol3, subject=subjectNamePycortex, xfmname=dataSetName, alpha=mask)
        else: dv = cortex.dataset.VolumeRGB(vol1, vol2, vol3, subject=subjectNamePycortex, xfmname=dataSetName)
        dv.red.vmax = max_stat # 1
        dv.red.vmin = 0
        dv.green.vmax = max_stat  # 1
        dv.green.vmin = 0
        dv.blue.vmax = max_stat  # 1
        dv.blue.vmin = 0
        
    ### Show results
    # **kwargs: 
    # roi_list=my_roi_list, with_rois=True, with_colorbar=True, colorbar_location=(.4, .8, .2, .04)
    # with_curvature=True, curvature_threshold=0.5, curvature_contrast=0.1, curvature_brightness=0.6
    
    _ = cortex.quickflat.make_figure(dv, thick=10, pixelwise=True, sampler='nearest', nanmean=True, **kwargs) 

    if len(saveFigName)>0: # save png
        cortex.quickflat.make_png(saveFigName, dv, thick=10,  pixelwise=True, sampler='nearest', nanmean=True, **kwargs) 



