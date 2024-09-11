
import scipy.io as sio
import numpy as np
from sklearn.utils.validation import check_random_state
from util import util_dataload as udl
import statsmodels.stats.multitest as multitest
from numpy.linalg import svd, matrix_rank
import scipy.stats as stats
import random




def generate_leave_one_run_out(n_samples, run_onsets, random_state=None,
                               n_runs_out=1):

    # (!!!) This function is an excerpt from Gallant lab's Github code (!!!)
    # https://github.com/gallantlab/voxelwise_tutorials/blob/main/voxelwise_tutorials/utils.py
    
    random_state = check_random_state(random_state)

    n_runs = len(run_onsets)
    
    all_val_runs = np.array(
        [random_state.permutation(n_runs) for _ in range(n_runs_out)])

    all_samples = np.arange(n_samples)
    runs = np.split(all_samples, run_onsets[1:])
    if any(len(run) == 0 for run in runs):
        raise ValueError("Some runs have no samples. Check that run_onsets "
                         "does not include any repeated index, nor the last "
                         "index.")

    for val_runs in all_val_runs.T:
        train = np.hstack(
            [runs[jj] for jj in range(n_runs) if jj not in val_runs])
        val = np.hstack([runs[jj] for jj in range(n_runs) if jj in val_runs])
        yield train, val


def my_zscore(X, mX=[], sX=[]):
    
    if len(mX)==0:
        mX = np.mean(X, axis=0)
    if len(sX)==0:
        sX = np.std(X, axis=0)
    zX = (X - mX)/sX
    
    return mX, sX, zX


def stack_mat(stim, stim_delays):
    
    for i in stim_delays:
        if i == stim_delays[0]: stim_new = np.roll(stim, i, axis=0)
        else: stim_new = np.concatenate([stim_new, np.roll(stim, i, axis=0)], axis=1)
            
    return stim_new


def resp_stim_loader(config, subjectID, featName, movIDs, stim_delays=[2,3,4,5,6], loadResp=True, loadStim=True, show_progress=True):
    
    Resp_ = np.array([])
    Stim_ = np.array([])
    cnt = 0
    for movID in movIDs:
    
        load_items = udl.set_load_items(config, subjectID, movID)
        nRuns = len(load_items['list_path_resp_base'])
        
        # load mask data
        if movID == movIDs[0] and loadResp == True: # run at once
            t_path_mask = load_items['mask_path']#.replace('%s', '{:s}').format(config['dir']['derivative'])
            mask = udl.load_mask_data(t_path_mask)
    
        # for each mov and each run
        for runID in range(nRuns):
    
            cnt = cnt + 1
            samples = load_items['list_samples'][runID]-1 # matlab_order to python_order
            
            t_path_bold = load_items['list_path_resp_base'][runID].format(config['dir']['derivative'])
            t_path_feat = load_items['list_path_feat_base'][runID].format(config['dir']['derivative'], featName)
            
            if show_progress == True:
                print('Loading ...\n(bold) {:s}\n(feat) {:s}\n'.format(t_path_bold, t_path_feat))
            
            # load bold
            if loadResp == True:
                resp = udl.load_bold_data(t_path_bold)
                resp = resp[:, mask] # use only vset_100 tvoxels
                if cnt == 1: Resp_ = resp[samples,:]
                else: Resp_ = np.concatenate([Resp_, resp[samples,:]], axis=0)
    
            # load stim
            if loadStim == True:
                stim = udl.load_stim_data(t_path_feat)
                stim = stack_mat(stim, stim_delays)
                if cnt == 1: Stim_ = stim[samples,:]
                else: Stim_ = np.concatenate([Stim_, stim[samples,:]], axis=0)

    return Resp_, Stim_


def get_resp_stim(config, subjectID, feat_names, movIDs_train_test, stim_delays=[1,2,3,4,5,6]):
    
    movIDs_train = movIDs_train_test['train']
    movIDs_test = movIDs_train_test['test']
    dimIDs = dict() # 
    dimID_onset = 0
    for featID, featName in enumerate(feat_names):

        if featID == 0: # loading resp in the first loop
            Resp_train, stim_train = resp_stim_loader(config, subjectID, featName, movIDs_train, stim_delays=stim_delays)
            Resp_test, stim_test = resp_stim_loader(config, subjectID, featName, movIDs_test, stim_delays=stim_delays)
            Stim_train = stim_train
            Stim_test = stim_test
        else:
            _, stim_train = resp_stim_loader(config, subjectID, featName, movIDs_train, stim_delays=stim_delays, loadResp=False)
            _, stim_test = resp_stim_loader(config, subjectID, featName, movIDs_test, stim_delays=stim_delays, loadResp=False)
            Stim_train = np.concatenate([Stim_train, stim_train], axis=1)
            Stim_test = np.concatenate([Stim_test, stim_test], axis=1)

        dimID_st, dimID_ed = 0, stim_train.shape[1] # stim dimension IDs (e.g. 0–1024)
        del stim_train, stim_test

        # Setting dimension IDs for each feat in the  concatenated vector
        t_dimIDs = list(range(dimID_st+dimID_onset, dimID_ed+dimID_onset))
        dimIDs[featName] = t_dimIDs
        dimID_onset = dimID_onset + dimID_ed # Update dimID_onset
        
        data_out = dict()
        data_out['Resp'] = dict()
        data_out['Resp'] = {'train':Resp_train, 'test':Resp_test}
        data_out['Stim'] = dict()
        data_out['Stim'] = {'train':Stim_train, 'test':Stim_test}
        data_out['dimIDs'] = dict()
        data_out['dimIDs'] = {'ids':dimIDs, 'onset':dimID_onset}
        
    return data_out 


def get_subset_samples_and_onset(n_samples, chunkLen=100, seedno=33):
    
    # Setting shuffle inds for cv (using )　and run ouset
    # shuffle resp and stim (100s chunk)

    random.seed(seedno)
    
    nChunks = np.ceil(n_samples/chunkLen).astype(int)
    sample_chunks = np.array_split(np.arange(n_samples),nChunks)
    rand_chunkIDs = random.sample(list(range(nChunks)), nChunks)
    # rand_sample_chunks
    rand_sample_chunks = [list(sample_chunks[rci]) for rci in rand_chunkIDs]
    rand_sample_chunks = sum(rand_sample_chunks, []) # flatten
    # chunk_onset
    chunk_onset = [len(sample_chunks[rci]) for rci in rand_chunkIDs]
    chunk_onset = [0] + list(np.cumsum(chunk_onset))[:-1]

    # append for subset
    shuffled_samples = rand_sample_chunks
    cv_chunk_onset  = chunk_onset
    
    return rand_sample_chunks, chunk_onset


def get_sig_voxel_inds(ps, alpha=0.05): # Get sig.voxel inds (FDR correction p<0.05)
    
    mask_voxel_inds = np.array( range(len(ps)) )
    nonan_mask = ~np.isnan(ps)
    rejected, ps_correction = multitest.fdrcorrection(ps[nonan_mask], alpha=0.05, method='indep', is_sorted=False)
    sig_voxel_inds = mask_voxel_inds[nonan_mask][rejected]
    ps_correction_new = ps.copy()
    ps_correction_new[nonan_mask] = ps_correction
    
    return ps_correction_new, sig_voxel_inds


def calc_svd(W, centering=True, contribution_thres=0.99):
    
    if centering == True:
        cW = W - W.mean(axis=0) # centering
    print('calc svd ...')
    u, s, vh = svd(cW.transpose(), full_matrices=False)
    print('done.')
    s_inds = list(range(len(s)))
    contribution_ratio = s/s.sum()
    s_inds = np.array(range(len(s)))
    s_inds_en = s_inds[np.cumsum(contribution_ratio)>contribution_thres][0]
    
    return vh[:,0:s_inds_en], s, contribution_ratio, s_inds[0:s_inds_en]


def get_test_results_r2(model, Stim_test, dimIDs, Resp_test, intercept=False):
    
    Stim_in = list()
    for keyname in dimIDs.keys():
        Stim_in.append(Stim_test[ :, dimIDs[keyname] ])

    if intercept == True:
        r2 = model.score(Stim_in, Resp_test, split=False)
        r2_split = None
    else:
        r2 = None
        r2_split = model.score(Stim_in, Resp_test, split=True)
    
    return r2, r2_split

def get_test_results_corr(model, t_Stim_test, Resp_test, dims=[], ps_alpha = 0.05, intercept=True):

    if len(dims)==0:
        dims = np.array(range(t_Stim_test.shape[1]))
    
    if intercept == True:
        p_Resp_test = np.dot(t_Stim_test[:,dims], model.coef_[dims, :]) + model.intercept_
    else:
        p_Resp_test = np.dot(t_Stim_test[:,dims], model.coef_[dims, :])
    rs = list()
    ps = list()
    print('Prediction: Calc rs and ps ...')
    for i in range(0, p_Resp_test.shape[1]):
        r_, p_ = stats.pearsonr( p_Resp_test[:,i], Resp_test[:,i] )
        rs.append(r_)
        ps.append(p_)
    # list to numpy array
    rs = np.array(rs)
    ps = np.array(ps)
    print('done.')
    
    # Get sig.voxel inds (FDR correction p<0.05)
    ps_correction, sig_voxel_inds = get_sig_voxel_inds(ps, alpha=ps_alpha)

    return rs, ps, ps_correction, sig_voxel_inds, ps_alpha

