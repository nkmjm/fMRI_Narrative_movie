import numpy as np
import h5py
import scipy.io as sio


# Prepare 'dt':
# with open('config__drama_data.yaml', 'r') as f_yml:
#   config = yaml.safe_load(f_yml)
# dt = sio.loadmat(config['path']['loadItems']) # subject

'''
def subjectNameFig2subjectName(config, subjectNameFig):
    
    # Converting subjectNameFig (e.g. S01) to subjectName(org) (e.g. DM01).
    # subjectID will be also returned.

    subjectNames      = config['subjectInfo']['names']
    subjectNamesFig = config['subjectInfo']['names_fig']
    tmpCode = np.where(np.array(subjectNamesFig) == subjectNameFig)[0][0]
    subjectID = config['subjectInfo']['loadIntems_no'][tmpCode]
    subjectName = subjectNames[tmpCode]
    
    return subjectID, subjectName
    
 '''

def set_load_items(config, subjectID=0, movID=0): # annotType = '__annotType__'

    dt = sio.loadmat( config['path']['dataInfo'] )
    
    nSubject = len(dt['subject'][0])
    nMov = len(dt['subject'][0][subjectID]['mov'][0])
    subjectName = dt['subject'][0][subjectID]['name'][0]
    mask_base = dt['subject'][0][subjectID]['brain'][0]['maskfile'][0][0].replace('%s', '{:s}')
    mask_path = mask_base.format(config['dir']['derivative'])
    datasize = dt['subject'][0][subjectID]['brain'][0]['datasize'][0][0]
    title = dt['subject'][0][subjectID]['mov'][0][movID]['title'][0]
    nRuns = len(dt['subject'][0][subjectID]['mov'][0][movID]['run'][0])
    
    list_f_resp_base = list()
    list_samples = list()
    list_duration = list()
    list_f_feat_base = list()
    #list_f_motioncorrection_base = list()
    for runID in range(nRuns):
        
        f_resp_base = dt['subject'][0][subjectID]['mov'][0][movID]['run'][0][runID]['file'][0].replace('%s', '{:s}')
        samples = dt['subject'][0][subjectID]['mov'][0][movID]['run'][0][runID]['samples'][0]
        duration = dt['subject'][0][subjectID]['mov'][0][movID]['run'][0][runID]['duration'][0][0] # including dummy (the first 20 samples)
        f_feat_base = dt['annot'][0][0]['mov'][0][movID]['run'][0][runID]['file__feature'][0].replace('%s_%s', '%s').replace('%s', '{:s}')
        #f_motioncorrection_base = f_resp_base.replace('trendRemoved-', 'motionCorrected').replace('%s/'+ subjectName, '%s')
        
        list_f_resp_base.append(f_resp_base)
        list_samples.append(samples)
        list_duration.append(duration)
        list_f_feat_base.append(f_feat_base)
        #list_f_motioncorrection_base.append(f_motioncorrection_base)
        
    load_items = dict()
    load_items = {'nSubject':nSubject, 'nMov':nMov, 'subjectName':subjectName, \
                  'dir_derivative':config['dir']['derivative'], 'mask_path':mask_path, 'datasize':datasize, 'title':title, 'nRuns':nRuns, \
                  'list_path_resp_base':list_f_resp_base, 'list_samples':list_samples, 'list_duration':list_duration, 'list_path_feat_base':list_f_feat_base}

    return load_items

def load_mc_data(path_mc):
    with h5py.File(path_mc,'r') as f:
        # check keys
        #for k in f:
        #    print(k)
        dt_trs = np.array(f['trs'])
    return dt_trs

def load_bold_data(path_bold):
    with h5py.File(path_bold,'r') as f:
        bold = np.array(f['dataDT'])
    return bold

def load_stim_data(path_annot):
    dt = sio.loadmat(path_annot)
    return dt['stim']

def load_mask_data(path_mask):
    dt_msk = sio.loadmat(path_mask)
    mask = dt_msk['tvoxels'].squeeze() - 1 # matlab --> python order
    return mask

def get_datasize(config, subjectID):   
    items = set_load_items(config, subjectID)
    datasize = items['datasize']
    return datasize
    

### get  items for data loading

def get_subjectID_from_subjectName(config, subjectName):
    subjectID = np.where(np.array(config['subjectInfo']['names']) == subjectName)[0][0]
    return subjectID

def get_movIDs_from_movTitleIDs(movtitleIDs_): # Get movIDs from movTitleIDs
    movIDs = list()
    titleLabels = np.array(config['movieInfo']['titleLabels'])
    for mti in movtitleIDs_:
        # Convert movIDs to movTitleIDs
        t_movIDs = np.where(titleLabels==mti)[0]
        movIDs = movIDs + list(t_movIDs)
    return movIDs

def get_train_test_movIDs(config, movietitle_test):
    
    movietitles_all = list( np.unique(config['movieInfo']['titleNames']) )
    print( 'Available movie titles: {:s}'.format(',  '.join(movietitles_all)) )
    
    movIDs_test = np.where(  np.array(config['movieInfo']['titleNames']) == movietitle_test  )[0]
    movIDs_train = np.array(  range(len(config['movieInfo']['titleLabels']))  )
    movIDs_train = np.delete(movIDs_train, movIDs_test)
    movietitles_train = list(np.unique( np.array( config['movieInfo']['titleNames'] )[movIDs_train] ))
    
    print('Test movie title:  "{:s}" '.format(movietitle_test))
    print('Train movie titles:  "{:s}" '.format( ', '.join(movietitles_train) ))
    
    return movIDs_train, movIDs_test

def get_n_samples_from_movIDs(config, movIDs_):
    n_samples = 0
    for mi in movIDs_:
        tmp_items = set_load_items(config, movID=mi)
        nRuns = len(tmp_items['list_samples'])
        for runID in range(nRuns):
            t_samples = tmp_items['list_samples'][runID]
            n_samples = n_samples + len(t_samples)
            del t_samples
    return n_samples


def get_featNames_label(config, feat_names):
    
    featNames_label = ''
    for i, feat_name in enumerate(feat_names):

        annotID = np.where(np.array(config['annotInfo']['annotTypes']) == feat_name.split('_')[0] )[0][0]
        annot_abbr = config['annotInfo']['abbreviations_annot'][annotID]
        model_abbr = config['annotInfo']['abbreviations_model'][annotID]
        model_name = config['annotInfo']['modelNames'][annotID]
        model_subname = feat_name.split(model_name)[1][1::]
        subnameID = np.where(np.array(config['ridgeInfo']['modelSubNames'][model_name]) == model_subname)[0][0]
        layer_abbr = config['ridgeInfo']['modelSubNames']['abbreviations'][subnameID]

        t_feat_name = annot_abbr + '_' + model_abbr + '_' + layer_abbr
        featNames_label = featNames_label + t_feat_name +  '__'
    
    featNames_label = featNames_label[:-2] # remove '__'
    
    return featNames_label


