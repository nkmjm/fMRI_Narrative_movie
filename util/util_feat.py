import numpy as np
import scipy.io as sio
import torch


def set_load_and_save_info(config, annot_name, model_name):

    DT_item = sio.loadmat( config['path']['dataInfo'] )
    
    list_path_text = []
    list_path_feat = []
    for movCnt in range(11):
        nRun = len(DT_item['annot'][0,0]['mov'][0,movCnt]['run'][0])
        for runCnt in range(nRun):
            # setting path_annot, path_feat
            fileBase_text = DT_item['annot'][0,0]['mov'][0,movCnt]['run'][0,runCnt]['file__txt'][0].replace('%s', '{:s}')
            fileBase_feat = DT_item['annot'][0,0]['mov'][0,movCnt]['run'][0,runCnt]['file__feature'][0].replace('%s', '{:s}')
            path_text = fileBase_text.format(config['dir']['derivative'], annot_name).replace('//', '/')
            path_feat = fileBase_feat.format(config['dir']['derivative'], annot_name, model_name).replace('//', '/')
            # append
            list_path_text.append(path_text)
            list_path_feat.append(path_feat)
          
    return list_path_text, list_path_feat


def text_cleaning(line):

    line = line.replace('\n', '')
    line = line.replace('\u3000', '')
    line = line.replace('不明', '..')

    return line


def preproc_line(featName, line): 
    
    # Text cleaning
    stID = line.find(']') + 1
    line = line[stID::]
    line = text_cleaning(line)
    
    # Preprocessing for a specific featName
    if 'speechTranscription' in featName:
        line = line.split('メイン#')[1]
        splt_line = line.split('/')
        del line
        line = []
        for t_line in splt_line:
            if ':' in t_line:
                t_line = t_line.split(':')[1]
            line.append(t_line)
    else:
        line = line.split('/')
    
    return line


def initial_setup_for_model(model_name):

    
    ### GPT2medium_jp
    if 'GPT2medium_jp' in model_name:

        # ref: https://huggingface.co/rinna/japanese-gpt2-medium
        from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModel
        
        tokenizer = AutoTokenizer.from_pretrained("rinna/japanese-gpt2-medium", use_fast=False)
        tokenizer.do_lower_case = True  # due to some bug of tokenizer config loading
        
        model = AutoModel.from_pretrained("rinna/japanese-gpt2-medium", output_hidden_states=True)
        if torch.cuda.is_available():
            model = model.to("cuda")
        model.eval()
            
        ### Check special tokens
        #print('***** special tokens *****')
        #print(tokenizer.special_tokens_map)
        
        model_set = {'model_name':model_name,'model':model, 'tokenizer':tokenizer}
        

    ### GPT2medium_en    
    elif 'GPT2medium_en' in model_name:
        
        # ref: https://huggingface.co/transformers/v2.2.0/pretrained_models.html
        # https://huggingface.co/transformers/v4.2.2/model_doc/gpt2.html
        # -------------
        # 24-layer, 1024-hidden, 16-heads, 345M parameters.
        # OpenAI’s Medium-sized GPT-2 English model
        
        from transformers import GPT2Tokenizer, GPT2Model, GPT2LMHeadModel
        
        tokenizer = GPT2Tokenizer.from_pretrained("gpt2-medium") # gpt2-large, gpt2-medium(24 layers), gpt2(12 layers)
        model = GPT2Model.from_pretrained('gpt2-medium',  output_hidden_states=True) # gpt2-large, gpt2-medium, gpt
        if torch.cuda.is_available():
            model = model.to("cuda")
        model.eval()
        
        ### Check special tokens
        #print('***** special tokens *****')
        #print(tokenizer.special_tokens_map)
        
        model_set = {'model_name':model_name,'model':model, 'tokenizer':tokenizer}
        
    return model_set


def merge_hidden_states(hidden_states, merge_type):
    
    if merge_type == 'eos':
        t_stim = hidden_states[0,-1,:].unsqueeze(0)
    elif merge_type == 'sum':
        t_stim = hidden_states[0,:,:].sum(axis=0).unsqueeze(0)
    elif merge_type == 'mean':
        t_stim = hidden_states[0,:,:].mean(axis=0).unsqueeze(0) 
        
    return t_stim


def feature_extraction_using_GPT2(model_set, text):
    
    model_name = model_set['model_name']

    # init set up
    model = model_set['model']
    tokenizer = model_set['tokenizer']
    items_gptparams = model_name.split('_')
    merge_type = items_gptparams[3]
    layerID = int(items_gptparams[2].split('layer')[1])

    text = tokenizer.bos_token + text + tokenizer.eos_token # add bos/ eos_token
    encoded_input = tokenizer(text, return_tensors='pt')
    encoded_input = encoded_input.to('cuda')

    with torch.no_grad():
        output = model(**encoded_input)

    hidden_states = output[2][layerID] # [0]:embedding, [1]layer1, [2]layer2, ..., [12]layer12
    feat = merge_hidden_states(hidden_states, merge_type)

    return feat


def feature_extraction_using_GPT2__annotation(model_set, texts, show_log=False):
    
    model_name = model_set['model_name']

    # init set up
    model = model_set['model']
    tokenizer = model_set['tokenizer']
    items_gptparams = model_name.split('_')
    merge_type = items_gptparams[3]
    layerID = int(items_gptparams[2].split('layer')[1])
    
    # check the structure of "texts"
    if (isinstance(texts, list)* isinstance(texts[0], list)==1) and (isinstance(texts[0][0], list)==False):
        progress = True
    else:
        print('The structure of "texts" should be [[sentences 1, sentences 2, ...], [sentences k, ...], ...]')
        progress = False
        features = None
        return featrures
    

    # Feature extraction
    if progress == True:
        
        print('********** Feature extraction using {:s} **********'.format(model_name))
        
        for li_ti, li_text in enumerate(texts):

            if np.mod(li_ti, 100)==0: print('Now processing for line {:d}–{:d}'.format(li_ti, li_ti+99))

            for ti, text in enumerate(li_text):

                # cleaning for blank 
                if text == '..': text = ''
                elif text == '.': text = ''

                if show_log == True:
                    print('(line ID:{:d}):{:s}'.format(li_ti, text))

                #text = tokenizer.bos_token + text + tokenizer.eos_token # add bos/ eos_token
                encoded_input = tokenizer(text, return_tensors='pt')
                encoded_input = encoded_input.to('cuda')

                with torch.no_grad():
                    output = model(**encoded_input)

                hidden_states = output[2][layerID] # [0]:embedding, [1]layer1, [2]layer2, ..., [12]layer12

                # concatenate tmp. feat
                tt_feat = merge_hidden_states(hidden_states, merge_type)
                #tt_feat = tt_feat.to('cpu').detach().numpy().copy()

                if ti == 0: t_feat = tt_feat
                else: t_feat = t_feat + tt_feat

            t_feat = t_feat/len(li_text)

            if li_ti == 0:  features = t_feat
            else:  features = torch.cat([features, t_feat], axis=0)

        features = features.to('cpu').detach().numpy().copy()
        torch.cuda.empty_cache()

        return features

        