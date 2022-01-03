#!/usr/bin/env python
# coding: utf-8

# # Python codes to read in annotation and implement NER

# In[ ]:


import glob
import xml.etree.ElementTree as ET
import spacy
import random
import en_core_web_sm
from pathlib import Path


# ## Read in annotated XML files

# In[ ]:


## function to collect entities into training data
def load_ent(datafile, filename):
    # read in xml file
    tree = ET.parse('C:/Users/jim2012/Documents/Essure MAUDE/NLP full data/'+ datafile + '/' + filename)
    root = tree.getroot()

    # passing on text
    text = root[0].text
    # labeled content
    entities = []

    for child in root[1]:
        # do not include link
        if child.tag not in ('LINKTAG'):
            # get the start and end point of the label
            start = int(child.attrib['spans'].split('~')[0])
            end = int(child.attrib['spans'].split('~')[1])
            entities.append((start, end, child.tag))

    #append text and labeled entites, return the list
    training_doc = []
    training_doc.append((text, {"entities" : entities}))
    return training_doc


# In[ ]:


# extract all training data
train_filelist = glob.glob('C:/Train_annotate/*')

filenames = []
for filename in train_filelist:
    filenames.append((filename.split("\\", 1)[1]))
    
TRAIN_ENT = []
datafile='Train_annotate'
for filename in filenames:
    training_doc = load_ent(datafile, filename)
    TRAIN_ENT.extend(training_doc)
    
# extract all testing data
test_filelist = glob.glob('C:/Test_annotate/*')

filenames = []
for filename in test_filelist:
    filenames.append((filename.split("\\", 1)[1]))

TEST_ENT = []
datafile='Test_annotate'
for filename in filenames:
    testing_doc = load_ent(datafile, filename)
    TEST_ENT.extend(testing_doc)


# ## Model training

# In[ ]:


# set seed
random.seed(123)

# use an established English language model
nlp = en_core_web_sm.load()

# remove original ner
if "ner" in nlp.pipe_names:
    nlp.remove_pipe("ner")

ner = nlp.create_pipe('ner')
nlp.add_pipe(ner, last=True)

# add labels
for _, annotations in TRAIN_ENT:
    for ent in annotations.get('entities'):
        ner.add_label(ent[2])
        
# get names of other pipes to disable them during training
other_pipes = [pipe for pipe in nlp.pipe_names if pipe != 'ner']  

# updating model
ner = nlp.get_pipe("ner")

# add labels
for _, annotations in TRAIN_ENT:
    for ent in annotations.get('entities'):
        ner.add_label(ent[2])
        
# get names of other pipes to disable them during training
other_pipes = [pipe for pipe in nlp.pipe_names if pipe != 'ner']    

# only train NER
with nlp.disable_pipes(*other_pipes):  
    optimizer = nlp.begin_training()
    for itn in range(1000):
        print("Starting iteration " + str(itn))
        random.shuffle(TRAIN_ENT)
        losses = {}
        for text, annotations in TRAIN_ENT:
            nlp.update(
                [text],  # batch of texts
                [annotations],  # batch of annotations
                drop=0.2,  # dropout - make it harder to memorize data
                sgd=optimizer,  # callable to update weights
                losses=losses)
        print(losses)


# In[ ]:


# save model to output directory
output_dir = Path('C:/nlp_model')
nlp.to_disk(output_dir)


# ## Assessing NLP performance

# In[ ]:


### Exact definition
## define function to assess precision and recall
def ner_eval(ner_model, examples, ent_type):
    # set to 0 to begin, fp and fn set to 1e-8 to avoid devision by 0
    tp = 0
    fp = 1e-8
    fn = 1e-8
    
    desire_list = ['SOURCE','TIMING','PROCESS','SYMPTOM','PROCEDURE','COMPLICATION']

    for text, annotation in examples:
        # get all entities from original labeling and predicted labeling
        ents_all = annotation['entities']
        pred_all = ner_model(text)
        ents_pred_all = []
        for ent in pred_all.ents: 
            ents_pred_all.append((ent.start_char, ent.end_char, ent.label_))
        
        if ent_type == 'ALL':
            ents_gold = ents_all
            ents_pred = ents_pred_all
        elif ent_type == 'DESIRE':
            # get six desired entities from original labeling and predicted labeling   
            ents_gold = [ent for ent in ents_all if ent[2] in desire_list]
            ents_pred = [ent for ent in ents_pred_all if ent[2] in desire_list]
        else:
            # get specific entities from original labeling and predicted labeling
            ents_gold = [ent for ent in ents_all if ent[2] == ent_type]
            ents_pred = [ent for ent in ents_pred_all if ent[2] == ent_type]

        # calculate tru positive, false positive, and false negative
        tp += len(set(ents_gold) & set(ents_pred))
        fp += len(set(ents_pred) - set(ents_gold))
        fn += len(set(ents_gold) - set(ents_pred))

    # calculate precision, recall, and fscore
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    fscore = 2 * (precision * recall) / (precision + recall + 1e-8)
    
    return({'precision': precision, 'recall': recall, 'fscore': fscore})


# In[ ]:


# precision and recall on testing data
for ent_type in ['ALL', 'DESIRE',                  'SOURCE', 'TIMING', 'PROCESS', 'SYMPTOM', 'PROCEDURE', 'COMPLICATION', 'DEVICE', 'LOCATION', 'CERTAINTY']:
    test_eva = ner_eval(nlp, TEST_ENT, ent_type)
    print(ent_type, "labels: ", test_eva)


# In[ ]:


### Lenient defintion: allow overlap
## define function to assess precision and recall
def ner_eval_lenient(ner_model, examples, ent_type):
    # set to 0 to begin, fp and fn set to 1e-8 to avoid devision by 0
    tp = 0
    fp = 1e-8
    fn = 1e-8
    
    desire_list = ['SOURCE','TIMING','PROCESS','SYMPTOM','PROCEDURE','COMPLICATION']

    for text, annotation in examples:
        # get all entities from original labeling and predicted labeling
        ents_all = annotation['entities']
        pred_all = ner_model(text)
        ents_pred_all = []
        for ent in pred_all.ents: 
            ents_pred_all.append((ent.start_char, ent.end_char, ent.label_))
        
        if ent_type == 'ALL':
            ents_gold = ents_all
            ents_pred = ents_pred_all
        elif ent_type == 'DESIRE':
            # get six desired entities from original labeling and predicted labeling   
            ents_gold = [ent for ent in ents_all if ent[2] in desire_list]
            ents_pred = [ent for ent in ents_pred_all if ent[2] in desire_list]
        else:
            # get specific entities from original labeling and predicted labeling
            ents_gold = [ent for ent in ents_all if ent[2] == ent_type]
            ents_pred = [ent for ent in ents_pred_all if ent[2] == ent_type]

        # find overlapped ones
        ents_match_gold = []
        ents_match_pred = []
        
        for gold in ents_gold:
            for pred in ents_pred:
                if max(gold[1], pred[1]) - min(gold[0], pred[0]) < (gold[1] - gold[0]) + (pred[1] - pred[0]):
                    ents_match_gold.append(gold)
                    ents_match_pred.append(pred)
            
        # calculate tru positive, false positive, and false negative
        tp += len(ents_match_gold)
        fp += len(set(ents_pred) - set(ents_match_pred))
        fn += len(set(ents_gold) - set(ents_match_gold))

    # calculate precision, recall, and fscore
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    fscore = 2 * (precision * recall) / (precision + recall + 1e-8)
    
    return({'precision': precision, 'recall': recall, 'fscore': fscore})


# In[ ]:


# precision and recall on testing data
for ent_type in ['ALL', 'DESIRE',                  'SOURCE', 'TIMING', 'PROCESS', 'SYMPTOM', 'PROCEDURE', 'COMPLICATION', 'DEVICE', 'LOCATION', 'CERTAINTY']:
    test_eva = ner_eval_lenient(nlp, TEST_ENT, ent_type)
    print(ent_type, "labels: ", test_eva)

