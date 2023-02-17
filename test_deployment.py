import re
import regex
import torch
import transformers
import editdistance
import pandas as pd
from tqdm import tqdm
from utils import move_to_device
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer



def parse_t5(text):
    tmp = [i for i in text.split('\n') if len(i) >0]
    new = []
    for stuff in tmp:
        item = re.sub(r'\([^)]*\)', '', stuff).strip().lower()
        if item not in (
            'no movie titles mentioned.', 
            'none', 
            'no movie titles are mentioned in the text.',
            'no movie titles appear in the text.',
        'no movies mentioned in the text.',
        'no movies mentioned in the text.',
        'none.'):
            try:
                if item[:1].isnumeric():
                    new.append(item.split('.')[1].strip())
                elif item[:2] == '- ':
                    new.append(item[2:].strip())
                else:
                    new.append(item)
            except:
                pass
    new = [i.replace('-','') for i in new if len(i)>1]
    newer = []
    for item in new:
        for sub in item.split(','):
            newer.append(sub.lower().strip())
    out = [i.rstrip("""'""").lstrip("""'""") for i in newer]
    out = [i for i in out if len(i) > 0]
    return set(out)

def eval_one_pair(true, pred):
    inter = pred.intersection(true)
    if len(true) == len(pred) == 0:
        return 1,1,1
    if len(inter) == 0:
        return 0,0,0
    p = len(inter)/len(pred)
    r = len(inter)/len(true)
    f = 2*p*r/ (p+r)

    return f, p, r

def fuzzy_align_towards(src, reference, edit_threshold=2):
    """
    given two sets of string, src and referece,
    for any string in src that is fuzzily similar to the first string found in reference,
        we replace the string in src using the similar string in reference
    """
    out = set()
    for word in src:
        sub_is_found = False
        for ref in reference:
            if editdistance.eval(word, ref) <= edit_threshold:
                out.add(ref)
                sub_is_found = True
                break
        if not sub_is_found:
            out.add(word)
    return out

def approximately_in(pattern, document, edit_threshold):
    rule = '('+pattern+'){e<='+str(edit_threshold)+'}'
    found = regex.findall(rule, document, overlapped=False)
    return len(found) > 0

def eval_t5_results(
    t5_responses,
    anno,
    eval_df
):
    
    gold = dict(zip(eval_df['Input.selftext'], eval_df['Answer.crowd_input']))
    gold_mapping1 = {}
    for text, label in gold.items():
        if label == ',':
            label = set()
        else:
            label = set([i.strip() for i in str(label).strip().split(',') if len(i)>0])
            label = set([i.replace('*','').rstrip('.') for i in label])
        gold_mapping1[text.lower()] = label
        
    results = dict(zip([i.lower() for i in anno['Input.selftext']], [parse_t5(text) for text in t5_responses]))
    
    extractive_results = {}
    for k,v in results.items():
        valid_match = [i for i in v if i.lower() in k.lower()]
        extractive_results[k] = set(valid_match)
        
    F, P, R = [], [], []
    
    for stuff in extractive_results:
        text = stuff.replace('\\n', '\n')
        extracted = set([i.lower() for i in results[text]])
        true = set([i.lower() for i in gold_mapping1[text]])
        extracted = set([i for i in extracted if approximately_in(i.lower(), text.lower(), 2)])
        extracted = fuzzy_align_towards(extracted, true, edit_threshold=2)
        
        
        # print(extracted, true)

        f, p , r = eval_one_pair(true, extracted)
        # print(f, p, r)
        F.append(f)
        P.append(p)
        R.append(r)
    import numpy as np
    print('F;P;R: ', np.mean(F), np.mean(P), np.mean(R))


def main(args):
    tokenizer = AutoTokenizer.from_pretrained(args.base_model_name_or_path)
    model = AutoModelForSeq2SeqLM.from_pretrained(args.base_model_name_or_path).cuda()
    anno = pd.read_csv(args.eval_df_path)
    anno['Input.selftext'] = \
    ['Perform entity recognition and extract movie titles from the following text: \n'+i for i in anno['Input.selftext']]

    responses = []
    with torch.no_grad():
        for text in tqdm(anno['Input.selftext'].astype(str)):
            inputs = tokenizer(text, return_tensors="pt")
            inputs = move_to_device(dict(inputs), torch.device('cuda'))
            outputs = model.generate(**inputs)
            outputs = tokenizer.batch_decode(outputs, skip_special_tokens=True)
            output = outputs[0]
            responses.append(output)

    anno['responses'] = responses
    anno = anno.sort_values('Input.selftext')
    responses = anno['responses']

    res = eval_t5_results(
        t5_responses = responses,
        anno = anno,
        eval_df = anno
    )

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Generator Model')
    parser.add_argument('--eval_df_path', type=str, default=None,
                        help='load eval df')
    parser.add_argument('--base_model_name_or_path', type=str, default='./checkpoints/generative_movie_title_ner',
                        help='load eval df')
    args = parser.parse_args()

    # from easydict import EasyDict as edict
    # args = dict(
    #     eval_df_path='./data/turker_responses/turk0.csv', 
    #     base_model_name_or_path='./checkpoints/generative_movie_title_ner/checkpoint-900'
    # )
    # args = edict(args)
    # main(args)

    # args = dict(
    #     eval_df_path='./data/turker_responses/turk1.csv', 
    #     base_model_name_or_path='./checkpoints/generative_movie_title_ner/checkpoint-900'
    # )
    # args = edict(args)
    # main(args)

    # args = dict(
    #     eval_df_path='./data/turker_responses/turk2.csv', 
    #     base_model_name_or_path='./checkpoints/generative_movie_title_ner/checkpoint-900'
    # )
    # args = edict(args)

    main(args)

