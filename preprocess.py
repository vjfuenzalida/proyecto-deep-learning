import os
import random
from collections import Counter, defaultdict

import numpy as np
import pandas as pd
import spacy
import tensorflow as tf
from tqdm import tqdm

import ujson as json

nlp = spacy.blank("en")

def word_tokenize(sent):
    doc = nlp(sent)
    return [token.text for token in doc]

def convert_idx(text, tokens):
    current = 0
    spans = []
    for token in tokens:
        current = text.find(token, current)
        if current < 0:
            print("Token {} cannot be found".format(token))
            raise Exception()
        spans.append((current, current + len(token)))
        current += len(token)
    return spans

def process_file(filename, data_type, word_counter, char_counter):
    print("Generating {} examples...".format(data_type))
    examples = []
    eval_examples = {}
    total = 0
    with open(filename, "r") as fh:
        source = json.load(fh)
        for article in tqdm(source["data"]):
            for para in article["paragraphs"]:
                context = para["context"].replace(
                    "''", '" ').replace("``", '" ')
                context_tokens = word_tokenize(context)
                context_chars = [list(token) for token in context_tokens]
                spans = convert_idx(context, context_tokens)
                for token in context_tokens:
                    word_counter[token] += len(para["qas"])
                    for char in token:
                        char_counter[char] += len(para["qas"])
                for qa in para["qas"]:
                    total += 1
                    ques = qa["question"].replace(
                        "''", '" ').replace("``", '" ')
                    ques_tokens = word_tokenize(ques)
                    ques_chars = [list(token) for token in ques_tokens]
                    for token in ques_tokens:
                        word_counter[token] += 1
                        for char in token:
                            char_counter[char] += 1
                    y1s, y2s = [], []
                    answer_texts = []
                    for answer in qa["answers"]:
                        answer_text = answer["text"]
                        answer_start = answer['answer_start']
                        answer_end = answer_start + len(answer_text)
                        answer_texts.append(answer_text)
                        answer_span = []
                        for idx, span in enumerate(spans):
                            if not (answer_end <= span[0] or answer_start >= span[1]):
                                answer_span.append(idx)
                        y1, y2 = answer_span[0], answer_span[-1]
                        y1s.append(y1)
                        y2s.append(y2)
                    example = {"context_tokens": context_tokens, "context_chars": context_chars,
                               "ques_tokens": ques_tokens,
                               "ques_chars": ques_chars, "y1s": y1s, "y2s": y2s, "id": total}
                    examples.append(example)
                    eval_examples[str(total)] = {
                        "context": context, "spans": spans, "answers": answer_texts, "uuid": qa["id"]}
        random.shuffle(examples)
        print("{} questions in total".format(len(examples)))
    return examples, eval_examples

def get_embedding(counter, data_type, limit=-1, emb_file=None, size=None, vec_size=None):
    print("Generating {} embedding...".format(data_type))
    embedding_dict = {}
    filtered_elements = [k for k, v in counter.items() if v > limit]
    if emb_file is not None:
        with open(emb_file, "r", encoding="utf-8") as fh:
            for line in tqdm(fh, total=size):
                array = line.split()
                word = "".join(array[0:-vec_size])
                vector = list(map(float, array[-vec_size:]))
                if word in counter and counter[word] > limit:
                    embedding_dict[word] = vector
        print("{} / {} tokens have corresponding {} embedding vector".format(
            len(embedding_dict), len(filtered_elements), data_type))
    else:
        assert vec_size is not None
        for token in filtered_elements:
            embedding_dict[token] = [np.random.normal(
                scale=0.1) for _ in range(vec_size)]
        print("{} tokens have corresponding embedding vector".format(
            len(filtered_elements)))

    NULL = "--NULL--"
    OOV = "--OOV--"
    token2idx_dict = {token: idx for idx, token in enumerate(embedding_dict.keys(), 1)}
    idx2token_dict={}
    idx2token_dict[0]=NULL
    idx2token_dict[len(embedding_dict)+1]=OOV
    for k in token2idx_dict:
        idx2token_dict[token2idx_dict[k]]=k
    token2idx_dict[NULL] = 0
    token2idx_dict[OOV] = len(embedding_dict)+1
    embedding_dict[NULL] = [0. for _ in range(vec_size)]
    embedding_dict[OOV] = np.random.random((1,vec_size))/2-0.25
    idx2emb_dict = {idx: embedding_dict[token] for token, idx in token2idx_dict.items()}
    emb_mat = [idx2emb_dict[idx] for idx in range(len(idx2emb_dict))]
    return emb_mat, token2idx_dict, idx2token_dict

from collections import Counter
import numpy as np
word_counter, char_counter = Counter(), Counter()
train_examples, train_eval = process_file('datasets/squad/train-v1.1.json', "train", word_counter, char_counter)
dev_examples, dev_eval = process_file('datasets/squad/dev-v1.1.json', "dev", word_counter, char_counter)
test_examples, test_eval = process_file('datasets/squad/dev-v1.1.json', "dev", word_counter, char_counter)    

# save train_eval and dev_eval
with open('dataset/train_eval.json', "w") as fh:
    json.dump(train_eval, fh)
with open('dataset/dev_eval.json','w') as fh:
    json.dump(dev_eval, fh)
with open('dataset/test_eval.json','w') as fh:
    json.dump(test_eval, fh)

word_emb_mat, word2idx_dict, id2word_dict = get_embedding(
    word_counter, "word", emb_file='datasets/glove/glove.840B.300d.txt', size=int(2.2e6), vec_size=300)
char_emb_mat, char2idx_dict, id2char_dict = get_embedding(
        char_counter, "char", emb_file=None, size=None, vec_size=200)

# save id2word
import pandas as pd
df_id2word=[]
for k in id2word_dict:
    df_id2word.append([k,id2word_dict[k]])
df_id2word=pd.DataFrame(df_id2word)
df_id2word.to_csv('dataset/id2word.csv',index=None)

word_size=len(word_emb_mat)
char_input_size=len(char_emb_mat)-1
print(word_size)
print(char_input_size)
word_mat=np.zeros((len(word_emb_mat),len(word_emb_mat[0])))
for i,w in enumerate(word_emb_mat):
    word_mat[i,:]=w
print(word_mat.shape)
char_mat=np.zeros((len(char_emb_mat),len(char_emb_mat[0])))
for i,w in enumerate(char_emb_mat):
    char_mat[i,:]=w
print(char_mat.shape)
np.save('dataset/word_emb_mat.npy',word_mat)
np.save('dataset/char_emb_mat.npy',char_mat)

import pandas as pd
def get_indexs(exa, word2idx_dict, char2idx_dict, cont_limit=400, ques_limit=50, ans_limit=30, char_limit=16):
    n=len(exa)
    miss_word=0
    miss_char=0
    cont_index=np.zeros((n,cont_limit))
    ques_index=np.zeros((n,ques_limit))
    cont_char_index=np.zeros((n,cont_limit,char_limit))
    ques_char_index=np.zeros((n,ques_limit,char_limit))
    cont_len=np.zeros((n,1))
    ques_len=np.zeros((n,1))
    y_start = np.zeros((n,cont_limit))
    y_end = np.zeros((n,cont_limit))
    qid = np.zeros((n))
    overlimit=0
    
    # cont
    for i in tqdm(range(n)):
        qid[i]=int(exa[i]['id'])
        
        contexts=exa[i]['context_tokens']
        cont_len[i,0]=min(cont_limit,len(contexts))
        for j,c in enumerate(contexts):
            if j>=cont_limit:
                break
            if c in word2idx_dict:
                cont_index[i,j]=word2idx_dict[c]
            else:
                miss_word+=1
                cont_index[i,j]=word2idx_dict['--OOV--']
        contexts_char=exa[i]['context_chars']
        for j,c in enumerate(contexts_char):
            if j>=cont_limit:
                break
            for j2,c2 in enumerate(c):
                if j2>=char_limit:
                    break
                if c2 in char2idx_dict:
                    cont_char_index[i,j,j2]=char2idx_dict[c2]
                else:
                    miss_char+=1
                    cont_char_index[i,j,j2]=char2idx_dict['--OOV--']
        # ans
        st=exa[i]['y1s'][0]
        ed=exa[i]['y2s'][0]
        if st<cont_limit:
            y_start[i,st]=1
        if ed<cont_limit:
            if ed-st>ans_limit:
                y_end[i,st+ans_limit]=1
                overlimit+=1
            else:
                y_end[i,ed]=1
        
        # ques
        contexts=exa[i]['ques_tokens']
        ques_len[i,0]=min(ques_limit,len(contexts))
        for j,c in enumerate(contexts):
            if j>=ques_limit:
                break
            if c in word2idx_dict:
                ques_index[i,j]=word2idx_dict[c]
            else:
                miss_word+=1
                ques_index[i,j]=word2idx_dict['--OOV--']
        contexts_char=exa[i]['ques_chars']
        for j,c in enumerate(contexts_char):
            if j>=ques_limit:
                break
            for j2,c2 in enumerate(c):
                if j2>=char_limit:
                    break
                if c2 in char2idx_dict:
                    ques_char_index[i,j,j2]=char2idx_dict[c2]
                else:
                    miss_char+=1
                    ques_char_index[i,j,j2]=char2idx_dict['--OOV--']
    print('miss word:',miss_word)
    print('miss char:',miss_char)
    print('over limit:',overlimit)
        
    return cont_index, ques_index, cont_char_index, ques_char_index, cont_len, ques_len, y_start, y_end, qid

contw_input, quesw_input, contc_input, quesc_input, cont_len, ques_len, y_start, y_end, qid\
=get_indexs(train_examples, word2idx_dict, char2idx_dict)

np.save('dataset/train_contw_input.npy',contw_input)
np.save('dataset/train_quesw_input.npy',quesw_input)
np.save('dataset/train_contc_input.npy',contc_input)
np.save('dataset/train_quesc_input.npy',quesc_input)
np.save('dataset/train_cont_len.npy',cont_len)
np.save('dataset/train_ques_len.npy',ques_len)
np.save('dataset/train_y_start.npy',y_start)
np.save('dataset/train_y_end.npy',y_end)
np.save('dataset/train_qid.npy',qid)

contw_input, quesw_input, contc_input, quesc_input, cont_len, ques_len, y_start, y_end, qid\
=get_indexs(dev_examples, word2idx_dict, char2idx_dict)

np.save('dataset/dev_contw_input.npy',contw_input)
np.save('dataset/dev_quesw_input.npy',quesw_input)
np.save('dataset/dev_contc_input.npy',contc_input)
np.save('dataset/dev_quesc_input.npy',quesc_input)
np.save('dataset/dev_cont_len.npy',cont_len)
np.save('dataset/dev_ques_len.npy',ques_len)
np.save('dataset/dev_y_start.npy',y_start)
np.save('dataset/dev_y_end.npy',y_end)
np.save('dataset/dev_qid.npy',qid)

contw_input, quesw_input, contc_input, quesc_input, cont_len, ques_len, y_start, y_end, qid\
=get_indexs(test_examples, word2idx_dict, char2idx_dict)

np.save('dataset/test_contw_input.npy',contw_input)
np.save('dataset/test_quesw_input.npy',quesw_input)
np.save('dataset/test_contc_input.npy',contc_input)
np.save('dataset/test_quesc_input.npy',quesc_input)
np.save('dataset/test_cont_len.npy',cont_len)
np.save('dataset/test_ques_len.npy',ques_len)
np.save('dataset/test_y_start.npy',y_start)
np.save('dataset/test_y_end.npy',y_end)
np.save('dataset/test_qid.npy',qid)