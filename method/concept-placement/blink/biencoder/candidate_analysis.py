# analyse and output candidates per mention from the output of eval_biencoder.py
# get results on insertion to edges with evaluation metrics (precision@k and recall@k)
# and also generate prompts 

import os 
import argparse
import torch
from tqdm import tqdm
import json
import pandas as pd
import csv

#output str content to a file
#input: filename and the content (str)
def output_to_file(file_name,str):
    with open(file_name, 'w', encoding="utf-8-sig") as f_output:
        f_output.write(str)

def form_edge_str(edge_json_info, with_degree=True):
    if with_degree:
        return '%s (%s) -> %s (%s), degree %d' % (edge_json_info["parent"], edge_json_info["parent_idx"], edge_json_info["child"], edge_json_info["child_idx"], edge_json_info["degree"])
    else:
        return '%s (%s) -> %s (%s)' % (edge_json_info["parent"], edge_json_info["parent_idx"], edge_json_info["child"], edge_json_info["child_idx"])
    
def construct_prompt(mention, context_left, context_right, parent, child):
    '''
    A potential prompt template
    Considering [mention] (marked with asterisk) in the context below: "[context-left] *[mention]* [context-right]" Is [mention] a direct child of [parent-concept]? Please answer briefly with yes or no.

    Considering [mention] (marked with asterisk) in the context below: "[context-left] *[mention]* [context-right]" Is [mention] a direct parent of [child-concept]? Please answer briefly with yes or no.

    #Considering [mention] (marked with asterisk) in the context below: "[context-left] *[mention]* [context-right]" Is [parent-concept] -> [mention] -> [child-concept] a direct taxonomy path from parent to child? Please answer briefly with yes or no.
    '''
    if parent != "TOP":
        prompt_parent = "Considering %s (marked with asterisk) in the context below: \"%s *%s* %s\" Is %s a direct child of %s? Please answer briefly with yes or no." % (mention, context_left, mention, context_right, mention, parent)
    else:
        prompt_parent = ""
    if child != "NULL":
        prompt_child = "Considering %s (marked with asterisk) in the context below: \"%s *%s* %s\" Is %s a direct parent of %s? Please answer briefly with yes or no." % (mention, context_left, mention, context_right, mention, child)
    else:
        prompt_child = ""
    #prompt_path = "Considering %s (marked with asterisk) in the context below: \"%s *%s* %s\" Is %s -> %s -> %s a direct taxonomy path from parent to child? Please answer briefly with yes or no." % (mention, context_left, mention, context_right, parent, mention, child)
    return prompt_parent, prompt_child#, prompt_path

def construct_prompt_edge(mention, context_left, context_right, list_edge_info):
    '''
    Can you identify the correct ontological edges for the given mention based on the provided context (context_left and context_right)? The ontological edge consists of a pair where the left concept represents the parent of the mention, and the right concept represents the child of the mention. If the mention is a leaf node, the right side of the edges will be NULL (SCTID_NULL). If the context is not relevant to the options, make your decision solely based on the mention itself. There may be multiple correct options. Please answer briefly using option numbers, separated by a colon. If none of the options is correct, please answer None. 

    context_left:
    
    mention:

    context_right:
    
    options:
    '''

    # question_head = "Can you choose the correct ontological edges of the mention from the options given the context_left, context_right, and the mention below? An ontological edge is a pair where the left concept is the parent of the mention and the right concept is the child of the mention. The mention is a leaf node when NULL (SCTID_NULL) is on the right hand side of the edges. If the context is not relevant to the options, just make your decision based on the mention itself. There might be multiple correct options. Answer briefly, only with option numbers, separated by column, or None, if none of the options is correct." # manual 
    
    question_head = "Can you identify the correct ontological edges for the given mention based on the provided context (context_left and context_right)? The ontological edge consists of a pair where the left concept represents the parent of the mention, and the right concept represents the child of the mention. If the mention is a leaf node, the right side of the edges will be NULL (SCTID_NULL). If the context is not relevant to the options, make your decision solely based on the mention itself. There may be multiple correct options. Please answer briefly using option numbers, separated by a colon. If none of the options is correct, please answer None." # paraphrased by ChatGPT: "Can you help revise this paragraph so that you understand it better? \n [original-question-head-above]"

    list_edge_options = []
    for edge_rank_ind, edge_info in enumerate(list_edge_info):
        edge_str_without_degree = form_edge_str(edge_info,with_degree=False)
        list_edge_options.append('%d.%s' % (edge_rank_ind, edge_str_without_degree))
    edge_options = '\n'.join(list_edge_options)    
    prompt = "%s\n\ncontext_left:\n%s\n\nmention:\n%s\n\ncontext_right:\n%s\n\noptions:\n%s" % (question_head, context_left, mention, context_right, edge_options)
    
    return prompt

def eval_edges(dict_ctx_mention_to_list_2d_edge_strs, list_top_k=[1,5,10,50], tp_marking="(**p) (**c)"):
    '''
    Evaluate the edges and calculate MR, MRR, precision@k, recall@k
    Input: the 2d list (list of mention/query of list of edge strings), the list of top-k values for precision and recall
    Output: the metric scores
    '''
    mr = 0.0
    mrr = 0.0

    for top_k_value in list_top_k:
        tp = 0
        num_all_gold_edges = 0    
        for list_2d_edge_strs in dict_ctx_mention_to_list_2d_edge_strs.values():
            for list_edge_strs in list_2d_edge_strs:
                #print('len(list_edge_strs):',len(list_edge_strs))
                if top_k_value > len(list_edge_strs):
                    print('top-k value %d beyond predictions' % top_k_value)
                num_all_gold_edges += 1
                for ind, edge_str in enumerate(list_edge_strs[:top_k_value]):                  
                    if tp_marking in edge_str:
                        tp += 1          
                    # if tp_marking in edge_str:
                    #     mr += float(ind+1)/len(list_edge_strs)
                    #     mrr += 1/float(ind+1)
                    # else:
                    #     mr += 100
        # mr = mr/len(list_2d_mention_list_edge_strs)

        p_at_k = float(tp)/len(dict_ctx_mention_to_list_2d_edge_strs)/top_k_value
        r_at_k = float(tp)/num_all_gold_edges
        print('tp:',tp, 'num_mentions:', len(dict_ctx_mention_to_list_2d_edge_strs),'num_gold_edges:',num_all_gold_edges)
        print('p_at_%d:' % top_k_value,p_at_k,'r_at_%d:' % top_k_value,r_at_k)
        #return p_at_k,r_at_k

# add an element to a dict of list of elements for the id
def add_dict_list(dict,id,ele):
    if not id in dict:
        dict[id] = [ele] # one-element list
    else:
        list_ele = dict[id]
        list_ele.append(ele)
        dict[id] = list_ele
    return dict

def add_dict_tuple_first_and_last_ele_list(dict,id,ele_first_str,ele_last):
    # update the ctx/doc ind list if the id exists 
    info_tuple = dict[id]
    info_list = list(info_tuple)
    # retrieve and update the first ele
    ctx_id_list = info_tuple[0]
    if not ele_first_str in ctx_id_list:
        ctx_id_list.append(ele_first_str)  
    info_list[0] = ctx_id_list    
    # retrieve and update the last ele    
    if type(ele_last) == bool:
        is_subsumption = info_tuple[-1]
        is_subsumption = is_subsumption or ele_last
        info_list[-1] = is_subsumption
    else:
        true_edge_id_list = info_tuple[-1]
        if not ele_last in true_edge_id_list:
            true_edge_id_list.append(ele_last)
        info_list[-1] = true_edge_id_list
    # update them into tuple (cast into list before updating)
    dict[id] = tuple(info_list)
    return dict

def main(params):
    #model_name = "mm+2017AA-tl-pubmedbert-NIL-tag-bs128"
    #num_top_k = 100
    fname = os.path.join(params["data_path"], "%s.t7" % params["data_split"]) # this file is generated with eval_biencoder.py (see https://github.com/facebookresearch/BLINK/issues/92#issuecomment-1126293605)
    data = torch.load(fname)
    #label_input = data["labels"]
    edge_inds = data["entity_inds"]
    #label_is_NIL_input = data["labels_is_NIL"]
    #print(len(edge_inds),edge_inds[0])

    #get edge catalogue info - load as a dict
    dict_ind_edge_json_info = {}
    with open(params["edge_catalogue_fn"],encoding='utf-8-sig') as f_content:
        doc = f_content.readlines()

        for ind, edge_info_json in enumerate(tqdm(doc)):
            edge_info = json.loads(edge_info_json)  
            #edge_str = '%s (%s) -> %s (%s), degree %d' % (edge_info["parent"], edge_info["parent_idx"], edge_info["child"], edge_info["child_idx"], edge_info["degree"])
            dict_ind_edge_json_info[ind] = edge_info

    #get mention info
    fname_ori_data = os.path.join(params["original_data_path"], '%s.jsonl' % params["data_split"])

    with open(fname_ori_data,encoding='utf-8-sig') as f_content:
        doc = f_content.readlines()

    assert len(doc) == len(edge_inds)

    list_mention_w_edge_preds_strs = [] # a 1d list for diplaying
    #list_2d_mention_list_edge_strs = [] # a 2d list for evaluation
    dict_ctx_mention_to_list_2d_edge_strs = {} # dict of contextual mention to the list of all list of edge strs w.r.t each gold edge.
    dict_ctx_mention_to_list_2d_edge_strs_leaf = {} # for leaf node only
    dict_ctx_mention_to_list_2d_edge_strs_non_leaf = {} # for non-leaf node only
    dict_prompt_strs = {} # dict of prompt strs - by subsumption
    dict_prompt_strs_by_edge = {} # dict of prompt - strs by edge
    for ind, mention_info_json in enumerate(tqdm(doc)):
        mention_info = json.loads(mention_info_json)  
        mention = mention_info["mention"]
        context_left = mention_info["context_left"]
        context_right = mention_info["context_right"]
        label_concept = mention_info["label_concept"]
        label_concept_ori = mention_info["label_concept_ori"]
        entity_label_title = mention_info["entity_label_title"]
        parent_concept = mention_info["parent_concept"]
        child_concept = mention_info["child_concept"]

        topk_pred_inds = edge_inds[ind]
        list_edge_strs = []
        list_edge_info = []
        gold_edge_ind_id = -1
        edge_ind_id = 0
        for edge_ind_id_, edge_ind in enumerate(topk_pred_inds):
            edge_info = dict_ind_edge_json_info[edge_ind]
            if params["filter_by_degree"] and edge_info["degree"] == 0:
                continue
            
            # store pred edge info
            list_edge_info.append(edge_info)
            # form edge string from edge info
            edge_str = form_edge_str(edge_info)
            
            # check if tp 
            is_child = edge_info["child_idx"] == child_concept
            is_parent = edge_info["parent_idx"] == parent_concept
            if is_child:
                edge_str = '(**c) ' + edge_str # child is correct
            if is_parent:
                edge_str = '(**p) ' + edge_str # parent is correct      
            if is_child and is_parent:
                gold_edge_ind_id = edge_ind_id          
            # add top-k order ind
            edge_str = str(edge_ind_id) + '.' + edge_str
            list_edge_strs.append(edge_str)         
            # construct prompts: subsumption level
            if params["gen_prompts"]:
                prompt_parent, prompt_child = construct_prompt(mention, context_left, context_right, edge_info["parent"], edge_info["child"]) #, prompt_path                            
                if prompt_parent != "":
                    # if edge_info["parent_idx"] == parent_concept:
                    #     prompt_parent = prompt_parent + ' ' + '[correct parent]'
                    # filter out the empty prompts due to a TOP parent or a NULL child

                    if prompt_parent in dict_prompt_strs:
                        # update the ctx/doc ind list if the prompt exists
                        dict_prompt_strs = add_dict_tuple_first_and_last_ele_list(dict_prompt_strs,prompt_parent,str(ind),is_parent)
                    else:    
                        dict_prompt_strs[prompt_parent] = ([str(ind)], 
                                                           mention, 
                                                           label_concept_ori, 
                                                           "parent", 
                                                           edge_info["parent"], 
                                                           mention, 
                                                           is_parent,
                                                           )
                if prompt_child != "":
                    # if edge_info["child_idx"] == child_concept:
                    #     prompt_child = prompt_child + ' ' + '[correct child]'
                    # filter out the empty prompts due to a TOP parent or a NULL child
                    if prompt_child in dict_prompt_strs:
                        # update the ctx/doc ind list if the prompt exists
                        dict_prompt_strs = add_dict_tuple_first_and_last_ele_list(dict_prompt_strs,prompt_child,str(ind),is_child)
                    else:    
                        dict_prompt_strs[prompt_child] = ([str(ind)], 
                                                          mention, 
                                                          label_concept_ori, 
                                                          "child", 
                                                          mention, 
                                                          edge_info["child"], 
                                                          is_child,
                                                        )
            # update edge id
            edge_ind_id += 1
             # up to k edges recommended
            if len(list_edge_strs) == params["top_k_filtering"]:
                break
            
        # construct prompts: edge level
        if params["gen_prompts"]:
            prompt_by_edge = construct_prompt_edge(mention, context_left, context_right, list_edge_info)
            #dict_prompt_strs_by_edge[prompt_by_edge] = 1
            #TODO 
            if prompt_by_edge in dict_prompt_strs_by_edge:
                # update the ctx/doc ind list if the prompt exists
                dict_prompt_strs_by_edge = add_dict_tuple_first_and_last_ele_list(dict_prompt_strs_by_edge,prompt_by_edge,str(ind),str(gold_edge_ind_id))
            else:
                dict_prompt_strs_by_edge[prompt_by_edge] = ([str(ind)], 
                                                          mention, 
                                                          label_concept_ori, 
                                                          [str(gold_edge_ind_id)],
                                                        )

        list_mention_w_edge_preds_strs.append(mention_info_json + ':\n\t' + '\n\t'.join(list_edge_strs))
        #list_2d_mention_list_edge_strs.append(list_edge_strs)
        dict_ctx_mention_to_list_2d_edge_strs = add_dict_list(
                                                dict=dict_ctx_mention_to_list_2d_edge_strs,
                                                id=(mention, context_left, context_right),
                                                ele=list_edge_strs,
                                                )
        if child_concept == "SCTID_NULL":
            dict_ctx_mention_to_list_2d_edge_strs_leaf = add_dict_list(
                                                dict=dict_ctx_mention_to_list_2d_edge_strs_leaf,
                                                id=(mention, context_left, context_right),
                                                ele=list_edge_strs,
                                                )
        else:   
            dict_ctx_mention_to_list_2d_edge_strs_non_leaf = add_dict_list(
                                                dict=dict_ctx_mention_to_list_2d_edge_strs_non_leaf,
                                                id=(mention, context_left, context_right),
                                                ele=list_edge_strs,
                                                )

    print('all edges results:')
    eval_edges(dict_ctx_mention_to_list_2d_edge_strs)
    print('leaf edges results:')
    eval_edges(dict_ctx_mention_to_list_2d_edge_strs_leaf)
    print('non-leaf edges results:')
    eval_edges(dict_ctx_mention_to_list_2d_edge_strs_non_leaf)

    output_fn = os.path.join(params["data_path"], "%s-top%d-preds%s.txt" % (params["data_split"],
                                                                    params["top_k_filtering"],
                                                                    '-degree-1' if params["filter_by_degree"] else ''))
    output_to_file(output_fn,'\n'.join(list_mention_w_edge_preds_strs))

    if params["gen_prompts"]:
        # prompt by subsumptions
        output_fn_prompts = os.path.join(params["data_path"], "%s-top%d-preds%s-prompts.txt" % (params["data_split"],
                                                                    params["top_k_filtering"],
                                                                    '-degree-1' if params["filter_by_degree"] else ''))
        output_to_file(output_fn_prompts, '\n'.join(list(dict_prompt_strs.keys())))
        print('prompts in .txt saved to %s' % output_fn_prompts)

        # prompt by edges
        output_fn_prompts_by_edges = os.path.join(params["data_path"], "%s-top%d-preds%s-prompts-by-edges.txt" % (params["data_split"],
                                                                    params["top_k_filtering"],
                                                                    '-degree-1' if params["filter_by_degree"] else ''))
        output_to_file(output_fn_prompts_by_edges, '\n'.join(list(dict_prompt_strs_by_edge.keys())))
        print('prompts in .txt saved to %s' % output_fn_prompts_by_edges)

        # csv form - by subsumptions
        print('len(dict_prompt_strs):',len(dict_prompt_strs))
        output_fn_prompts_csv = output_fn_prompts[:len(output_fn_prompts)-len('.txt')] + '.csv'
        prompt_list = list(dict_prompt_strs.keys())
        ctx_id_list = ['|'.join(tuple_type_answer[0]) for tuple_type_answer in list(dict_prompt_strs.values())] # make it a comma-separated string
        mention_list = [tuple_type_answer[1] for tuple_type_answer in list(dict_prompt_strs.values())]
        snomedct_iri_ori_list = ['http://snomed.info/id/' + tuple_type_answer[2] for tuple_type_answer in list(dict_prompt_strs.values())]
        pc_type_list = [tuple_type_answer[3] for tuple_type_answer in list(dict_prompt_strs.values())]
        parent_list = [tuple_type_answer[4] for tuple_type_answer in list(dict_prompt_strs.values())]
        child_list = [tuple_type_answer[5] for tuple_type_answer in list(dict_prompt_strs.values())]
        anwser_list = [tuple_type_answer[6] for tuple_type_answer in list(dict_prompt_strs.values())]
        
        dict_data_prompts = {'ctx_id': ctx_id_list, 'prompt': prompt_list, 'mention': mention_list, 'snomedct_iri_ori': snomedct_iri_ori_list, 'parent': parent_list, 'child': child_list, 'type': pc_type_list, 'answer': anwser_list}
        df_data_prompts = pd.DataFrame.from_dict(dict_data_prompts)
        df_data_prompts.to_csv(output_fn_prompts_csv, index=True)
        print('prompts in .csv saved to %s' % output_fn_prompts_csv)

        # csv form - by edges
        print('len(dict_prompt_strs_by_edge):',len(dict_prompt_strs_by_edge))
        output_fn_prompts_by_edges_csv = output_fn_prompts_by_edges[:len(output_fn_prompts_by_edges)-len('.txt')] + '.csv'
        prompt_edge_list = list(dict_prompt_strs_by_edge.keys())
        ctx_id_list = ['|'.join(tuple_type_answer[0]) for tuple_type_answer in list(dict_prompt_strs_by_edge.values())] # make it a comma-separated string
        mention_list = [tuple_type_answer[1] for tuple_type_answer in list(dict_prompt_strs_by_edge.values())]
        snomedct_iri_ori_list = ['http://snomed.info/id/' + tuple_type_answer[2] for tuple_type_answer in list(dict_prompt_strs_by_edge.values())]
        anwser_list = [','.join(tuple_type_answer[3]) for tuple_type_answer in list(dict_prompt_strs_by_edge.values())]
        
        dict_data_prompts = {'ctx_id': ctx_id_list, 'prompt': prompt_edge_list, 'mention': mention_list, 'snomedct_iri_ori': snomedct_iri_ori_list, 'answer': anwser_list}
        df_data_prompts = pd.DataFrame.from_dict(dict_data_prompts)
        df_data_prompts.to_csv(output_fn_prompts_by_edges_csv, index=True)
        print('prompts in .csv saved to %s' % output_fn_prompts_by_edges_csv)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="output candidates from bi-encoder")
    parser.add_argument('--data_path', type=str,
                        help="data path of candidates generated by the bi-encoder", 
                        default='') 
    parser.add_argument('--original_data_path', type=str,
                        help="original data path",
                        default="data/MedMentions-preprocessed+/st21pv_syn_attr-edges-NIL") 
    parser.add_argument('--data_split', type=str,
                        help="data split, which is a part of data filename",
                        default="test") 
    parser.add_argument('--data_splits', type=str,
                        help="data splits, which are a part of data filename. Can be separated by comma",
                        default="valid,test") 
    parser.add_argument('--edge_catalogue_fn', type=str, 
                        help='filepath to entities to encode (.jsonl file)',
                        default="ontologies/SNOMEDCT-US-20140901-Disease-edges.jsonl")
    parser.add_argument('--top_k_filtering', type=int, 
                        help='a filtered number of top-k',
                        default="50")                    
    parser.add_argument('--filter_by_degree', 
                        action="store_true",
                        help='whether to only generate edges with degree of 1, to note that the complex edges are of degree 0',
                        )                    
    parser.add_argument('--gen_prompts', 
                        action="store_true",
                        help='whether to generate prompts to query LMs',
                        )                   
    args = parser.parse_args()
    print(args)
    params = args.__dict__

    data_split_lists = params["data_splits"].split(',') # param["mode"] as 'train,valid'
    for data_split in data_split_lists:
        new_params = params
        new_params["data_split"] = data_split
        main(new_params)
