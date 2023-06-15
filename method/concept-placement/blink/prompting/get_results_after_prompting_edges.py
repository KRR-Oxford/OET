# get results after prompting
# direct-subsumption-level results
# TODO edge-level results - with assuming no child prediction as leaf node

import pandas as pd
from tqdm import tqdm
import sys
sys.path.insert(0, '../../preprocessing')
from onto_snomed_owl_util import load_SNOMEDCT_deeponto,load_deeponto_reasoner,load_deeponto_verbaliser,load_SNOMEDCT_owl2dict_ids,deeponto2dict_ids,deeponto2dict_ids_obj_prop,deeponto2dict_ids_ann_prop,get_definition_in_onto_from_iri,get_rdfslabel_in_onto_from_iri,get_preflabel_in_onto_from_iri,get_altlabel_in_onto_from_iri,get_title_in_onto_from_iri,get_entity_info,get_SCTID_from_OWLobj,get_iri_from_SCTID_id,get_SCTID_id_from_iri,get_in_KB_direct_children,get_in_KB_direct_parents,get_entity_graph_info,get_concept_tit_from_id_strs,check_subsumption,check_leaf
import json

top_k_value = 50
prompts_answers_fn = "../../models/biencoder/mm+2017AA-tl-pubmedbert-NIL-tag-bs128/top200_candidates/test-top%d-preds-degree-1-prompts-gpt3.5-by-arxiv.csv" % top_k_value

df = pd.read_csv(prompts_answers_fn,index_col=0)
print(df.head())
print(list(df.keys()))
def interpreting_gpt35_bin(answer):
    '''
    interpreting binary answers in texts, if unable to deterine, return -1.
    '''
    if answer[:3].lower() == "yes":
        return True
    if answer[:2].lower() == "no":
        return False
    if "the answer is \"no\"" in answer.lower():
        return False
    if "the answer is no" in answer.lower():
        return False
    return -1

# add an element to a dict of list of elements for the id
def add_dict_list(dict,id,ele,unique=False):
    if not id in dict:
        dict[id] = [ele] # one-element list
    else:
        list_ele = dict[id]
        if unique:
            if not ele in list_ele:
                list_ele.append(ele)
        else:   
            list_ele.append(ele)
        dict[id] = list_ele
    return dict

def display_dict_direct_subs(dict_id_direct_subs):
    '''
        display the dict
    '''
    print('length of dict:', len(dict_id_direct_subs))
    for id, list_dir_sub_tuple in dict_id_direct_subs.items():
        print(id)
        list_dir_sub_str = [' -> '.join(dir_sub_tuple) for dir_sub_tuple in list_dir_sub_tuple]
        print('\t' + '\n\t'.join(list_dir_sub_str))

def display_dict_edges_with_gold_mark(dict_mention_info_edges_pred,dict_mention_info_edges_gold,onto_sno,dict_SCTID_onto,onto_reasoner,dict_title_to_owl_entity):
    '''
        display the dict
        1. correct direct edges in gold - mark with (*)
        2. correct indirect edges in the newer ontology - mark with (+p) for parent->NIL and (+c) for NIL->child and (+p)(+c) for correct indirect edges.
    '''
    print('length of dict:', len(dict_mention_info_edges_pred))
    for mention_info_tuple, list_edges_tuple in dict_mention_info_edges_pred.items():
        print(mention_info_tuple)
        #list_edge_tuple_str = [('(*)' + ' -> '.join(edge_tuple)) if edge_tuple in dict_id_edges_gold.get(id,[]) else ' -> '.join(edge_tuple) for edge_tuple in list_edges_tuple] # mark the correct ones with (*) at the begining
        list_edge_strs = []
        for edge_tuple in list_edges_tuple:
            edge_str = ' -> '.join(edge_tuple)
            if edge_tuple in dict_mention_info_edges_gold.get(mention_info_tuple,[]):
                edge_str = '(*)' + edge_str
            else:
                _, _, snomedct_iri_ori = mention_info_tuple
                # title_ori = get_title_in_onto_from_iri(onto_sno,snomedct_iri_ori)
                # parent_title = edge_tuple[0]
                # child_title = edge_tuple[1] 
                # is_parent = check_subsumption_from_titles(onto_reasoner,dict_title_to_owl_entity,parent_title,title_ori)
                # if child_title != 'NULL':
                #     is_child = check_subsumption_from_titles(onto_reasoner,dict_title_to_owl_entity,title_ori,child_title)
                # else:
                #     is_child = check_leaf(onto_sno,dict_SCTID_onto,snomedct_iri_ori)
                is_edge, is_parent, is_child = check_edge_by_onto(snomedct_iri_ori,edge_tuple,onto_sno,dict_SCTID_onto,onto_reasoner,dict_title_to_owl_entity)
                if is_child:
                    edge_str = '(+c)' + edge_str
                if is_parent:
                    edge_str = '(+p)' + edge_str                
            list_edge_strs.append(edge_str)    
        str_edge_tuple_str = '\n\t'.join(list_edge_strs)
        print('\t' + str_edge_tuple_str)

def check_edge_by_onto(mention_snomedct_iri_ori,edge_tuple,onto_sno,dict_SCTID_onto,onto_reasoner,dict_title_to_owl_entity):
    '''
        check weather the (direct or indirect) edge exist in the ontology
        return: is_edge, is_parent (parent -> NIL), is_child (NIL -> child)
    '''
    title_ori = get_title_in_onto_from_iri(onto_sno,mention_snomedct_iri_ori)
    parent_title = edge_tuple[0]
    child_title = edge_tuple[1] 
    is_parent = check_subsumption_from_titles(onto_reasoner,dict_title_to_owl_entity,parent_title,title_ori)
    if child_title != 'NULL':
        is_child = check_subsumption_from_titles(onto_reasoner,dict_title_to_owl_entity,title_ori,child_title)
    else:
        is_child = check_leaf(onto_sno,dict_SCTID_onto,snomedct_iri_ori)
    is_edge = is_parent and is_child
    return is_edge,is_parent,is_child

def check_subsumption_from_titles(onto_reasoner,dict_title_to_owl_entity,parent,child):
    if not child in dict_title_to_owl_entity:
        print("child", child, "not in ontology")
        return False
    else:  
        sub_entity = dict_title_to_owl_entity[child]

    if not parent in dict_title_to_owl_entity:
        print("parent", parent, "not in ontology")
        return False
    else: 
        super_entity = dict_title_to_owl_entity[parent]
    return check_subsumption(onto_reasoner,sub_entity,super_entity)

def display_results(tp, tn, fp, fn):
    acc = float(tp+tn)/(tp+tn+fp+fn)
    prec = float(tp)/(tp+fp)
    rec = float(tp)/(tp+fn)
    f1 = 2*prec*rec/(prec+rec)
    print('tp:', tp, 'tn:', tn, 'fp:', fp, 'fn:', fn)
    print('acc:',acc,'prec:',prec,'rec:',rec,'f1:',f1)

def get_dict_title_to_owl_entity(onto, dict_IRI_onto):
    dict_title_to_owl_entity = {}
    for iri,owl_entity in dict_IRI_onto.items():
        set_rdfslabels = get_rdfslabel_in_onto_from_iri(onto,iri)
        if len(set_rdfslabels) > 0:
            title = list(set_rdfslabels)[0]
        else:
            title = ''
        dict_title_to_owl_entity[title] = owl_entity
    print('dict_title_to_owl_entity:',len(dict_title_to_owl_entity))
    return dict_title_to_owl_entity

def load_gold_standard_edges(data_fn):
    dict_mention_info_to_list_edges={}
    with open(data_fn,encoding='utf-8-sig') as f_content:
        doc = f_content.readlines()

    #get ctx_ids 
    dict_mention_info={}
    for ind, mention_info_json in enumerate(tqdm(doc)):
        mention_info = json.loads(mention_info_json)  
        mention = mention_info["mention"]
        context_left = mention_info["context_left"]
        context_right = mention_info["context_right"]    
        dict_mention_info = add_dict_list(dict=dict_mention_info,
                                          id=(mention,context_left,context_right),
                                          ele=str(ind),
                                          unique=True)
    #print('dict_mention_info:',dict_mention_info)

    for ind, mention_info_json in enumerate(tqdm(doc)):
        mention_info = json.loads(mention_info_json)  
        mention = mention_info["mention"]
        context_left = mention_info["context_left"]
        context_right = mention_info["context_right"]
        label_concept_ori = mention_info["label_concept_ori"]
        #parent_concept = mention_info["parent_concept"]
        #child_concept = mention_info["child_concept"]
        parent_title = mention_info["parent"]
        child_title = mention_info["child"]
        ctx_ids = dict_mention_info[(mention,context_left,context_right)]
        ctx_ids = '|'.join(ctx_ids)
        snomedct_iri_ori = get_iri_from_SCTID_id(label_concept_ori)
        dict_mention_info_to_list_edges = add_dict_list(dict=dict_mention_info_to_list_edges,
                                                   id=(ctx_ids,mention,snomedct_iri_ori),
                                                   ele=(parent_title,child_title))        
    return dict_mention_info_to_list_edges

def eval_edges_by_gold(dict_mention_info_to_gold_edges, dict_mention_info_to_pred_edges):
    tp = 0
    num_all_gold_edges = 0
    num_all_pred_edges = 0
    for mention_info, list_gold_edges in dict_mention_info_to_gold_edges.items():
        #ctx_ids,mention,_,_ = mention_info
        list_pred_edges = dict_mention_info_to_pred_edges.get(mention_info,[])
        #print('list_pred_edges:',list_pred_edges)
        tp += count_num_overlapping_lists(list_gold_edges,list_pred_edges)
        num_all_gold_edges += len(list_gold_edges)
        num_all_pred_edges += len(list_pred_edges)

    #prec = float(tp)/len(dict_mention_info_to_gold_edges)/top_k_value
    prec = float(tp)/num_all_pred_edges
    rec = float(tp)/num_all_gold_edges
    print('tp:',tp, 'num_mentions:', len(dict_mention_info_to_gold_edges),'num_gold_edges:',num_all_gold_edges, 'num_pred_edges:', num_all_pred_edges)
    print('precision:',prec,'recall:',rec)

def eval_direct_and_indirect_edges_by_gold_and_onto(dict_mention_info_to_gold_edges, dict_mention_info_to_pred_edges,onto_sno,dict_SCTID_onto,onto_sno_reasoner,dict_title_to_owl_entity):    
    '''
        not used, given that the recall cannot be estimated, and the precision is the same as in eval_direct_and_indirect_edges_by_onto() below
    '''
    tp_direct = 0
    tp_indirect = 0
    tp = 0
    num_all_direct_edges = 0
    num_all_indirect_edges = 0
    num_all_pred_edges = 0
    for mention_info, list_gold_edges in dict_mention_info_to_gold_edges.items():
        _,_,snomedct_iri_ori = mention_info
        list_pred_edges = dict_mention_info_to_pred_edges.get(mention_info,[])
        #print('list_pred_edges:',list_pred_edges)
        for pred_edge_tuple in list_pred_edges:
            if pred_edge_tuple in list_gold_edges:
                tp_direct += 1
            else:
                is_edge, is_parent, is_child = check_edge_by_onto(snomedct_iri_ori,pred_edge_tuple,onto_sno,dict_SCTID_onto,onto_sno_reasoner,dict_title_to_owl_entity)
                if is_edge:
                    num_all_indirect_edges += 1
                    tp_indirect += 1     
        #tp += count_num_overlapping_lists(list_gold_edges,list_pred_edges)
        num_all_direct_edges += len(list_gold_edges)
        num_all_pred_edges += len(list_pred_edges)

    tp = tp_direct + tp_indirect
    num_all_gold_edges = num_all_direct_edges + num_all_indirect_edges
    #prec = float(tp)/len(dict_mention_info_to_gold_edges)/top_k_value
    prec = float(tp)/num_all_pred_edges
    rec = float(tp)/num_all_gold_edges
    print('tp:',tp, 'num_mentions:', len(dict_mention_info_to_gold_edges),'num_gold_edges:',num_all_gold_edges, 'num_pred_edges:', num_all_pred_edges)
    print('precision:',prec,'recall:',rec)

def eval_direct_and_indirect_edges_by_onto(dict_mention_info_to_gold_edges, dict_mention_info_to_pred_edges,onto_sno,dict_SCTID_onto,onto_sno_reasoner,dict_title_to_owl_entity):    
    tp = 0
    num_all_pred_edges = 0
    for mention_info, list_gold_edges in dict_mention_info_to_gold_edges.items():
        _,_,snomedct_iri_ori = mention_info
        list_pred_edges = dict_mention_info_to_pred_edges.get(mention_info,[])
        #print('list_pred_edges:',list_pred_edges)
        for pred_edge_tuple in list_pred_edges:
            
            is_edge, is_parent, is_child = check_edge_by_onto(snomedct_iri_ori,pred_edge_tuple,onto_sno,dict_SCTID_onto,onto_sno_reasoner,dict_title_to_owl_entity)
            if is_edge:
                tp += 1     
        num_all_pred_edges += len(list_pred_edges)

    #prec = float(tp)/len(dict_mention_info_to_gold_edges)/top_k_value
    prec = float(tp)/num_all_pred_edges
    print('tp:',tp, 'num_mentions:', len(dict_mention_info_to_gold_edges), 'num_pred_edges:', num_all_pred_edges)
    print('precision:',prec)

def count_num_overlapping_lists(lst1,lst2):
    lst_overlap = [ele for ele in lst1 if ele in lst2]
    return len(lst_overlap)

#loop over entities in the ontology and create an entity catelogue
onto_ver = '20140901-Disease'
ontology_name = "SNOMEDCT-US-%s-final.owl" % onto_ver
fn_ontology = "../../ontologies/%s" % ontology_name 
onto_sno = load_SNOMEDCT_deeponto(fn_ontology)
#onto_sno_reasoner = load_deeponto_reasoner(onto_sno)
dict_SCTID_onto = deeponto2dict_ids(onto_sno)
#dict_SCTID_onto_older_obj_prop = deeponto2dict_ids_obj_prop(onto_sno_older)
#dict_SCTID_onto_older_ann_prop = deeponto2dict_ids_ann_prop(onto_sno_older)
dict_title_to_owl_entity = get_dict_title_to_owl_entity(onto_sno,dict_SCTID_onto)

onto_ver = '20170301-Disease'
ontology_name_newer = "SNOMEDCT-US-%s-final.owl" % onto_ver
fn_ontology_newer = "../../ontologies/%s" % ontology_name_newer 
onto_sno_newer = load_SNOMEDCT_deeponto(fn_ontology_newer)
onto_sno_reasoner = load_deeponto_reasoner(onto_sno_newer)
dict_SCTID_onto_newer = deeponto2dict_ids(onto_sno_newer)
dict_title_to_owl_entity_newer = get_dict_title_to_owl_entity(onto_sno_newer,dict_SCTID_onto_newer)
# combine the dict_title_to_owl_entity_newer with dict_title_to_owl_entity
for title, owl_entity in dict_title_to_owl_entity_newer.items():
    if not title in dict_title_to_owl_entity:
        dict_title_to_owl_entity[title] = owl_entity

df["onto-subsumption"] = ""
dict_pred_direct_subs={}
tp, tn, fp, fn = 0, 0, 0, 0
tp_w_indirect, tn_w_indirect, fp_w_indirect, fn_w_indirect = 0, 0, 0, 0
for i, row in tqdm(df.iterrows()):
    if interpreting_gpt35_bin(row["gpt-3.5-turbo"]):
        ctx_ids = row["ctx_id"]
        mention = row["mention"]
        snomedct_iri_ori = row["snomedct_iri_ori"]
        mention_info_tuple = (ctx_ids,mention,snomedct_iri_ori)
        pred_direct_subs = (row["parent"],row["child"])
        dict_pred_direct_subs = add_dict_list(dict_pred_direct_subs,mention_info_tuple,pred_direct_subs)

    if row["answer"] == True and interpreting_gpt35_bin(row["gpt-3.5-turbo"]) == True:
        tp += 1        
    if row["answer"] == False and interpreting_gpt35_bin(row["gpt-3.5-turbo"]) != True:
        tn += 1
    if row["answer"] == False and interpreting_gpt35_bin(row["gpt-3.5-turbo"]) == True:
        fp += 1
    if row["answer"] == True and interpreting_gpt35_bin(row["gpt-3.5-turbo"]) != True:
        fn += 1

    snomedct_iri_ori = row["snomedct_iri_ori"]
    title_ori = get_title_in_onto_from_iri(onto_sno_newer,snomedct_iri_ori)

    if row["type"]=="child":
        parent_title = title_ori
        child_title = row["child"]
    if row["type"]=="parent":
        parent_title = row["parent"]
        child_title = title_ori
    is_subsumption = check_subsumption_from_titles(onto_sno_reasoner,dict_title_to_owl_entity,parent_title,child_title)
    df.at[i,"onto-subsumption"] = is_subsumption
    if is_subsumption == True and interpreting_gpt35_bin(row["gpt-3.5-turbo"]) == True:
        tp_w_indirect += 1
    if is_subsumption == False and interpreting_gpt35_bin(row["gpt-3.5-turbo"]) != True:
        tn_w_indirect += 1
    if is_subsumption == False and interpreting_gpt35_bin(row["gpt-3.5-turbo"]) == True:
        fp_w_indirect += 1
    if is_subsumption == True and interpreting_gpt35_bin(row["gpt-3.5-turbo"]) != True:
        fn_w_indirect += 1

# output dataframe with all subsumptions
prompts_answers_updated_fn = prompts_answers_fn[:len(prompts_answers_fn)-len('.csv')] + '_all_subs.csv'
df.to_csv(prompts_answers_updated_fn, index=True)
print('updated .csv with all subs saved to %s' % prompts_answers_updated_fn)

# get direct subsumption level results
print('direct subsumptions:')
display_results(tp, tn, fp, fn)
print('direct and in-direct subsumptions:')
display_results(tp_w_indirect, tn_w_indirect, fp_w_indirect, fn_w_indirect)

# get edge level results
data_fn = "../../data/MedMentions-preprocessed+/st21pv_syn_attr-edges-NIL/test.jsonl"
dict_mention_info_to_list_edges_gold = load_gold_standard_edges(data_fn)

display_dict_direct_subs(dict_pred_direct_subs)

# infer edges: if no child predicted for the mention, then assume that the mention is linked to a leaf node.
dict_mention_info_to_list_edges_pred = {}
for mention_info_tuple, list_pred_dir_subs in dict_pred_direct_subs.items():
    # gather list of parents and children
    list_edge_parents = []
    list_edge_children = []
    ctx_ids, mention, snomedct_iri_ori = mention_info_tuple
    for pred_dir_sub_tuple in list_pred_dir_subs:
        sub_parent_title, sub_child_title = pred_dir_sub_tuple
        if mention == sub_parent_title:
            list_edge_children.append(sub_child_title)
        if mention == sub_child_title:
            list_edge_parents.append(sub_parent_title)
    # form list of edges
    if len(list_edge_children) == 0:
        list_edge_children = ["NULL"]
    # if len(list_edge_parents) == 0:
    #     list_edge_parents = ["TOP"]
    list_edges = [(parent,child) for parent in list_edge_parents for child in list_edge_children]
    dict_mention_info_to_list_edges_pred[(ctx_ids, mention, snomedct_iri_ori)] = list_edges
    # for ctx_id in ctx_ids.split('|'):
    #     dict_mention_info_to_list_edges_pred[(ctx_id, mention)] = list_edges

# display gold and pred edges 
print('dict_mention_info_to_list_edges_gold:',len(dict_mention_info_to_list_edges_gold))
print('\t\n'.join([' '.join(ele_tuple) for ele_tuple in list(dict_mention_info_to_list_edges_gold.keys())]))

print('dict_mention_info_to_list_edges_pred:',len(dict_mention_info_to_list_edges_pred))
print('\t\n'.join([' '.join(ele_tuple) for ele_tuple in list(dict_mention_info_to_list_edges_pred.keys())]))

for mention_into_tuple in dict_mention_info_to_list_edges_gold.keys():
    if not mention_info_tuple in dict_mention_info_to_list_edges_pred:
        print(mention_info_tuple, 'not in pred')

print("display gold edges")
display_dict_direct_subs(dict_mention_info_to_list_edges_gold)
print("display pred edges")
display_dict_edges_with_gold_mark(dict_mention_info_to_list_edges_pred,dict_mention_info_to_list_edges_gold,onto_sno_newer,dict_SCTID_onto_newer,onto_sno_reasoner,dict_title_to_owl_entity)

# calculate edge-level results
print('results on direct edges')
eval_edges_by_gold(dict_mention_info_to_list_edges_gold,dict_mention_info_to_list_edges_pred)

print('results on direct+indirect edges - precision only')
# eval_direct_and_indirect_edges_by_gold_and_onto(dict_mention_info_to_list_edges_gold, dict_mention_info_to_list_edges_pred,onto_sno_newer,dict_SCTID_onto_newer,onto_sno_reasoner,dict_title_to_owl_entity)
eval_direct_and_indirect_edges_by_onto(dict_mention_info_to_list_edges_gold, dict_mention_info_to_list_edges_pred,onto_sno_newer,dict_SCTID_onto_newer,onto_sno_reasoner,dict_title_to_owl_entity)
