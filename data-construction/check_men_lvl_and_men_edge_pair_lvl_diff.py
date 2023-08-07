# check whether the output of format_mm_data_for_edge_insertion.py retains the edges of mentions.

from tqdm import tqdm
import json

snomed_subset = "CPP"

men_lvl_data_path = "../data/MedMentions-preprocessed+/%s/st21pv_syn_attr-all-complexEdge-unsup" % snomed_subset
men_edge_pair_lvl_data_path = "../data/MedMentions-preprocessed+/%s/st21pv_syn_attr-all-complexEdge-edges-final" % snomed_subset

#fn = "test-in-KB.jsonl"
fn = "test-NIL.jsonl"

men_lvl_data_fn = "%s/%s" % (men_lvl_data_path, fn)
men_edge_lvl_data_fn = "%s/%s" % (men_edge_pair_lvl_data_path, fn)

# record dict ment tuple to edges for mention-level data
dict_men_tuple_to_edges_men_lvl = {}
num_edges = 0
with open(men_lvl_data_fn,encoding='utf-8-sig') as f_content:
    doc = f_content.readlines()

for ind, mention_info_json in enumerate(tqdm(doc)):
    mention_info = json.loads(mention_info_json)  
    mention = mention_info["mention"]
    context_left = mention_info["context_left"]
    context_right = mention_info["context_right"]
    sctid_ori = mention_info["label_concept_ori"]
    pc_paths = mention_info["parents-children_concept"]
    list_pc_paths = pc_paths.split("|")

    mention_tuple = (mention,context_left,context_right,sctid_ori)
    dict_men_tuple_to_edges_men_lvl[mention_tuple] = list_pc_paths
    if list_pc_paths != ['']:
        num_edges = num_edges + len(list_pc_paths)

print("num_edges:",num_edges)
print("dict_men_tuple_to_edges_men_lvl:",len(dict_men_tuple_to_edges_men_lvl))

# record dict ment tuple to edges for mention-pair-level data
dict_men_tuple_to_edges_men_pair_lvl = {}
with open(men_edge_lvl_data_fn,encoding='utf-8-sig') as f_content:
    doc = f_content.readlines()
for ind, mention_info_json in enumerate(tqdm(doc)):
    mention_info = json.loads(mention_info_json)  
    mention = mention_info["mention"]
    context_left = mention_info["context_left"]
    context_right = mention_info["context_right"]
    sctid_ori = mention_info["label_concept_ori"]
    parent_concept = mention_info["parent_concept"]
    child_concept = mention_info["child_concept"]

    pc_path_str = "%s-%s" % (parent_concept,child_concept)
    mention_tuple = (mention,context_left,context_right,sctid_ori)
    if mention_tuple in dict_men_tuple_to_edges_men_pair_lvl:
        list_pc_paths = dict_men_tuple_to_edges_men_pair_lvl[mention_tuple]
        list_pc_paths.append(pc_path_str)
        dict_men_tuple_to_edges_men_pair_lvl[mention_tuple] = list_pc_paths
    else:
        dict_men_tuple_to_edges_men_pair_lvl[mention_tuple] = [pc_path_str]

print("dict_men_tuple_to_edges_men_pair_lvl:",len(dict_men_tuple_to_edges_men_pair_lvl))

# compare two dictionaries
for mention_tuple, list_pc_paths_men_pair_lvl in dict_men_tuple_to_edges_men_pair_lvl.items():
    list_pc_paths_men_lvl = dict_men_tuple_to_edges_men_lvl[mention_tuple]
    if len(list_pc_paths_men_lvl) != len(list_pc_paths_men_pair_lvl): 
        # there might be repeated pc-paths (due to repeated anns in MedMentions)
        #if len(list_pc_paths_men_lvl) != len(list(set(list_pc_paths_men_pair_lvl))):
        print("mention tuple:", mention_tuple)
        print("list_pc_paths_men_lvl:",list_pc_paths_men_lvl)
        print("list_pc_paths_men_pair_lvl:",list_pc_paths_men_pair_lvl)
