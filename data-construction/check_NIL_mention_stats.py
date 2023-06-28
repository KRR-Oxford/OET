# check the unique numbers of NIL mentions
# input: ../data/MedMentions-preprocessed+/[snomed_subset_mark]/st21pv_syn_attr-all-complexEdge
# output: unique NIL mention statistics

from tqdm import tqdm
import json

snomed_subset_mark = "CPP"
#input_folder = "../data/MedMentions-preprocessed+/%s/st21pv_syn_attr-all-complexEdge" % snomed_subset_mark

#input_folder = "../data/MedMentions-preprocessed+/%s/st21pv_syn_attr-all-complexEdge-filt" % snomed_subset_mark

input_folder = "../data/MedMentions-preprocessed+/%s/st21pv_syn_full-all-complexEdge-filt" % snomed_subset_mark

#input_folder = "../data/MedMentions-preprocessed+/%s/st21pv_syn_attr-all-complexEdge-edges-final" % snomed_subset_mark

dict_mentions = {}
list_mentions = []
for data_split_mark in ["train", "valid", "test"]:
#for data_split_mark in ["train", "valid", "test-in-KB"]:
#for data_split_mark in ["test-NIL"]:
    dict_mentions_split = {}
    list_mentions_split = []
    with open("%s/%s.jsonl" % (input_folder,data_split_mark),encoding='utf-8-sig') as f_content:
        doc = f_content.readlines()

    for ind, mention_info_json in enumerate(tqdm(doc)):
        mention_info = json.loads(mention_info_json)
        context_left = mention_info["context_left"]
        mention = mention_info["mention"]
        context_right = mention_info["context_right"]
        label_concept = mention_info["label_concept"]
        if label_concept == "SCTID-less":
            if not (mention,context_left,context_right) in dict_mentions:
                dict_mentions[(mention,context_left,context_right)] = 1
            else:    
                dict_mentions[(mention,context_left,context_right)] += 1
            list_mentions.append((mention,context_left,context_right))
            
            dict_mentions_split[(mention,context_left,context_right)] = 1
            list_mentions_split.append((mention,context_left,context_right))

    print("num_unique_mentions in %s:" % data_split_mark, len(dict_mentions_split))
    print("num_repeated_mentions in %s:" % data_split_mark, len(list_mentions_split))

print("num_unique_mentions:", len(dict_mentions))
print("num_repeated_mentions:", len(list_mentions))
print("repeated mentions:")
for mention_tuple, freq in dict_mentions.items():
    if freq > 4:
        print('\t',mention_tuple, freq)

'''
Final NIL mentions

Disease
100%|██████████████████████████████████████████████████████████████████████████████████████████████████████| 12141/12141 (54979/54979) [00:00<00:00, 31390.28it/s]
num_unique_mentions in train: 329
num_repeated_mentions in train: 329
100%|████████████████████████████████████████████████████████████████████████████████████████████████████████| 4409/4409 [00:00<00:00, 28686.13it/s]
num_unique_mentions in valid: 161
num_repeated_mentions in valid: 161
100%|████████████████████████████████████████████████████████████████████████████████████████████████████████| 4085/4085 [00:00<00:00, 30507.86it/s]
num_unique_mentions in test: 114
num_repeated_mentions in test: 115
num_unique_mentions: 604
num_repeated_mentions: 605
repeated mentions:

CPP
100%|██████████████████████████████████████████████████████████████████████████████████████████████████████| 35272/35272 (133151/133151) [00:00<00:00, 44058.98it/s]
num_unique_mentions in train: 568
num_repeated_mentions in train: 568
100%|██████████████████████████████████████████████████████████████████████████████████████████████████████| 11967/11967 [00:00<00:00, 41852.70it/s]
num_unique_mentions in valid: 260
num_repeated_mentions in valid: 260
100%|██████████████████████████████████████████████████████████████████████████████████████████████████████| 11736/11736 [00:00<00:00, 44283.55it/s]
num_unique_mentions in test: 171
num_repeated_mentions in test: 172
num_unique_mentions: 999
num_repeated_mentions: 1000
repeated mentions:
'''