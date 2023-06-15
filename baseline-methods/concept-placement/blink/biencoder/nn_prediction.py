# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

# adapted by HD, added features for out-of-KB entity detection

import json
import logging #TODO
import torch
from tqdm import tqdm

import blink.candidate_ranking.utils as utils
from blink.biencoder.zeshel_utils import WORLDS, Stats
from rank_bm25 import BM25Okapi # BM25-based candidate generation (replicating Ji et al., 2020)
import numpy as np

# function adapted from https://www.w3resource.com/python-exercises/list/python-data-type-list-exercise-32.php
def is_Sublist(s, l):
	sub_set = False
	if s == []:
		sub_set = True
	elif s == l:
		sub_set = True
	elif len(s) > len(l):
		sub_set = False
	else:
		for i in range(len(l)):
			if l[i] == s[0]:
				n = 1
				while (n < len(s)) and ((i+n) < len(l)) and (l[i+n] == s[n]):
					n += 1
				
				if n == len(s):
					sub_set = True
	return sub_set

# count the set diff between two lists, considering both directions if chosen the pair comparision, otherwise just count those in the first set but not in the second set; in all cases, clip the count to 0 if below 0.
def count_list_set_diff(list_a,list_b,pair_comparison=True):
    if pair_comparison:
        return max(0,len(set(list_a) - set(list_b))) + max(0,len(set(list_b) - set(list_a)))
    else:
        return max(0,len(set(list_a) - set(list_b)))

# form a set of features per mention of whether the mention has no, one, or several matching names in the entities through string matching (exact or fuzzy) (Rao, McNamee, and Dredze 2013; McNamee et al. 2009)
# we only consider the entities from the candidate list
# input: (i) list_mention_input, the list of sub-token ids in a mention (as formed in data_process.get_mention_representation())
#        (ii) list_2d_label_input, the list of label titles + descriptions (as formed in data_process.get_context_representation()), where each is a list of sub-token ids; this list can be either the full candidate list or the top-k candidate list after the candidate generation stage
def get_is_men_str_matchable_features(list_mention_input,list_2d_label_input,index_title_special_token=3,fuzzy_tolerance=2):
    #print("mention_input:",len(list_mention_input),list_mention_input)
    #print("label_input:",list_2d_label_input,len(list_2d_label_input),len(list_2d_label_input[0]))

    #for mention_sub_token_list in list_2d_mention_input:
    # clean mention input
    mention_sub_token_list = [sub_token_id for sub_token_id in list_mention_input if sub_token_id >= 3]
    mention_matched_exact = 0
    mention_matched_exact_w_desc = 0
    mention_matched_fuzzy = 0
    mention_matched_fuzzy_w_desc = 0
    for label_sub_token_list in list_2d_label_input:
        # get list of *title* sub token ids 
        label_tit_sub_token_list = get_title_ids_from_label_sub_token_list(
                                        label_sub_token_list,
                                        index_title_special_token=index_title_special_token)
        
        # exact matching
        if mention_matched_exact < 2:
            if mention_sub_token_list == label_tit_sub_token_list:
                mention_matched_exact += 1

        if mention_matched_exact_w_desc < 2:
            if is_Sublist(mention_sub_token_list,label_sub_token_list):
                mention_matched_exact_w_desc += 1
        
        # fuzzy matching
        if mention_matched_fuzzy < 2:
            num_set_diff_men_tit = count_list_set_diff(mention_sub_token_list,label_tit_sub_token_list,pair_comparison=True)
            if num_set_diff_men_tit <= fuzzy_tolerance:
                mention_matched_fuzzy += 1

        if mention_matched_fuzzy_w_desc < 2:
            num_set_diff_men_tit_desc = count_list_set_diff(mention_sub_token_list,label_sub_token_list,pair_comparison=False)
            if num_set_diff_men_tit_desc <= fuzzy_tolerance:
                mention_matched_fuzzy_w_desc += 1
        
    mention_matchable_exact = mention_matched_exact > 0
    mention_matchable_exact_w_desc_one = mention_matched_exact_w_desc == 1
    mention_matchable_exact_w_desc_several = mention_matched_exact_w_desc > 1
    mention_matchable_fuzzy_one = mention_matched_fuzzy == 1
    mention_matchable_fuzzy_several = mention_matched_fuzzy > 1
    mention_matched_fuzzy_w_desc_one = mention_matched_fuzzy_w_desc == 1
    mention_matchable_fuzzy_w_desc_several = mention_matched_fuzzy_w_desc > 1

    is_men_str_matchable_features = [mention_matchable_exact,mention_matchable_exact_w_desc_one,mention_matchable_exact_w_desc_several,mention_matchable_fuzzy_one,mention_matchable_fuzzy_several,mention_matched_fuzzy_w_desc_one,mention_matchable_fuzzy_w_desc_several]

    #print('is_men_str_matchable_features:',is_men_str_matchable_features)
    return is_men_str_matchable_features

# normalise to the "syn as canonical ent" row-id from the "syn as ent" row-id
def _normalise_local_id(local_id,local_id2wikipedia_id,wikipedia_id2local_id):
    if local_id in local_id2wikipedia_id:
        local_id_normalised = wikipedia_id2local_id[local_id2wikipedia_id[local_id]]
    else:
        local_id_normalised = local_id
    return local_id_normalised

# normalise to the original ent row-id from the "syn as ent" row-id 
def _normalise_to_ori_local_id(local_id,local_id2wikipedia_id,wikipedia_id2original_local_id):
    return _normalise_local_id(local_id,local_id2wikipedia_id,wikipedia_id2original_local_id)

def _aggregating_indices_synonyms(indicies_per_datum,local_id2wikipedia_id,wikipedia_id2local_id,top_k):
    #aggregating indicies of synonyms (by maximum)
    indicies_per_datum_ori = indicies_per_datum[:]
    #normalise the indicies to those of the canonical names (not synonyms).
    indicies_per_datum = [_normalise_local_id(int(indice),local_id2wikipedia_id,wikipedia_id2local_id) for indice in indicies_per_datum]
    #remove duplicates in normalised indicies
    indicies_per_datum = list(dict.fromkeys(indicies_per_datum))[:top_k]    
    if len(indicies_per_datum) != top_k:
        print('indicies_per_datum:',len(indicies_per_datum),'top_k:',top_k)
        print('ori->new indicies_per_datum:',indicies_per_datum_ori,'->',indicies_per_datum)
    indicies_per_datum = np.array(indicies_per_datum)
    return indicies_per_datum

# we use max (see the function above) as averaging is not working well due to the discrepancy in scores.
def _aggregating_indices_synonyms_ave(indicies_per_datum,scores_per_datum,local_id2wikipedia_id,wikipedia_id2local_id,top_k):
    #aggregating indicies of synonyms (by average among top topk*10)
    #normalise the indicies to those of the canonical names (not synonyms).
    indicies_per_datum = [wikipedia_id2local_id[local_id2wikipedia_id[int(indice)]] for indice in indicies_per_datum]
    #get dict of indice to list of scores of all same entities (inc. synonyms)
    dict_indice_to_score = {}
    #print('scores_per_datum:',scores_per_datum)
    #normalise the score with softmax
    #scores_per_datum = _softmax(scores_per_datum)
    #print('scores_per_datum:',scores_per_datum)
    for indice, score in zip(indicies_per_datum,scores_per_datum):
        if not indice in dict_indice_to_score:
            dict_indice_to_score[indice] = [score]
        else:
            list_scores_indice = dict_indice_to_score[indice]
            list_scores_indice.append(score)
            dict_indice_to_score[indice] = list_scores_indice
    #average the list of scores to one and update the dict
    for indice, list_of_scores in dict_indice_to_score.items():
        score_ave = np.mean(np.array(list_of_scores))
        dict_indice_to_score[indice] = score_ave
    #print('dict_indice_to_score:',dict_indice_to_score)
    #rank by value (averaged scores)    
    dict_indice_to_score = {k: v for k, v in sorted(dict_indice_to_score.items(), key=lambda item: item[1])}    
    #output top_k
    indicies_per_datum = list(dict_indice_to_score.keys())[:top_k]
    assert len(indicies_per_datum) == top_k
    indicies_per_datum = np.array(indicies_per_datum)
    return indicies_per_datum

def get_title_ids_from_label_sub_token_list(label_sub_token_list,index_title_special_token=3):
    label_sub_token_list = label_sub_token_list[1:-1]
    # get the position of title mark 
    if index_title_special_token in label_sub_token_list:
        pos_title_mark = label_sub_token_list.index(index_title_special_token)
    else:
        # no title mark, thus everything is in title
        print('get_is_men_str_matchable_features(): no title mark found for ', label_sub_token_list)
        pos_title_mark = len(label_sub_token_list) 
    # get title sub tokens as a list
    label_tit_sub_token_list = label_sub_token_list[:pos_title_mark]
    # get desc sub tokens as a list
    #label_desc_sub_token_list = label_sub_token_list[pos_title_mark+1:]
    return label_tit_sub_token_list

# def get_list_2d_title_ids_from_candidate_pool(list_2d_canditate_pool,index_title_special_token=3):
#     list_2d_candidate_title_ids = []
#     for label_sub_token_list in list_2d_canditate_pool:
#         label_tit_sub_token_list = get_title_ids_from_label_sub_token_list(label_sub_token_list)
#         list_2d_candidate_title_ids.append(label_tit_sub_token_list)
#     return list_2d_candidate_title_ids

# get ranking indices with BM25
def get_ranking_indices_w_BM25(list_mention_input,list_2d_candidate_title_ids,topn=100,index_title_special_token=3):
    #clean the ids by removing special token ids and padding ids
    mention_sub_token_list = [sub_token_id for sub_token_id in list_mention_input if sub_token_id >= 3]
    # list_2d_candidate_title_ids = []
    # for label_sub_token_list in list_2d_canditate_pool:
    #     # clean label ids
    #     label_sub_token_list = label_sub_token_list[1:-1]
    #     # get the position of title mark 
    #     if index_title_special_token in label_sub_token_list:
    #         pos_title_mark = label_sub_token_list.index(index_title_special_token)
    #     else:
    #         # no title mark, thus everything is in title
    #         print('get_ranking_indices_w_BM25(): no title mark found for ', label_sub_token_list)
    #         pos_title_mark = len(label_sub_token_list) 
    #     # get title sub tokens as a list
    #     label_tit_sub_token_list = label_sub_token_list[:pos_title_mark]
    #     # get desc sub tokens as a list
    #     #label_desc_sub_token_list = label_sub_token_list[pos_title_mark+1:]
    #     list_2d_candidate_title_ids.append(label_tit_sub_token_list)
    label_tit_sub_word_id_bm25 = BM25Okapi(list_2d_candidate_title_ids)
    scores = label_tit_sub_word_id_bm25.get_scores(mention_sub_token_list)
    topn_indicies = np.argsort(scores)[::-1][:topn]
    topn_scores = scores[topn_indicies]
    #topn_by_subwords_ids = label_tit_sub_word_id_bm25.get_top_n(mention_sub_token_list, list_2d_candidate_title_ids, n=topn)
    #topn_indicies = [list_2d_candidate_title_ids.index(label_title_subword_ids) for label_title_subword_ids in topn_by_subwords_ids]
    #scores = []

    #print('topn_indicies:',topn_indicies, type(topn_indicies))
    #print('topn_scores:',topn_scores, type(topn_scores))
    return topn_indicies.tolist(), topn_scores

# some changes applied based on https://github.com/facebookresearch/BLINK/issues/115#issuecomment-1119282640
# this function generates new data to train cross-encoder, from the candidates (or nns) generated by biencoder.
def get_topk_predictions(
    reranker,
    train_dataloader,
    candidate_pool, # the candidate token ids (w [SYN]-concatenated in the syn mode), this is only used for output
    cand_encode_list,
    wikipedia_id2local_id,
    local_id2wikipedia_id,
    silent,
    logger,
    top_k=100,
    is_zeshel=False,
    save_predictions=False,
    save_true_predictions_only=False,
    add_NIL=False, # add NIL to the last element of the biencoder predicted entity indicies, if NIL was not predicted
    NIL_ent_id=88150,
    use_BM25=False,
    candidate_pool_for_BM25=None, # the candidate token ids for BM25 (w syn as entities in the syn mode), this is used for BM25 only for searching ents from ments.
    get_is_men_str_mat_fts=False,
    index_title_special_token=3,
    #aggregating_factor=20, # for top_k entities & synonyms aggregation (for synonyms as entities)
):
    reranker.model.eval()
    device = reranker.device
    #print('device:',device)
    logger.info("Getting top %d predictions." % top_k)
    if silent:
        iter_ = train_dataloader
    else:
        iter_ = tqdm(train_dataloader)

    nn_context = []
    nn_candidates = []
    nn_labels = []
    nn_labels_is_NIL = []
    nn_entity_inds = []
    nn_is_mention_str_matchable_fts = []
    nn_worlds = []
    stats = {}

    if is_zeshel:
        world_size = len(WORLDS)
    else:
        # only one domain
        world_size = 1
        candidate_pool = [candidate_pool]
        candidate_pool_for_BM25 = [candidate_pool_for_BM25]
        cand_encode_list = [cand_encode_list]

    # process candidate_pool_for_BM25
    if use_BM25:
        list_2d_canditate_pool = candidate_pool_for_BM25[0].cpu().tolist()
        # by list comprehension (instead of get_list_2d_title_ids_from_candidate_pool())
        list_2d_candidate_title_ids = [get_title_ids_from_label_sub_token_list(label_sub_token_list,index_title_special_token=index_title_special_token) for label_sub_token_list in list_2d_canditate_pool]
        
    logger.info("World size : %d" % world_size)
    #print('candidate_pool:',candidate_pool)
    #print('candidate_pool:',candidate_pool,len(candidate_pool),candidate_pool[0].size())
    #1 torch.Size([88151, 128])
    #print('cand_encode_list:',cand_encode_list)
    '''
    candidate_pool: [tensor([[  101, 13878,  1010,  ...,     0,     0,     0],
        [  101, 21419, 13675,  ...,     0,     0,     0],
        [  101, 21419,  9253,  ...,     0,     0,     0],
        ...,
        [  101, 18404, 10536,  ...,     0,     0,     0],
        [  101,  1040,  7274,  ...,     0,     0,     0],
        [  101,  9152,  2140,  ...,     0,     0,     0]])]
    cand_encode_list: [tensor([[ 0.2102,  0.1818, -0.3594,  ..., -0.3182, -0.8104, -0.1960],
        [ 0.0642,  0.2399, -0.0787,  ..., -0.4488, -0.6695, -0.4290],
        [-0.0145,  0.1526, -0.0516,  ..., -0.4228, -0.4721, -0.2667],
        ...,
        [ 0.5740, -0.0637, -0.1766,  ...,  0.2560, -0.3511, -0.2073],
        [ 0.3947,  0.1827,  0.0299,  ...,  0.0638, -0.5476, -0.0607],
        [ 0.1874,  0.0835, -0.0825,  ..., -0.1674, -0.6785, -0.1951]])]
    '''

    for i in range(world_size):
        stats[i] = Stats(top_k)
    
    # get dict of wikipedia_id2original_local_id
    wikipedia_id2_ori_local_id = {k:ori_id for ori_id, (k,v) in enumerate(wikipedia_id2local_id.items())}
    #print('wikipedia_id2_ori_local_id:',list(wikipedia_id2_ori_local_id.items())[:100])

    #oid = 0
    for step, batch in enumerate(iter_):
        batch = tuple(t.to(device) for t in batch)
        #context_input, _, srcs, label_ids = batch
        if is_zeshel:
            mention_input, context_input, label_input, srcs, label_ids, is_label_NIL = batch
        else:
            mention_input, context_input, label_input, label_ids, is_label_NIL = batch
            # here you can also know whether the label is NIL - may be useful later
            srcs = torch.tensor([0] * context_input.size(0), device=device)    
        src = srcs[0].item()
        #print('src:',src)

        if not use_BM25:
            cand_encode_list[src] = cand_encode_list[src].to(device)
            scores = reranker.score_candidate(
                context_input, 
                None, 
                #cand_encs=cand_encode_list[src].to(device)
                cand_encs=cand_encode_list[src]
            )
            #values, indicies = scores.topk(top_k*aggregating_factor)
            values, indicies = scores.topk(top_k)
            indicies = indicies.data.cpu().numpy()
            #print('indicies before aggregation:',indicies)
            # aggregating results
            #indicies = np.array([_aggregating_indices_synonyms(indicies_per_datum,local_id2wikipedia_id,wikipedia_id2local_id,top_k) for indicies_per_datum in indicies])
            #print('indicies after aggregation:',indicies)
            # # add NIL to the end
            # if add_NIL:
            #     for ind, indices_per_datum in enumerate(indicies):
            #         if not NIL_ent_id in indices_per_datum:
            #             indicies[ind][-1] = NIL_ent_id
            #indicies_np = indicies.data.cpu().numpy()
            #nn_entity_inds.extend(indicies_np)
        
            #print('values in get_topk_predictions():',values)
            #print('indicies in get_topk_predictions():',indicies)
        #print('label_ids:',label_ids) # the problem is that this is the label_ids, but the indices are the indexes in the entity catalogue.
        #print('is_label_NIL:',is_label_NIL)
        old_src = src
        #print('context_input.size(0):',context_input.size(0))
        for i in range(context_input.size(0)):
            #oid += 1
            
            if use_BM25:
                # get indicies through BM25 - the candidate pool (sub-tokens, w or w/o synonyms, according to the input) is used to match with mention sub-tokens
                inds, _ = get_ranking_indices_w_BM25(mention_input[i].cpu().tolist(),list_2d_candidate_title_ids,topn=top_k*aggregating_factor,
                index_title_special_token=index_title_special_token)
                # aggregating results
                inds = np.array(_aggregating_indices_synonyms(inds,local_id2wikipedia_id,wikipedia_id2local_id,top_k))
                #inds = inds.tolist()
                # add NIL to the end            
                if add_NIL:
                    if not NIL_ent_id in inds:
                        inds[-1] = NIL_ent_id

                #inds = torch.tensor(inds)
            else:    
                inds = indicies[i]
                
                if srcs[i] != old_src:
                    src = srcs[i].item()
                    # not the same domain, need to re-do
                    new_scores = reranker.score_candidate(
                        context_input[[i]], 
                        None,
                        cand_encs=cand_encode_list[src].to(device)
                    )
                    _, inds = new_scores.topk(top_k)
                    inds = inds[0]
            #if i<=3:
            #   print('cand inds (first %d/3):' % i, inds)
            pointer = -1
            is_pointer_NIL = False
            for j in range(top_k):
                label_id = label_ids[i].item()
                label_id_normalised = _normalise_local_id(label_id,local_id2wikipedia_id,wikipedia_id2local_id)
                if inds[j].item() == label_id_normalised: #label_ids[i].item():
                    pointer = j
                    is_pointer_NIL = is_label_NIL[i]
                    break
            stats[src].add(pointer)
            
            #print('save_predictions:',save_predictions)            
            #print('save_true_predictions_only:',save_true_predictions_only)            
            if pointer == -1 and save_true_predictions_only: # not save predictions when the gold is not predicted in the top-k; otherwise, save all predictions
                continue
            if not save_predictions:
                continue
            
            # get current, topk candidates' token ids
            # transform inds (with syns as rows counted) to original inds (w only each entity as a row)
            inds_ori_local_id = [_normalise_to_ori_local_id(ind,local_id2wikipedia_id,wikipedia_id2_ori_local_id) for ind in inds]
            #if i<=3:
            #   print('inds_ori_local_id (first %d/3):' % i, inds_ori_local_id)
            #cur_candidates = candidate_pool[src][inds]
            cur_candidates = candidate_pool[srcs[i].item()][inds_ori_local_id]
            #print('cur_candidates:',cur_candidates,cur_candidates.size())

            # get features: does the mention have matching name in the entities
            #print('label_input:',len(label_input),label_input.size()) #label_input: 32 torch.Size([32, 128])
            #is_men_str_matchable_fts = get_is_men_str_matchable_features(mention_input[i].cpu().tolist(), label_input.cpu().tolist()) # search in a batch of labels
            #is_men_str_matchable_fts = get_is_men_str_matchable_features(mention_input[i].cpu().tolist(), candidate_pool[0].cpu().tolist()) # search in all labels from the entity catelogue

            #if not use_BM25:
            if get_is_men_str_mat_fts:
                is_men_str_matchable_fts = get_is_men_str_matchable_features(mention_input[i].cpu().tolist(), cur_candidates.cpu().tolist(),index_title_special_token=index_title_special_token) # search in the topk labels
                nn_is_mention_str_matchable_fts.append(is_men_str_matchable_fts)
            #else:
            #    is_men_str_matchable_fts = []
            # add examples in new_data
            nn_context.append(context_input[i].cpu().tolist())
            nn_candidates.append(cur_candidates.cpu().tolist())
            nn_labels.append(pointer)
            nn_labels_is_NIL.append(is_pointer_NIL)
            nn_entity_inds.append(inds)#(inds.data.cpu().numpy())            
            nn_worlds.append(src)

    # the stats and res below are only for zero-shot senario.
    if is_zeshel:
        res = Stats(top_k)
        for src in range(world_size):
            if stats[src].cnt == 0:
                continue
            if is_zeshel:
                logger.info("In world " + WORLDS[src])
            output = stats[src].output()
            logger.info(output)
            res.extend(stats[src])

        logger.info(res.output())

    nn_context = torch.LongTensor(nn_context)
    nn_candidates = torch.LongTensor(nn_candidates)
    nn_labels = torch.LongTensor(nn_labels)
    nn_labels_is_NIL = torch.Tensor(nn_labels_is_NIL).bool()
    if get_is_men_str_mat_fts:
        nn_is_mention_str_matchable_fts = torch.Tensor(nn_is_mention_str_matchable_fts)
    nn_data = {
        'context_vecs': nn_context,
        'candidate_vecs': nn_candidates,
        'labels': nn_labels,
        'labels_is_NIL': nn_labels_is_NIL, # whether the label is NIL - bool type, Tensor
        'entity_inds': nn_entity_inds, # the predicted entity indices from the bi-encoder, a list of np_arrays
        'mention_matchable_fts': nn_is_mention_str_matchable_fts if get_is_men_str_mat_fts else None, # mention matchable features
    }
    print('nn_data[\'labels\']:',nn_data['labels'])
    num_tp = len(nn_data['labels'][nn_data['labels'] != -1]) # get the ones not -1, i.e. gold in topk candidates
    num_ori_data = len(train_dataloader.dataset)
    logger.info('num of nn_data: %d' % len(nn_data['labels']))
    logger.info('biencoder recall@k: %.2f (%d/%d)' % (float(num_tp)/num_ori_data, num_tp, num_ori_data))
    #print('num of nn_data:',len(nn_data['entity_inds']))
    if is_zeshel:
        nn_data["worlds"] = torch.LongTensor(nn_worlds)
    
    return nn_data