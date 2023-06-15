# prompting chatGPT

import openai
# TEST
openai.api_key = "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"
import os

# prompt = """
# Considering chronic kidney disease (marked with asterisk) in the context below: "Medication Intervention for Chronic Kidney Disease Patients Transitioning from Hospital to Home: Study Design and Baseline Characteristics The hospital readmission rate in the population with  *chronic kidney disease*  (CKD) is high and strategies to reduce this risk are urgently needed. The CKD - Medication Intervention Trial (CKD - MIT; www.clinicaltrials.gov; NCTO1459770) is a single-blind (investigators), randomized, clinical trial conducted at Providence Health Care in Spokane, Washington. Study participants are hospitalized patients with CKD stages 3-5 (not treated with kidney replacement therapy) and acute illness. The study intervention is a pharmacist-led, home-based, medication management intervention delivered within 7 days after hospital discharge. The primary outcome is a composite of hospital readmissions and visits to emergency departments and urgent care centers for 90 days following hospital discharge. Secondary outcomes are achievements of guideline-based targets for CKD risk factors and complications. Enrollment began in February 2012 and ended in May 2015. At baseline, the age of participants was" Is chronic kidney disease a direct child of renal impairment (disorder)? Please answer briefly with yes or no.
# """

# completion = openai.ChatCompletion.create(
#              model="gpt-3.5-turbo",
#              n = 1,
#              max_tokens=1024,
#              messages=[
#                {"role": "user", "content": prompt}
#               ]
#              )

def prompting(prompt,model="gpt-3.5-turbo"):
    completion = openai.ChatCompletion.create(
             model=model,
             n = 1,
             max_tokens=1024,
             messages=[
               {"role": "user", "content": prompt}
              ]
             )    
    return completion

def retrieve_results_from_arxiv(prompt, prompts_arxiv_df, model="gpt-3.5-turbo"):
    return prompts_arxiv_df[prompts_arxiv_df['prompt'] == prompt][model].to_string(index=False)

snomed_subset='CPP'
top_k_base_value = 500
top_k_value = 50
filter_by_degree = True
prompts_fn = "../../models/biencoder/mm+%s2017AA-tl-pubmedbert-NIL-bs128/top%d_candidates/test-NIL-top%d-preds%s-prompts-by-edges.csv" % (snomed_subset,top_k_base_value,top_k_value, "-degree-1" if filter_by_degree else "")
model="gpt-3.5-turbo"

prompts_arxiv_fn = "../../models/biencoder/mm+%s2017AA-tl-pubmedbert-NIL-bs128/top%d_candidates/test-NIL-top%d-preds%s-prompts-by-edges-gpt3.5-arxiv.csv" % (snomed_subset,top_k_base_value,top_k_value,"-degree-1" if filter_by_degree else "")

import pandas as pd
# avoid ... in showing long sequence
pd.set_option("display.max_colwidth", 10000)
from tqdm import tqdm

saving_step=100

prompts_df = pd.read_csv(prompts_fn,index_col=0)
print('prompts_df:',prompts_df.head())

use_arxiv_prompt_answers = False
if os.path.isfile(prompts_arxiv_fn):
    print('arxiv prompt file found')
    use_arxiv_prompt_answers = True
    prompts_arxiv_df = pd.read_csv(prompts_arxiv_fn,index_col=0)
    print('prompts_arxiv_df:',prompts_arxiv_df.head())

prompts_df[model] = ""
for i, row in tqdm(prompts_df.iterrows(),total=len(prompts_df)):
    if i != 0 and i % saving_step == 0:
        prompts_df.to_csv(prompts_fn[:len(prompts_fn)-4] + '-gpt3.5-step%d.csv' % i,index=True)
    # if i==3:
    #     break
    if str(row["answer"]) == "-1":
        continue
    #if i != 3301:
    #    continue
    prompt = row["prompt"]
    #answer = ""
    if use_arxiv_prompt_answers:
        if prompt in prompts_arxiv_df["prompt"].values:
            answer = retrieve_results_from_arxiv(prompt, prompts_arxiv_df, model=model)
        else:
            # not available in arxiv - thus prompt now
            completion_json = prompting(prompt,model=model)
            answer = completion_json["choices"][0]["message"]["content"]
            prompts_df.at[i,model] = answer
    #print(answer)
    else:
        # prompt now
        completion_json = prompting(prompt,model=model)
        answer = completion_json["choices"][0]["message"]["content"]
        prompts_df.at[i,model] = answer

#prompts_df.to_csv(prompts_fn[:len(prompts_fn)-4] + '-gpt3.5-by-arxiv.csv',index=True)
prompts_df.to_csv(prompts_fn[:len(prompts_fn)-4] + '-gpt3.5.csv',index=True)