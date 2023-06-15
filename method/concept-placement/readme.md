# Edge-bi-encoder

The edge-bi-encoder and evaluation with the data can be run with the script `step_all_BLINKout+_eval_bienc.sh`

# Prompting GPT 3.5

A sample of the prompt is below, where the `context_left`, `mention`, and `context_right` are from the data and the `options` are from the results (top-50) of the Edge-Bi-encoder:

```
Can you identify the correct ontological edges for the given mention based on the provided context (context_left and context_right)? The ontological edge consists of a pair where the left concept represents the parent of the mention, and the right concept represents the child of the mention. If the mention is a leaf node, the right side of the edges will be NULL (SCTID_NULL). If the context is not relevant to the options, make your decision solely based on the mention itself. There may be multiple correct options. Please answer briefly using option numbers, separated by a colon. If none of the options is correct, please answer None.

context_left:
Palmoplantar pustulosis - a cross-sectional analysis in Germany Palmoplantar pustulosis (PPP) is a recalcitrant chronic 

mention:
inflammatory skin disease

context_right:
. Data relevant for the medical care of patients with PPP are scarce. Thus, the aim of this work was to investigate the disease burden, clinical characteristics, and comorbidity of PPP patients in Germany. PPP patients were examined in a crosssectional study at seven specialized psoriasis centers in Germany. Of the 172 included patients with PPP, 79.1% were female and 69.8% were smokers .In addition, 25.0% suffered from psoriasis vulgaris, 28.2% had documented psoriatic arthritis, and 30.2% had a family history of psoriasis. In 77 patients the mean Dermatology Life Quality Index (DLQI) was 12.2 ± 7.7 (mean ± SD). The mean Psoriasis Palmoplantar Pustulosis Area and Severity Index (PPPASI) was 12.6 ± 8.6. Mean body mass index was above average at 27.1 ± 5.5. The PPP patients

options:
0.disorder of skin (disorder) (95320005) -> histologic type of inflammatory skin disorder (disorder) (277408007)
1.disorder by body site (disorder) (123946008) -> inflammation of skin and/or subcutaneous tissue (disorder) (363168001)
2.disorder of skin (disorder) (95320005) -> inflammation of perioral skin fold (disorder) (446932005)
3.disorder of skin (disorder) (95320005) -> immunobullous disease (disorder) (402716004)
4.disorder of skin and/or subcutaneous tissue (disorder) (80659006) -> reaction to thorn and/or spine in skin (disorder) (403186009)
5.disorder of skin (disorder) (95320005) -> cutaneous munchausen syndrome by proxy (disorder) (403590001)
6.disorder of skin (disorder) (95320005) -> inflammatory hyperkeratotic dermatosis, acute (disorder) (123689005)
7.disorder of skin and/or subcutaneous tissue (disorder) (80659006) -> reaction to metallic ring, stud and/or infibulata in skin (disorder) (403187000)
8.disorder by body site (disorder) (123946008) -> collagen and elastic tissue disorders affecting skin (disorder) (238846003)
9.disorder of skin (disorder) (95320005) -> inflammation related to voluntary body tattooing (disorder) (109249009)
10.disorder by body site (disorder) (123946008) -> degenerative skin disorder (disorder) (396325007)
11.disorder of skin (disorder) (95320005) -> inflammation related to voluntary body piercing (disorder) (109248001)
12.disorder by body site (disorder) (123946008) -> fibrohistiocytic proliferation of the skin (disorder) (21985009)
13.disorder of skin and/or subcutaneous tissue (disorder) (80659006) -> dermatosis due to parasite (disorder) (402139000)
14.disorder of skin and/or subcutaneous tissue (disorder) (80659006) -> cutaneous inflammation due to cytotoxic therapy (disorder) (403637008)
15.disorder of skin and/or subcutaneous tissue (disorder) (80659006) -> neonatal dermatosis (disorder) (402795001)
16.disorder by body site (disorder) (123946008) -> disorder of skin appendage (disorder) (238714008)
17.disorder of skin (disorder) (95320005) -> infective dermatosis of perianal skin (disorder) (402709000)
18.disorder of skin (disorder) (95320005) -> dermatological pathomimicry (disorder) (231531009)
19.skin lesion (disorder) (95324001) -> weal (disorder) (247472004)
20.disorder by body site (disorder) (123946008) -> disorder of skin and/or subcutaneous tissue of flank (disorder) (127338001)
21.disorder by body site (disorder) (123946008) -> disorder of skin (disorder) (95320005)
22.disorder of skin (disorder) (95320005) -> factitious skin disease (disorder) (402736003)
23.disorder of skin (disorder) (95320005) -> skin deposits (disorder) (40449001)
24.disorder of integument (disorder) (128598002) -> skin disease attributable to corticosteroid therapy (disorder) (402753005)
25.skin lesion (disorder) (95324001) -> tache noire (disorder) (409987005)
26.atrophic condition of skin (disorder) (400190005) -> granulomatous slack skin disease (disorder) (277796003)
27.disorder of skin (disorder) (95320005) -> leukokeratosis of skin (disorder) (48810007)
28.disorder by body site (disorder) (123946008) -> foreign body dermatosis (disorder) (273965001)
29.disorder of skin and/or subcutaneous tissue (disorder) (80659006) -> dermatosis due to hair as foreign body (disorder) (402163008)
30.disorder by body site (disorder) (123946008) -> dermatosis associated with biotin deficiency (disorder) (402482007)
31.dermatitis (disorder) (182782007) -> staphylococcal scarlatina (disorder) (238427006)
32.dermatitis (disorder) (182782007) -> streptococcal intertrigo (disorder) (238411004)
33.disorder by body site (disorder) (123946008) -> dorsal dermal sinus (disorder) (239151003)
34.disorder of integument (disorder) (128598002) -> systemic diseases affecting skin (disorder) (238980001)
35.skin disorder due to physical agent and/or foreign substance (disorder) (105965008) -> hydroa vacciniforme (disorder) (200837006)
36.disorder of skin (disorder) (95320005) -> skin sinus (disorder) (271766002)
37.disorder of skin and/or subcutaneous tissue (disorder) (80659006) -> skin reaction to suture material (disorder) (238505000)
38.skin lesion (disorder) (95324001) -> post-inflammatory scarring (disorder) (402680006)
39.disorder of skin (disorder) (95320005) -> cutaneous polyarteritis nodosa (disorder) (239926000)
40.disorder of skin (disorder) (95320005) -> skin ulcer (disorder) (46742003)
41.disorder of skin (disorder) (95320005) -> acute febrile neutrophilic dermatosis (disorder) (84625002)
42.disorder of skin (disorder) (95320005) -> edematous skin (disorder) (95322002)
43.skin lesion (disorder) (95324001) -> skin lesion due to intravenous drug abuse (disorder) (403746009)
44.chronic disease of skin (disorder) (128236002) -> schnitzler syndrome (disorder) (402415001)
45.disorder of integument (disorder) (128598002) -> acute skin disorder (disorder) (127334004)
46.disorder of skin (disorder) (95320005) -> livedo reticularis (disorder) (238772004)
47.disorder of skin (disorder) (95320005) -> mucinosis affecting skin (disorder) (402721001)
48.disorder of skin (disorder) (95320005) -> skin lesion in drug addict (disorder) (402765009)
49.inflammation of specific body structures or tissue (disorder) (363170005) -> cutaneous inflammation due to cytotoxic therapy (disorder) (403637008)
```