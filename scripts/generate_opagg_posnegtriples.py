from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors
# import scipy

import numpy as np
import jsonlines, os

from tqdm import tqdm
# from collections import defaultdict
# from nltk.tokenize import word_tokenize
# import nltk
# nltk.download('wordnet')
# from nltk.corpus import wordnet




import argparse

import sys
print('Running OpAgg dataset script with args:')
print(sys.argv)

parser = argparse.ArgumentParser(
    description="Script to generate triples of (query, [pos targets], [neg targets])",
)
parser.add_argument(
 "--data_dir", type=str, metavar="PATH", default='./data/', help="Path to data folder"
)

parser.add_argument(
 "--dataset", type=str, metavar="PATH", default='space', help="Source dataset"
)


parser.add_argument(
 "--num_cores", type=int,  default=1, help=""
)
parser.add_argument(
 "--max_samples", type=int,  default=None, help=""
)
parser.add_argument(
 "--max_tokens", type=int,  default=None, help=""
)

parser.add_argument(
 "--min_pos_score", type=float,  default=0.5, help=""
)
parser.add_argument(
 "--min_overlap", type=float,  default=0.5, help=""
)

parser.add_argument("--ignore_neutral", action="store_true", help="Require contradiction for neg samples")

# parser.add_argument("--topic_scores", action="store_true", help="Include tfidf similarity in overall score")

parser.add_argument("--unsorted", action="store_true", help="Dont sort examples by tfidf similarity")

parser.add_argument("--sort_by_min_tfidf", action="store_true", help="Sort examples by min tfidf overlap")
parser.add_argument("--sort_by_max_tfidf", action="store_true", help="Sort examples by min tfidf overlap")

parser.add_argument("--topk_neighbours", action="store_true", help="Sort examples by min tfidf overlap")

parser.add_argument("--debug", action="store_true", help="")

args = parser.parse_args()

INPUT_DATASET_NAME = args.dataset

dataset_slug = INPUT_DATASET_NAME

MIN_OVERLAP = args.min_overlap

MIN_POS_ENTAILMENT = args.min_pos_score

dataset_slug += '-minoverlap{:}'.format(str(MIN_OVERLAP).replace('.',''))
dataset_slug += '-minpos{:}'.format(str(MIN_POS_ENTAILMENT).replace('.',''))

if args.ignore_neutral:
    MAX_NEG_ENTAILMENT = -0.2
    dataset_slug += '-ignoreneutral'
else:
    MAX_NEG_ENTAILMENT = 0.2

if args.topk_neighbours:
    dataset_slug += '-topkneighbours'

if args.unsorted:
    dataset_slug += '-unsorted'
# if args.topic_scores:
#     dataset_slug += '-topicscores'
# else:
#     dataset_slug += '-nliscores'
if args.sort_by_min_tfidf:
    dataset_slug += '-tfidfminsorted'
if args.sort_by_max_tfidf:
    dataset_slug += '-tfidfmaxsorted'

BSZ = 256
DEBUG_LIMIT=100000000000000000

if args.max_samples is not None:
    dataset_slug += '-LIMIT'+str(args.max_samples)

    

if args.debug:
    dataset_slug += '-DEBUG'
    DEBUG_LIMIT=10

print('Dataset slug will be:')
print(dataset_slug)
    
if INPUT_DATASET_NAME == 'space':
    DATASET_INPUT = f'opagg/{INPUT_DATASET_NAME}-filtered-all'
    DATASET_OUTPUT = f'opagg/{INPUT_DATASET_NAME}-triples/' + dataset_slug
elif INPUT_DATASET_NAME == 'space-25toks-1pronouns':
    DATASET_INPUT = f'opagg/space-filtered/space-filtered-25toks-1pronouns-charfilt-all'
    DATASET_OUTPUT = f'opagg-twostage/space-retrievaltriples-tfidf/' + dataset_slug
elif INPUT_DATASET_NAME == 'amasum-electronics-25toks-1pronouns':
    DATASET_INPUT = f'opagg/amasum-filtered/amasum-filtered-electronics-25toks-1pronouns-charfilt-all'
    DATASET_OUTPUT = f'opagg-twostage/amasum-retrievaltriples-tfidf/' + dataset_slug
elif INPUT_DATASET_NAME == 'amasum-electronics-25toks-0pronouns':
    DATASET_INPUT = f'opagg/amasum-filtered/amasum-electronics-filtered-25toks-0pronouns-charfilt-all'
    DATASET_OUTPUT = f'opagg-twostage/amasum-retrievaltriples-tfidf/' + dataset_slug
elif INPUT_DATASET_NAME == 'amasum-tools-25toks-0pronouns':
    DATASET_INPUT = f'opagg/amasum-filtered/amasum-tools-filtered-25toks-0pronouns-charfilt-all'
    DATASET_OUTPUT = f'opagg-twostage/amasum-retrievaltriples-tfidf/' + dataset_slug
elif INPUT_DATASET_NAME == 'amasum-shoes-25toks-0pronouns':
    DATASET_INPUT = f'opagg/amasum-filtered/amasum-shoes-filtered-25toks-0pronouns-charfilt-all'
    DATASET_OUTPUT = f'opagg-twostage/amasum-retrievaltriples-tfidf/' + dataset_slug
elif INPUT_DATASET_NAME == 'amasum-home-kitchen-25toks-0pronouns':
    DATASET_INPUT = f'opagg/amasum-filtered/amasum-home-kitchen-filtered-25toks-0pronouns-charfilt-all'
    DATASET_OUTPUT = f'opagg-twostage/amasum-retrievaltriples-tfidf/' + dataset_slug
elif INPUT_DATASET_NAME == 'amasum-sports-outdoors-25toks-0pronouns':
    DATASET_INPUT = f'opagg/amasum-filtered/amasum-sports-outdoors-filtered-25toks-0pronouns-charfilt-all'
    DATASET_OUTPUT = f'opagg-twostage/amasum-retrievaltriples-tfidf/' + dataset_slug
else:
    raise Exception("Unknown input dataset!")

DATA_PATH = args.data_dir


os.makedirs(os.path.join(DATA_PATH, DATASET_OUTPUT), exist_ok=True)



with jsonlines.open(os.path.join(DATA_PATH, DATASET_INPUT, 'reviews.train.jsonl')) as f:
    input_rows = list(f)
with jsonlines.open(os.path.join(DATA_PATH, DATASET_INPUT, 'reviews.dev.jsonl')) as f:
    input_rows_dev = list(f)


# Dedupe
input_rows_filtered = []
train_sents = set()
for row in input_rows:
    if row['sentence'] in train_sents:
        continue
    train_sents.add(row['sentence'])
    input_rows_filtered.append(row)

input_rows_filtered_dev = []
dev_sents = set()
for row in input_rows_dev:
    if row['sentence'] in dev_sents:
        continue
    dev_sents.add(row['sentence'])
    input_rows_filtered_dev.append(row)

all_sents = [x['sentence'] for x in input_rows_filtered ]
all_sents_dev = [x['sentence'] for x in input_rows_filtered_dev]

print('Loaded {:} train and {:} dev sentences'.format(len(all_sents), len(all_sents_dev)))
 

embedded = None
embedded_dev = None
# all_sents_bow = [set(word_tokenize(sent)) for sent in all_sents]
# all_sents_dev_bow = [set(word_tokenize(sent)) for sent in all_sents_dev]


vectorizer = TfidfVectorizer(
    stop_words=('english'),
    min_df=5,
    # max_df=0.8 if args.ignore_stopwords else 1.0,
    max_df=0.5,
)

embedded = vectorizer.fit_transform(all_sents)

print('Transforming dev')
embedded_dev = vectorizer.transform(all_sents_dev)
# all_sents_bow = [None for sent in all_sents]
# all_sents_dev_bow = [None for sent in all_sents_dev]

import time
start = time.time()
print('Finding nearest neighbours...')
print('** This can take a few hours for a dataset with 1m input samples **')
from sklearn.neighbors import NearestNeighbors

from joblib import parallel_backend, Parallel, delayed

import sklearn


if args.num_cores > 1:
    if args.unsorted:
        raise Exception('Unsorted + multithreading is not implmented yet')
    with sklearn.config_context(working_memory=4096):

        num_cores = args.num_cores
        print('using {:} cores'.format(num_cores))
        print(' - Fitting train')
        nbrs = NearestNeighbors(algorithm='auto', metric='cosine').fit(embedded)
        print(' - Searching train')
        # distances, indices = nbrs.radius_neighbors(embedded[:args.max_samples], radius=1-MIN_OVERLAP)

        def get_neighbours(x, nbrs):
            # distances, indices = nbrs.radius_neighbors(x, radius=1-MIN_OVERLAP, sort_results=True)
            distances, indices = nbrs.kneighbors(x, n_neighbors=26)
            # return starting from 1, to not include self
            
            return distances[0][1:26], indices[0][1:26]

        res = Parallel(n_jobs=num_cores, backend='loky', batch_size='auto')(delayed(get_neighbours)(embedded[i], nbrs) for i in tqdm(range(embedded[:args.max_samples][:DEBUG_LIMIT].shape[0])))

        distances, indices = list(zip(*res))

        print(' - Fitting dev')
        nbrs_dev = NearestNeighbors(algorithm='auto', metric='cosine').fit(embedded_dev)
        print(' - Searching dev')
        # distances_dev, indices_dev = nbrs_dev.radius_neighbors(embedded_dev[:args.max_samples], radius=1-MIN_OVERLAP)
        distances_dev, indices_dev = list(zip(*Parallel(n_jobs=num_cores, backend='loky', batch_size='auto')(delayed(get_neighbours)(embedded_dev[i], nbrs_dev) for i in tqdm(range(embedded_dev[:args.max_samples][:DEBUG_LIMIT].shape[0])))))
else:
    nbrs = NearestNeighbors(algorithm='auto', metric='cosine').fit(embedded)
    if args.topk_neighbours:

        distances, indices = nbrs.kneighbors(embedded[:args.max_samples], n_neighbors=26)
        distances, indices = distances[:, 1:26], indices[:, 1:26]
    else:
        distances, indices = nbrs.radius_neighbors(embedded[:args.max_samples], radius=1-MIN_OVERLAP)
        distances, indices = [dists[:51] for dists in distances], [idxs[:51] for idxs in indices]
    
    
    nbrs_dev = NearestNeighbors(algorithm='auto', metric='cosine').fit(embedded_dev)
    if args.topk_neighbours:
        distances_dev, indices_dev = nbrs_dev.kneighbors(embedded_dev[:args.max_samples], n_neighbors=26)
        
        distances_dev, indices_dev = distances_dev[:, 1:26], indices_dev[:, 1:26]
    
    else:
        distances_dev, indices_dev = nbrs_dev.radius_neighbors(embedded_dev[:args.max_samples], radius=1-MIN_OVERLAP)
        distances_dev, indices_dev = [dists[:51] for dists in distances_dev], [idxs[:51] for idxs in indices_dev]
    
    

end = time.time()
print('done in {:}s'.format(end-start))




print('Building dataset')


from torchseq.pretrained.nli import PretrainedNliModel

nli_model = PretrainedNliModel()

triples = []
triples_dev = []





print('Checking NLI scores (train)')
all_candidates = []
for i,x in enumerate(all_sents[:args.max_samples]):
    this_sims, topk_indices = 1 - distances[i], indices[i]

    if len(topk_indices) < 10:
        all_candidates.append([])
        continue

    if this_sims[0] < 0.2:
        all_candidates.append([])
        continue


    all_candidates.append([(all_sents[ix], this_sims[j]) for j, ix in enumerate(topk_indices) if ix != i])
    

prem_hyp_pairs = [(sent, candidate[0]) for i, sent in enumerate(all_sents[:args.max_samples]) for candidate in all_candidates[i]]
all_premises,all_hypotheses = zip(*prem_hyp_pairs)


if args.ignore_neutral:
    all_nli_scores = nli_model.get_scores(premises=all_premises, hypotheses=all_hypotheses, bsz=BSZ, progress=True, return_entailment_prob=False)
    all_nli_scores = [x['ENTAILMENT'] - x['CONTRADICTION'] for x in all_nli_scores]
else:
    all_nli_scores = nli_model.get_scores(premises=all_premises, hypotheses=all_hypotheses, bsz=BSZ, progress=True)

grouped_nli_scores = []
i=0
for candidates in all_candidates:
    score_group = []
    for cand, topic_score in candidates:
        score_group.append((all_nli_scores[i], topic_score))
        i+=1
    grouped_nli_scores.append(score_group)

if len(grouped_nli_scores) != len(all_candidates):
    print('Number of NLI groups and sent groups is different!')
    exit()


for x, candidates, nli_scores in zip(all_sents[:args.max_samples], all_candidates, grouped_nli_scores):
    assert len(candidates) == len(nli_scores), "Candidates and scores have different lengths!"
    pos_targets = [(sent[0], nli_score, topic_score) for sent, (nli_score, topic_score) in zip(candidates, nli_scores) if nli_score > MIN_POS_ENTAILMENT]
    neg_targets = [(sent[0], nli_score, topic_score) for sent, (nli_score, topic_score) in zip(candidates, nli_scores) if nli_score < MAX_NEG_ENTAILMENT]

    if len(pos_targets) > 0 and len(neg_targets) > 0:
        triples.append({
            'query': x,
            'pos_targets': pos_targets,
            'neg_targets': neg_targets,
        })

print('Checking NLI scores (dev)')
all_candidates_dev = []
for i,x in enumerate(all_sents_dev[:args.max_samples]):
    this_sims, topk_indices = 1 - distances_dev[i], indices_dev[i]

    if len(topk_indices) < 10:
        all_candidates_dev.append([])
        continue

    all_candidates_dev.append([(all_sents_dev[ix], this_sims[j]) for j, ix in enumerate(topk_indices) if ix != i])

prem_hyp_pairs_dev = [(sent, candidate[0]) for i, sent in enumerate(all_sents_dev[:args.max_samples]) for candidate in all_candidates_dev[i]]
all_premises_dev,all_nli_scores_dev = zip(*prem_hyp_pairs_dev)


if args.ignore_neutral:
    all_nli_scores_dev = nli_model.get_scores(premises=all_premises_dev, hypotheses=all_nli_scores_dev, bsz=BSZ, progress=True, return_entailment_prob=False)
    all_nli_scores_dev = [x['ENTAILMENT'] - x['CONTRADICTION'] for x in all_nli_scores_dev]
else:
    all_nli_scores_dev = nli_model.get_scores(premises=all_premises_dev, hypotheses=all_nli_scores_dev, bsz=BSZ, progress=True)

grouped_nli_scores_dev = []
i=0
for candidates in all_candidates_dev:
    score_group = []
    for cand, topic_score in candidates:
        score_group.append((all_nli_scores_dev[i], topic_score))
        i+=1
    grouped_nli_scores_dev.append(score_group)


for x, candidates, nli_scores in zip(all_sents_dev[:args.max_samples], all_candidates_dev, grouped_nli_scores_dev):
    pos_targets = [(sent[0], nli_score, topic_score) for sent, (nli_score, topic_score) in zip(candidates, nli_scores) if nli_score > MIN_POS_ENTAILMENT]
    neg_targets = [(sent[0], nli_score, topic_score) for sent, (nli_score, topic_score) in zip(candidates, nli_scores) if nli_score < MAX_NEG_ENTAILMENT]

    

    if len(pos_targets) > 0 and len(neg_targets) > 0:
        triples_dev.append({
            'query': x,
            'pos_targets': pos_targets,
            'neg_targets': neg_targets,
        })



with jsonlines.open(os.path.join(DATA_PATH, DATASET_OUTPUT, 'clusters.train.jsonl'), 'w') as f:
    f.write_all(triples)

# dev sources should be deterministic
with jsonlines.open(os.path.join(DATA_PATH, DATASET_OUTPUT, 'clusters.dev.jsonl'), 'w') as f:
    f.write_all(triples_dev)

# Now construct the actual training data

MAX_TRIPLES = 5

import numpy as np

for split, rows in [('train', triples), ('dev', triples_dev)]:

    all_triples = []

    for row in rows:
        combos = []
        for pos_tgt in row['pos_targets']:
            # for neg_tgt in row['neg_targets']:
            neg_tgt = row['neg_targets'][np.random.choice(range(len(row['neg_targets'])))]
            
            combos.append({
                'query': row['query'],
                'pos_target': pos_tgt[0],
                'neg_target': neg_tgt[0],
                'pos_tfidf_score': pos_tgt[2],
                'neg_tfidf_score': neg_tgt[2],
                'pos_nli_score': pos_tgt[1],
                'neg_nli_score': neg_tgt[1],
            })
        if args.sort_by_min_tfidf:
            # Sort by tfidf similarity ascending
            combos = sorted(combos, key = lambda x: x['pos_tfidf_score'], reverse=False)
        elif args.sort_by_max_tfidf:
            # Sort by tfidf similarity DESCENDING
            combos = sorted(combos, key = lambda x: x['pos_tfidf_score'], reverse=True)
        else:
            np.random.shuffle(combos)
        all_triples.extend(combos[:MAX_TRIPLES])
        
    print(split, len(all_triples))

    with jsonlines.open(os.path.join(DATA_PATH, DATASET_OUTPUT, f'{split}.jsonl'),'w') as writer:
            writer.write_all(all_triples)