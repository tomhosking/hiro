# [HIRO: Hierarchical Indexing for Retrieval-Augmented Opinion Summarization](https://arxiv.org/abs/2403.00435)

Tom Hosking, Hao Tang and Mirella Lapata

## Setup

Create a fresh environment:

```sh
conda create -n hiroenv python=3.9
conda activate hiroenv
```
or 
```sh
python3 -m venv hiroenv
source hiroenv/bin/activate
```

Then install dependencies:

```sh
pip install -r requirements.txt
```

TODO: Upload data and checkpoints
Download data/models:
 - <a href="https://tomho.sk/hiro/data/hiro_data_shared.zip" download>Shared (tokenizers etc)</a> -> `./data`
 - <a href="https://tomho.sk/hiro/data/hiro_data_space.zip" download>Space</a> -> `./data`
 - <a href="https://tomho.sk/hiro/data/hiro_data_amasum.zip" download>AmaSum</a> -> `./data`
 - [Trained checkpoints](http://tomho.sk/hiro/models/) -> `./models`

Tested with Python 3.9.

## Generating Summaries

First, run the pre-processing eval recipe:

```sh
torchseq-eval --recipe opagg.hiro_pre --model ./models/20240130_183901_d671_space --test
```

TODO: Which files does this read/write?

Then, get generations from your preferred LLM (we used Mistral 7B Instruct v0.2), based on the prompts in the files `eval/llm_inputs_piecewise_test.jsonl` and  `eval/llm_inputs_oneshot_test.jsonl`. You might want to use my [TGI Client](https://github.com/tomhosking/tgi-client) to run efficient batched inference through a HuggingFace model:
```sh
python tgi-client/runner.py --input runs/hiro/space/llm_inputs_oneshot_test.jsonl --output runs/hiro/space/llm_outputs_oneshot_test_mistaral7b.js
onl --model mistralai/Mistral-7B-Instruct-v0.2
```

Then run the post-LLM eval recipe to get the scores:

```sh
torchseq-eval --recipe opagg.hiro_post --model ./models/20240130_183901_d671_space --test
```

## Training on Space/Amasum

(Optional) Rebuild the datasets:

Run the dataset filtering scripts `./scripts/opagg_filter_space.py` and `./scripts/opagg_filter_space_eval.py`


```sh
# SPACE
python scripts/generate_opagg_posnegtriples.py --dataset space-25toks-1pronouns  --min_pos_score 0.75 --ignore_neutral --unsorted --min_overlap 0.3

# AmaSum
python scripts/generate_opagg_posnegtriples.py --dataset amasum-electronics-25toks-0pronouns  --min_pos_score 0.75 --ignore_neutral --unsorted --min_overlap 0.3
python scripts/generate_opagg_posnegtriples.py --dataset amasum-shoes-25toks-0pronouns  --min_pos_score 0.75 --ignore_neutral --unsorted --min_overlap 0.3
python scripts/generate_opagg_posnegtriples.py --dataset amasum-sports-outdoors-25toks-0pronouns  --min_pos_score 0.75 --ignore_neutral --unsorted --min_overlap 0.3
python scripts/generate_opagg_posnegtriples.py --dataset amasum-home-kitchen-25toks-0pronouns  --min_pos_score 0.75 --ignore_neutral --unsorted --min_overlap 0.3

```

Train:

```sh
torchseq --train --reload_after_train --validate --config ./configs/hiro_space.json
```

## Training on a new dataset

Setting up to train on a new dataset can be a bit tricky - I'm happy to help run you through the process, just email me or raise an issueon Github.

- [ ] Make a copy of your dataset in a format expected by the preprocessing scripts
- [ ] Clean the training data and eval data 
- [ ] Generate training pairs
- [ ] Modify one of the configs to point to your data
- [ ] Train the model

```sh
torchseq --train --reload_after_train --validate --config ./configs/{YOUR_CONFIG}.json
```


## Citation

```
@misc{hosking2024hierarchical,
      title={Hierarchical Indexing for Retrieval-Augmented Opinion Summarization}, 
      author={Tom Hosking and Hao Tang and Mirella Lapata},
      year={2024},
      eprint={2403.00435},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
```