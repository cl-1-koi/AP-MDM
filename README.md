# 🔋 On Powerful Ways to Generate: Autoregression, Diffusion, and Beyond

This is the official implementation for paper "On Powerful Ways to Generate: Autoregression, Diffusion, and Beyond". 

<!-- We propose masked diffusion models with Any-Process Generation (AP-MDM). -->

Materials: 
[Paper](https://arxiv.org/pdf/2510.06190)

<p align="center">
    <img width="800" src="assets/overview.png">
</p>


## Installation

- Install other required packages via

```bash
pip install -r requirements.txt
```

- Install flash-attention

```bash
pip install flash-attn --no-build-isolation
```

## Datasets

### Sudoku

```bash
cd dataset/sudoku
python sudoku_generator.py sudoku-100.npy
```

This generates `sudoku-100.pkl.gz` containing APMDM training samples and `vocab_cache.pkl` containing the token vocabulary. You can process multiple files at once: `python sudoku_generator.py sudoku-100.npy sudoku-test.npy`. To visualize the generation process, run `python serve.py` and open http://localhost:8001/apmdm in your browser.

### Parity

```bash
cd dataset/parity
python parity_generator.py
```

This generates `parity_train.pkl.gz` containing 7 APMDM training samples (expanded to 1000 with repetition) and `parity_vocab_cache.pkl` containing the token vocabulary (5 tokens: BOS, EOS, MASK, 0, 1).

### Graph Generation

```bash
cd dataset/max_flow
python maxflow_solver.py \
  --num_instances 10000 \
  --min_nodes 10 --max_nodes 10 \
  --min_edges 50 --max_edges 50 \
  --output graph.pkl.gz
```

This generates `graph.pkl.gz` containing APMDM training samples for max-flow problems and `vocab_cache.pkl` containing the token vocabulary. Customize graph parameters: `--num_instances` (number of graphs), `--min_nodes/max_nodes` (node count range), `--min_edges/max_edges` (edge count range), `--min_flow/max_flow` (flow guarantee range).


## Training and Evaluation

See scripts in `train/scripts`.

> **Note:** In our implementation, we use the word *contraction* for *deletion* and *expansion* for *insertion*. **R**, **E**, **C** denote *remasking*, *expansion/insertion*, and *contraction/deletion* signals, respectively.

## Citation

If you find our codes useful, please consider citing our work

```bibtex
@article{yang2025powerful,
  title={On Powerful Ways to Generate: Autoregression, Diffusion, and Beyond},
  author={Yang, Chenxiao and Zhou, Cai and Wipf, David and Li, Zhiyuan},
  journal={arXiv preprint arXiv:2510.06190},
  year={2025}
}
```

## Acknowledgement

The training pipeline is adapted from [MDLM](https://github.com/kuleshov-group/mdlm).