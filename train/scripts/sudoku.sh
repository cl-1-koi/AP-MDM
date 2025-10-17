/workspace/env/bin/python main.py --config-name=sudoku \
                data.dataset_path='/workspace/projects/GDLM/sudoku/data/sudoku-selected-train.pkl.gz' \
                data.vocab_cache_path='/workspace/projects/GDLM/sudoku/data/vocab_sudoku.pkl' \
                model.vocab_size=31 \
                model.length=400 \
                wandb.offline=true \
                loader.num_workers=16 \
                checkpointing.save_dir='/workspace/projects/AP-MDM/train/outputs' \
