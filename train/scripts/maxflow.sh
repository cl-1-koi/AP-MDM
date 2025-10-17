/workspace/env/bin/python main.py --config-name=maxflow \
                data.dataset_path='/workspace/projects/GDLM/graph/data/graph.pkl.gz' \
                data.vocab_cache_path='/workspace/projects/GDLM/graph/data/vocab_cache.pkl' \
                model.vocab_size=60 \
                model.length=400 \
                wandb.offline=true \
                loader.num_workers=16 \
                checkpointing.save_dir='/workspace/projects/AP-MDM/train/outputs' \
                