python train_gnn.py --dataset cora --arch gat --n-epochs 300
python train_gnn.py --dataset citeseer --arch gat --n-epochs 300
python train_gnn.py --dataset pubmed --arch gat --n-epochs 300 --num-out-heads=8 --weight-decay=0.001
