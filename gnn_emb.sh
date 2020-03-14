python train_gnn.py --dataset cora --arch gat --n-epochs 300 --lr 0.005
python train_gnn.py --dataset citeseer --arch gat --n-epochs 300 --lr 0.005
python train_gnn.py --dataset pubmed --arch gat --n-epochs 300 --num-out-heads=8 --weight-decay=0.001 --lr 0.005
