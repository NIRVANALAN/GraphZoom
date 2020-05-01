#python train_gnn.py --dataset cora --arch gat --n-epochs 300 --lr 0.005 --coarse --self-loop
#python train_gnn.py --dataset pubmed --arch gat --n-epochs 300 --num-out-heads=8 --weight-decay=0.001 --lr 0.005
#python train_gnn.py --dataset pubmed --arch gat --n-epochs 300 --num-out-heads=8 --weight-decay=0.001 --lr 0.005 --coarse
#
#python train_gnn.py --dataset Amazon2M --prefix /mnt/yushi/ --self-loop --num-hidden 128
#python train_gnn.py --dataset reddit  --self-loop --num-hidden 128 --prefix /mnt/yushi/

#coarsen
# python train_gnn.py --dataset pubmed --arch gat --n-epochs 300 --num-out-heads=8 --weight-decay=0.001 --lr 0.005 --coarse
# python train_gnn.py --dataset pubmed --arch gcn --n-epochs 300 --coarse
# python train_gnn.py --dataset cora --self-loop --coarse --gpu 1 --num-hidden 128 --coarse
# python train_gnn.py --dataset citeseer --arch gat --n-epochs 300 --lr 0.005 --coarse --self-loop 
# python train_gnn.py --dataset citeseer --arch gcn --n-epochs 300 --coarse --self-loop 
# python train_gnn.py --dataset reddit  --self-loop --num-hidden 128 --prefix /mnt/yushi/ --coarse
# python train_gnn.py --dataset Amazon2M --prefix /mnt/yushi/ --self-loop --num-hidden 128 --coarse
python train_gnn.py --dataset reddit  --arch gat --n-epochs 300 --num-out-heads=8 --weight-decay=0.001 --lr 0.005 --coarse --prefix /mnt/yushi/ --coarse
python train_gnn_refine_one_hot.py --dataset reddit --level 1 --proj one_hot --pre /data/data0/yushi/
