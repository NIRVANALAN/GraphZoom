#!/bin/bash
for i in `seq 1 2`    ## coarsening level
    do
        ratio=$(case "$i" in
            (1)  echo 2;;
            (2)  echo 5;;
            #(3)  echo 12;;
            #(4)  echo 25;;
            #(5)  echo 50;;
        esac)
        # python graphzoom.py -m ft -d citeseer -r ${ratio} -n 10 
        # python graphzoom.py -m ft -d citeseer -r ${ratio} -n 10 -emb_arch GAT
        python graphzoom.py -m ft -d citeseer -r ${ratio} -n 10 -emb_arch GCN --proj border
        # python graphzoom.py -m ft -d citeseer -r ${ratio} -n 10 -emb_arch GCN
        #python graphzoom.py -m deepwalk -d citeseer -r ${ratio} -n 10 -f
done
