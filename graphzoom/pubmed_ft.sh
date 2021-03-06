#!/bin/bash
for i in `seq 1 2`    ## coarsening level
    do
        ratio=$(case "$i" in
            (1)  echo 2;; 
            (2)  echo 5;;
            # (3)  echo 9;;
            # (4)  echo 27;;
            # (5)  echo 60;;
        esac)
        # python graphzoom.py -m deepwalk -d pubmed -r ${ratio} 
        # python graphzoom.py -m ft -d pubmed -r ${ratio} -emb_arch GAT
        python graphzoom.py -m ft -d pubmed -r ${ratio} -emb_arch GCN --proj border
    done
