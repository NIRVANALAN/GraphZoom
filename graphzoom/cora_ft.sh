#!/bin/bash
for i in `seq 1 2`    ## coarsening level
    do
        ratio=$(case "$i" in
            (1)  echo 2;;
            (2)  echo 5;;
            # (3)  echo 9;;
            # (4)  echo 19;;
            # (5)  echo 100;;
        esac)
        python graphzoom.py -r ${ratio} -m ft  --proj border --embed_level 3
        #python graphzoom.py -r ${ratio} -m deepwalk -f
done
