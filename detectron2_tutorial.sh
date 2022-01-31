#!/bin/bash
#PBS -l select=1:ncpus=1:ngpus=1:mem=8gb:cl_konos=False:cl_gram=False:scratch_local=1gb
#PBS -l walltime=02:00:00 -q gpu
# modify/delete the above given guidelines according to your job's needs
# Please note that only one select= argument is allowed at a time.

# # PBS -l select=1:ncpus=1:mem=1gb:scratch_local=4gb

# add to qsub with:
# qsub detectron2_tutorial.sh

# nastaveni domovskeho adresare, v promenne $LOGNAME je ulozeno vase prihlasovaci jmeno
LOGDIR="/auto/plzen1/home/$LOGNAME/projects/tutorials/metacentrum/detectron2_cells/"
PROJECTDIR="/auto/plzen1/home/$LOGNAME/projects/tutorials/metacentrum/detectron2_cells/"
DATADIR="/auto/plzen1/home/$LOGNAME/data/cells/"


echo "job: $PBS_JOBID running on: `uname -n`"

lscpu

# nastaveni automatickeho vymazani adresare SCRATCH pro pripad chyby pri behu ulohy
trap 'clean_scratch' TERM EXIT

# vstup do adresare SCRATCH, nebo v pripade neuspechu ukonceni s chybovou hodnotou rovnou 1
cd $SCRATCHDIR || exit 1

echo "scratchdir=$SCRATCHDIR"

# priprava vstupnich dat (kopirovani dat na vypocetni uzel)
# vytvoreni adresaru na SCRATCHDIR
mkdir -p "$SCRATCHDIR/data/orig"
mkdir -p "$SCRATCHDIR/data/processed"

cd $SCRATCHDIR
ls -l

# vytvoreni adresare processed v adresari DATADIR
mkdir "$DATADIR/processed"

# kopirovani dat z DATADIR do SCRATCHDIR do data/orig
cp -r $DATADIR "$SCRATCHDIR/data/orig"

# spusteni aplikace - samotny vypocet

# activate environment option 1: miniconda installed
module add cuda-10.1
module add conda-modules-py37
module add gcc-8.3.0

#source conda activate drawnUI-conda
conda activate /auto/plzen1/home/$LOGNAME/miniconda3/envs/drawnUI-conda



#export PATH=/storage/plzen1/home/$LOGNAME/miniconda3/bin:$PATH
#source activate mytorch


# this is because of python click
export LC_ALL=C.UTF-8
export LANG=C.UTF-8


# Put your code here
python  $PROJECTDIR/detectron2_cells.py > results.txt

ls
# kopirovani vystupnich dat z vypocetnicho uzlu do domovskeho adresare,
# pokud by pri kopirovani doslo k chybe, nebude adresar SCRATCH vymazan pro moznost rucniho vyzvednuti dat
cp results.txt $LOGDIR || export CLEAN_SCRATCH=true
cp -r $SCRATCHDIR/data/processed $DATADIR/processed || export CLEAN_SCRATCH=true