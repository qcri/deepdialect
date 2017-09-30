#!/bin/bash -e


# this script will split all train dev and test wav files to 5...30 seconds each 


dialectlist=('EGY' 'GLF' 'MSA' 'NOR' 'LAV')
intervallist=(30 25 20 15 10 5)



find ../data -type d | grep wav$ | while read indir; do
  #outdir=$(echo $indir | sed 's:/data/sls/qcri/asr/data/vardial/vardial2017:../data/:')
  echo "Proceesing $indir"
  for dialect in ${dialectlist[*]}; do
    indir2=${indir}/$dialect
	outdir=$(echo $indir2 | sed 's:/wav/:../split_audio/:')
	echo $indir2 $outdir
	mkdir -p $outdir
	(cd $indir2; ls *wav) | while read wav; do
	  for interval in ${intervallist[*]}; do
	    ./split.py $indir2 $wav $outdir $interval
	  done 
	done 
  done
  echo '###'
done 

