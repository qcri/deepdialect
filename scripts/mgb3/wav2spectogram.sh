dialectlist=('EGY' 'GLF' 'MSA' 'NOR' 'LAV')

#This script will make spectrogram for each audio file in the split folder. 
#The spectrogram in mono 192*192 resolution

find ../data -type d | grep split_audio$ | while read indir; do
  image=$(echo $indir | sed 's:/split_audio/:/spectrogram/:')
  for dialect in ${dialectlist[*]}; do
     mkdir -p ${image}/$dialect
	 echo "Processing audio from ${indir}/$dialect  to spectrogram in ${image}/$dialect"
     python wav2spectogram.py -i ${indir}/$dialect -o ${image}/$dialect
  done 
done 

