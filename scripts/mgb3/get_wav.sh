for type in dev test train ; do
    ( cd ../../data/mgb3/$type/
      mkdir wav; cd wav
      cat ../wav.lst | while read wav; do 
        wget $wav
      done 
      for lang in EGY GLF LAV MSA NOR; do
        mkdir -p $lang
        awk '{print $1}' ../$lang.words | while read id; do
            mv $id.wav $lang
        done
      done
      rm *.wav  # more audio probably failed to extract word list,ivec and phoneme
    )
done