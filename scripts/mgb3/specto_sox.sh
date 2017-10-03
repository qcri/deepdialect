
find ../data/ -name *.wav  > i$$
cat i$$ | sed -i 's:/wav/:/specto_sox/:' -e 's:.wav$:.png:' > e$$
cat e$$ | while read file; do base=$(dirname $file); echo $base; done | sort -u | xargs mkdir -p
counter=0
paste -d ' ' i$$ e$$ | while read  in out; do
  counter=$((counter+1))
  sox -V0 $in -n remix 1 rate 10k spectrogram -y 129 -X 50 -m -r -o $out
  if (( counter % 1000 == 0 )); then 
    echo "Processed $counter from: $in to: $out"
  fi
done

