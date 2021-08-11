#!/bin/sh
DIR='experiments/'
# loop over files in folder experiments
for FILE in $(ls "$DIR"); do
	filename=$(basename "$FILE")
	echo "$filename"
	# if not simple_tagger then run and wait
	if [[ "$filename" == *"simplest_tagger"* ]]; then
		echo "Simplest tagger ignored"
		wait # do nothing
	else
		# run and store model on HDD
		fname="${filename%.*}"
		python run_tagger.py -c "$DIR$filename" -m "/mnt/sda/mwe/$fname"
		wait
	fi
done