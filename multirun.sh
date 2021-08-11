#!/bin/sh
DIR='/experiments/'
# loop over files in folder experiments
for FILE in ls "$DIR"*
do
	filename=$(basename "$FILE")
	# if not simple_tagger then run and wait
	if [["$FILENAME" == "simple_tagger.json"]]
		wait # do nothing
	else
		fname="${filename%.*}"
		python run_tagger.py -c "$FILE" -m "/mnt/sda/mwe/$fname"
		wait
done