#!/usr/bin/env bash

cd 0-all

# get the results of specific dataset, say DSTC2
find . -name '*csv' | grep 'dstc2'

# concatenate multiple csv files, ignore the header of csv tables
find . -name '*csv' | grep 'dstc2' | xargs tail -n +2 -q | wc -l

# get the results of specific column(s), and print with a new delimiter \t
# c12 is the testf1 in the train-valid-test output
find . -name '*csv' | grep 'dstc2' | xargs tail -n +2 -q | cut -d',' -f3,4,6- | tr ',' '\t'


# go into a specific folder
cd 0-all/20171026-100604.context=next.feature=0-all.similarity=false
cat dstc2.valid_test.csv | tail -n +2 -q | grep "#tree=512" | awk -F',' '{sum+=$12; ++n} END { print "Avg: "sum"/"n"="sum/n }'

# execute for every csv files
ls -1 $FOLDER | grep ".valid_test.csv" | cat | tail -n +2 -q | grep "#tree=512" | awk -F',' '{sum+=$12; ++n} END { print "Avg: "sum"/"n"="sum/n }'
awk -F',' '{sum+=$12; ++n} END { print "Avg: "sum"/"n"="sum/n }' *.valid_test.csv

# below two commands are same
find ./ -name "*.valid_test.csv" -exec sh -c 'cat $0| tail -n +2 -q | grep "#tree=512" | wc -l' {} \;
find ./ -name "*.valid_test.csv" -exec sh -c 'tail -n +2 -q $0| grep "#tree=512" | wc -l' {} \;

find ./ -name "*.valid_test.csv" -exec sh -c "cat $0| tail -n +2 -q | grep '#tree=512' | awk -F\',\' \'{sum+=$12; ++n} END { print \"Avg: \"sum\"/\"n\"=\"sum/n}\' " {} \;
find ./ -name "*.valid_test.csv" -exec sh -c "cat $0| tail -n +2 -q | grep '#tree=512' | awk -F',' '{sum+=$12; ++n} END { print sum/n}' " {} \;
find ./ -name "*.valid_test.csv" -exec sh -c 'cat $0| tail -n +2 -q | grep "#tree=512" | awk -F"," "{ print $1 }"   ' {} \;
for f in *.valid_test.csv; do echo "$f"; done
for f in $(find ./ -name "*.valid_test.csv"); do echo "$f"; done
for f in $(ls *.valid_test.csv); do echo "$f"; done

# this works
for f in *.valid_test.csv; do cat "$f"| tail -n +2 -q | grep "#tree=512" | awk -F"," '{ print $1 }'; done
# but this doesn't work, see the color of $1, it's not recoginized as a variable, why?
# Shell variables are extended only inside double quotes (or no quotes at all) , not single quotes.
#       From: https://stackoverflow.com/questions/28093236/use-bash-script-1-argument-in-awk-command
# Does this mean that $1 is rendered as the whole string piped from the last process? Thus it prints all the lines
for f in *.valid_test.csv; do cat "$f"| tail -n +2 -q | grep "#tree=512" | awk -F"," "{ print $1 }"; done

# Finally !!!
# Print like this: dstc2 0-allw/o similarity Random forest.#tree=512 0.677069
for f in *.valid_test.csv; do cat "$f"| tail -n +2 -q | grep "#tree=512" | awk -F"," -v OFS='\t\t' '{sum+=$12; ++n} END { print $1, $3, $4, sum/n}'; done
for f in *.valid_test.csv; do cat "$f"| tail -n +2 -q | grep "#tree=512" | awk -F"," '{acc+=$9;f1+=$12; ++n} END { printf ("%s\t\t%s\t\t%s\t\tacc=%.7f\t\tf1=%.7f\n", $1, $3, $4, acc/n, f1/n)}'; done

awk -F"," '{acc+=$2;++n} END { printf ("%s\t\t%s\t\t%s\t\tacc=%.7f\n", $1, $3, $4, acc/n)}' dstc2.y_test_corr.txt

#