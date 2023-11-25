#!/bin/bash

# We have to print out the result at the end along, we store the result to use in a sentence
# An important thing to consider is that here, we have to first sort numerically then to reverse the order but in the case of the cities, we have to do it all at once, this may be because of the way the numbers are encoded for these two columns
# 
result_country=$(awk -F$'\t' '$8 == "MSc" {count[$11]++} END {for (c in count) {print c, count[c]}}' merged_courses.tsv | sort -k2,2n -r | head -1)
echo "The country with the most Master's Degrees is $result_country"


result_city=$(awk -F$'\t' '$8 == "MSc" {count[$10]++} END {for (c in count) {print c, count[c]}}' merged_courses.tsv | sort -k2,2nr | head -1)
echo "The city with the most Master's Degrees is $result_city"

# Here, we only have to comb through one column so there is no loop, we can simply print out the answer inside the awk function
awk -F$'\t' '$4 == "Part time" {count++} ; END {print "The number of colleges offering Part-Time Education is:" count}'  merged_courses.tsv 

awk -F$'\t' '$1 ~ "Engineer" {count++} ; END {print "The percentage of Engineering courses is: " count / (NR - 1)}'  merged_courses.tsv 
