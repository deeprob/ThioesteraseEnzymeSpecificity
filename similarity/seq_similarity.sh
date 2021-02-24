file1=$1
file2=$2

blastp -query $file1 -subject $file2 -outfmt '6 pident'