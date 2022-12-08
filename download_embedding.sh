


# For references please visit https://wikipedia2vec.github.io/wikipedia2vec/pretrained/

wget -P /tmp http://wikipedia2vec.s3.amazonaws.com/models/en/2018-04-20/"$embed_file_name".bz2 >> /tmp/download_stdout.txt

bzip2 -d -q /tmp/"$embed_file_name".bz2
