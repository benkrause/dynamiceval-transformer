echo "=== Acquiring datasets ==="
echo "---"

mkdir -p data
cd data



echo "- Downloading WikiText-103 (WT2)"
if [[ ! -d 'wikitext-103' ]]; then
    wget --continue https://s3.amazonaws.com/research.metamind.io/wikitext/wikitext-103-v1.zip
    unzip -q wikitext-103-v1.zip
    cd wikitext-103
    mv wiki.train.tokens train.txt
    mv wiki.valid.tokens valid.txt
    mv wiki.test.tokens test.txt
    cd ..
fi

echo "- Downloading enwik8 (Character)"
if [[ ! -d 'enwik8' ]]; then
    mkdir -p enwik8
    cd enwik8
    wget --continue http://mattmahoney.net/dc/enwik8.zip
    wget https://raw.githubusercontent.com/salesforce/awd-lstm-lm/master/data/enwik8/prep_enwik8.py
    python3 prep_enwik8.py
    cd ..
fi

echo "- Downloading text8 (Character)"
if [[ ! -d 'text8' ]]; then
    mkdir -p text8
    cd text8
    wget --continue http://mattmahoney.net/dc/text8.zip
    python ../../prep_text8.py
    cd ..
fi




echo "---"
echo "Happy language modeling :)"
