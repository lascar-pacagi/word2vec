# Accompanying code for word embedding video

## Video

[Word embedding]()

## Installation

```bash
git clone https://github.com/lascar-pacagi/word2vec.git
cd word2vec
mkdir build
cd build
cmake ../source
make
```

You need a `c++` compiler, I used `g++-10` and [cmake](https://cmake.org/).

## Example

To obtain a word embedding, you first need a big text file with words separated by spaces. For example, you can get one text file
using

```bash
cd ..
mkdir data
wget http://mattmahoney.net/dc/text8.zip -O text8.gz
gzip -d text8.gz -f
```

then to learn an embedding from this file, you can do

```bash
cd ..
cd build
./word2vec --train text8 --stop --output ../data/embeddings.bin
```

to test the embedding you do

```bash
./distance --embeddings ../data/embeddings.bin
```

you can also use `word2vec2` instead of `word2vec` (difference in parallelization).

You can also get a bigger text file [here](http://mattmahoney.net/dc/enwik9.zip). To get the text file from this file you do

```bash
cd ..
cd data
wget http://mattmahoney.net/dc/enwik9.zip
unzip enwik9.zip
perl source/wikifil.pl enwik9 > enwik9.txt
```

## Reference

[Paper](https://arxiv.org/pdf/1301.3781.pdf)

[Implementation from the authors](https://code.google.com/archive/p/word2vec/)