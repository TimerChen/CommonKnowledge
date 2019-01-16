#!/bin/bash
mkdir pics valpics
mkdir minipics minivalpics
sudo apt install ttf-dejavu
wget http://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz
tar -xzf aclImdb_v1.tar.gz
