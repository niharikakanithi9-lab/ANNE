#!/usr/bin/env bash
set -e
mkdir -p data
cd data
echo "Downloading SIFT1M (~500MB)..."
wget -c ftp://ftp.irisa.fr/local/texmex/corpus/sift.tar.gz
tar -xzf sift.tar.gz
echo "Done. Files in data/sift/"