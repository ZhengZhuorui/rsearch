#! /bin/sh
RSEARCH_DIR="/home/${USER}/rsearch"
if [ ! -d $RSEARCH_DIR ]; then
	mkdir $RSEARCH_DIR
fi
RSEARCH_DATA_DIR="$RSEARCH_DIR/data"
if [ ! -d $RSEARCH_DATA_DIR ]; then
	mkdir $RSEARCH_DATA_DIR
fi
