if [ -d "../python/rsearch" ]; then
    if [ -f "../build/librsearch.so" ]; then
        cp -r ../python/rsearch/ ./rsearch_backend/
	cp ../build/librsearch.so ./rsearch_backend/
    fi
else 
    echo "Need make python library first."
fi
if [ ! -d "./rsearch_backend/image/" ]; then
    mkdir ./rsearch_backend/image/
fi
if [ ! -d "./rsearch_backend/dataset/" ]; then
    mkdir ./rsearch_backend/dataset
fi
