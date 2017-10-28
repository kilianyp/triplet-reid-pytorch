# calculate embeddings

MODEL=$1

if [ -z "$1" ]; then
    echo "No Model given!"
    exit
fi

DATA_DIR="~/Projects/triplet-reid-pytorch/datasets/Market-1501/"
TEST_CSV="~/Projects/cupsizes/data/market1501_test.csv"
QUERY_CSV="~/Projects/cupsizes/data/market1501_query.csv"

TEST_EMBD=$(python3 embed.py --csv_file $TEST_CSV --data_dir $DATA_DIR --model $MODEL 2>&1)
if [ $? -eq 0 ]; then
    echo "$TEST_EMBD was created successfully!"
else
    echo "$TEST_EMBD failed!"
    exit
fi

QUERY_EMBD=$(python3 embed.py --csv_file $QUERY_CSV --data_dir $DATA_DIR --model $MODEL 2>&1)

if [ $? -eq 0 ]; then
    echo "$QUERY_EMBD was created successfully!"
else
    echo "$QUERY_EMBD failed!"
    exit
fi

python3 ~/Projects/cupsizes/evaluate.py \
    --dataset market1501 \ 
    --query_dataset $QUERY_CSV \
    --query_embeddings $QUERY_EMBD \
    --gallery_dataset $TEST_CSV \
    --gallery_embeddings $TEST_EMBD \
    --metric euclidean \
    --batch_size 128
