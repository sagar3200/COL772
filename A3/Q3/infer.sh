LANG=$1
TEST_FILE=$2
OUTPUT_DIR=$3

if [ -z "$LANG" ] || [ -z "$TEST_FILE" ] || [ -z "$OUTPUT_DIR" ]; then
    echo "Usage: ./infer3.sh <en|hi|kn|or|tcy> <test_file_path> <output_dir>"
    exit 1
fi

if [[ "$LANG" != "en" && "$LANG" != "hi" && "$LANG" != "kn" && \
      "$LANG" != "or" && "$LANG" != "tcy" ]]; then
    echo "ERROR: lang must be one of en/hi/kn/or/tcy, got '$LANG'"
    exit 1
fi

if [ ! -f "$TEST_FILE" ]; then
    echo "ERROR: test file not found → $TEST_FILE"
    exit 1
fi

mkdir -p "$OUTPUT_DIR"

echo "============================================"
echo "  Task 3 ICL | lang=$LANG"
echo "  test=$TEST_FILE"
echo "  output=$OUTPUT_DIR"
echo "============================================"

GPU=$(nvidia-smi --query-gpu=index,memory.free \
      --format=csv,noheader,nounits 2>/dev/null \
      | sort -t',' -k2 -rn | head -1 | cut -d',' -f1 | tr -d ' ')

if [ -z "$GPU" ]; then
    GPU=0
fi
echo "  GPU: $GPU"

export CUDA_VISIBLE_DEVICES=$GPU

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1


python "$(dirname "$0")/infer.py" \
    --lang       "$LANG" \
    --test_file  "$TEST_FILE" \
    --output_dir "$OUTPUT_DIR"

OUTPUT_FILE="$OUTPUT_DIR/Q3_$LANG.jsonl"
if [ -f "$OUTPUT_FILE" ]; then
    echo ""
    echo "✓ Output saved → $OUTPUT_FILE"
    echo "  Lines: $(wc -l < "$OUTPUT_FILE")"
else
    echo "✗ ERROR: Output file not created"
    exit 1
fi