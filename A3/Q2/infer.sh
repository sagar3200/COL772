LANG=$1
TEST_FILE=$2
OUTPUT_DIR=$3
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [[ "$TEST_FILE" != /* ]]; then
    TEST_FILE="$SCRIPT_DIR/$TEST_FILE"
fi
if [[ "$OUTPUT_DIR" != /* ]]; then
    OUTPUT_DIR="$SCRIPT_DIR/$OUTPUT_DIR"
fi

echo "Task 2 Inference | lang=$LANG | test=$TEST_FILE | out=$OUTPUT_DIR"
export CUDA_VISIBLE_DEVICES=0
python "$SCRIPT_DIR/infer.py" "$LANG" "$TEST_FILE" "$OUTPUT_DIR"