LANG=$1
TEST_FILE=$2
OUTPUT_DIR=$3
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
echo "Task 1 Inference | lang=$LANG | test=$TEST_FILE | out=$OUTPUT_DIR"
export CUDA_VISIBLE_DEVICES=0
python infer.py "$LANG" "$TEST_FILE" "$OUTPUT_DIR"