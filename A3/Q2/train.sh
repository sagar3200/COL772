OUTPUT_DIR=${1:-"output"}
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [[ "$OUTPUT_DIR" != /* ]]; then
    OUTPUT_DIR="$SCRIPT_DIR/$OUTPUT_DIR"
fi

echo "Task 2 Training | output: $OUTPUT_DIR"
export CUDA_VISIBLE_DEVICES=0
python "$SCRIPT_DIR/train.py" "$OUTPUT_DIR"