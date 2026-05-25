
OUTPUT_DIR=${1:-"output"}
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "Task 1 Training | output: $OUTPUT_DIR"
python "$SCRIPT_DIR/train.py" "$OUTPUT_DIR"