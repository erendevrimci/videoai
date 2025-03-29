#!/bin/bash
# Run export tests and generate test files for verification

set -e  # Exit on error

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
TEST_OUTPUT_DIR="$SCRIPT_DIR/test_output"

echo "Running export tests from $SCRIPT_DIR"
echo "Project directory: $PROJECT_DIR"
echo "Test output directory: $TEST_OUTPUT_DIR"

# Create test output directory if it doesn't exist
mkdir -p "$TEST_OUTPUT_DIR"

# Run the export test suite
echo "Running test_export_with_target_apps.py..."
python "$SCRIPT_DIR/test_export_with_target_apps.py"

# Generate test timelines for all formats
echo "Generating test timelines for all formats..."
python "$SCRIPT_DIR/generate_test_timelines.py"

# Create a summary report
echo "Creating summary report..."
cat > "$TEST_OUTPUT_DIR/export_test_summary.txt" << EOL
Export Test Summary
==================

Tests run on: $(date)

Test Files Generated:
$(find "$TEST_OUTPUT_DIR" -type f | sort | sed 's/^/- /')

Next Steps:
1. Import the exported files into their target applications
2. Verify timeline structure, clips, effects, and metadata
3. Document any compatibility issues in docs/export_format_compatibility.md

For detailed verification procedures, see:
docs/export_verification_procedures.md
EOL

echo "Summary report created at $TEST_OUTPUT_DIR/export_test_summary.txt"
echo "Done running export tests!"