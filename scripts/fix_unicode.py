"""
Quick fix for Unicode characters in preprocess_data.py
"""

import re
from pathlib import Path

# Read the file
file_path = Path("scripts/preprocess_data.py")
content = file_path.read_text(encoding='utf-8')

# Replace Unicode symbols
replacements = {
    '✓': '[OK]',
    '✗': '[FAIL]',
    '⚠': '[WARN]',
    '📊': '[DATA]',
    '💰': '[COST]',
    '☁️': '[CLOUD]',
    '🔧': '[SERVICE]',
    '📁': '[FILES]',
    '📝': '[LOG]',
    '✅': '[SUCCESS]',
}

for old, new in replacements.items():
    content = content.replace(old, new)

# Also fix the FileHandler to use UTF-8
old_file_handler = "file_handler = logging.FileHandler(log_file)"
new_file_handler = "file_handler = logging.FileHandler(log_file, encoding='utf-8')"
content = content.replace(old_file_handler, new_file_handler)

# Write back
file_path.write_text(content, encoding='utf-8')
print(f"[OK] Fixed {file_path}")
print(f"[OK] Replaced {len(replacements)} Unicode symbols")