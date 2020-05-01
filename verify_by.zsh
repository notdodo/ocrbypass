#!/usr/bin/env zsh

local INPUT_IMAGE="$(basename ${1})"

local OUTPUT_TEXT="$(tesseract -c tessedit_char_whitelist=abcdefghijklmnopqrstuvwxyz1234567890 ${1} stdout 2>/dev/null)"
OUTPUT_TEXT="${OUTPUT_TEXT//[[:space:]]/}"

if [[ "${OUTPUT_TEXT}" = "${INPUT_IMAGE}" ]]; then
  echo "Bingo! ${OUTPUT_TEXT}"
else
  echo "Wrong! ${OUTPUT_TEXT}"
fi
