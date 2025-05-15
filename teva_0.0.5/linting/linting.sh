#!/bin/bash

input_file="$1"
output_file="$2"

echo $input_file
echo $output_file

if [ -z "$input_file" ]; then
  echo "Please provide an input file."
  exit 1
fi

# Run pylint on the input file and capture the exit code
if [ -n "$output_file" ]; then
  pylint "$input_file" --errors-only > "$output_file"
else
  pylint "$input_file" --errors-only
fi
pylint_exit_code=$?

#echo $pylint_output

# Check if errors were detected
if [[ $pylint_exit_code -eq 1 ]]; then
  echo "Pylint detected fatal errors."
  exit 1
elif [[ $pylint_exit_code -eq 2 ]]; then
  echo "Pylint detected errors."
  exit 1
fi

echo "Pylint detected no errors."

if [ -n "$output_file" ]; then
  pylint "$input_file" > "$output_file"
else
  pylint "$input_file"
fi

pylint_exit_code=$?
if [[ $pylint_exit_code -eq 0 ]]; then
  echo "Pylint detected no issues."
else
  echo "Pylint detected convention issues."
fi

exit 0

