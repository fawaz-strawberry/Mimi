# validate.py
# Read the result from the file
with open("output.txt", "r") as file:
    result = int(file.read().strip())

# Validate the result
expected_result = 35
assert result == expected_result, f"Validation failed: expected {expected_result}, got {result}"
