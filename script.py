# script.py
x = 15
y = 20
result = x + y

print(result)

# Write the result to a file
with open("output.txt", "w") as file:
    file.write(str(result))
