import sys
import os

if len(sys.argv) != 3:
    print(f"Usage: {os.path.basename(sys.argv[0])} <input_file> <search_term>")
    sys.exit(1)

input_file = sys.argv[1]
search_term = sys.argv[2]
output_file = f"filtered_{search_term}_links.txt"

with open(input_file, "r", encoding="utf-8") as infile:
    lines = infile.readlines()

filtered = [line for line in lines if search_term in line]

with open(output_file, "w", encoding="utf-8") as outfile:
    outfile.writelines(filtered)

print(f"Filtered {len(filtered)} magnet links containing '{search_term}' to {output_file}")