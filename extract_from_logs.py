import os
import sys
import re
import pandas as pd


# Process each line and extract only the message, ignore the name and timestamp
def process_line(line: str) -> str:
    ret = re.sub(r'\t', '', line)
    ret = re.search(r"<[\s\S]+> (.*)", ret)
    if ret is not None:
        return ret.group(1)
    return None

# Extract the lines from the file and return them as a vector of strings
def extract_from_logs(filename: str) -> list[str]:
    preprocessed_lines: list[str] = []

    with open(filename, 'r', errors='ignore') as f:
        # Read line but skip it if it throws an excpetion
        for line in f:
            try:
                processed_line = process_line(line)
                if processed_line is None:
                    continue
                preprocessed_lines.append(processed_line)
            except Exception as e:
                print(e)
                continue
    return preprocessed_lines

# Create negative examples from the lines of the IC logs
def create_neg_files(preprocessed_lines: list[str], output_dir: str):
    for i, line in enumerate(preprocessed_lines):
        # open file and write the line and a 0 separated by tab
        with open(os.path.join(output_dir, 'data.csv'), 'a') as f:
            f.write(line + '\t0\n')

# Create prositive files by taking user input from the console and writing it to a file
def create_pos_files(output_dir: str):
    while True:
        line = input("Enter a line: ")
        if line == "":
            break
        with open(os.path.join(output_dir, 'data.csv'), 'a') as f:
            f.write(line + '\t0\n')

# start the main function

if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Description: Extracts lines of dialogue from the IRC logs which are from irssi")
        print("\tUsage: python extract_from_logs.py <input_file> <output_neg_dir> <output_pos_dir>")
        sys.exit(1)

    input_file = sys.argv[1]
    output_neg_dir = sys.argv[2]
    output_pos_dir = sys.argv[3]

    #preprocessed_lines = extract_from_logs(input_file)
    #create_neg_files(preprocessed_lines, output_neg_dir)

    create_pos_files(output_pos_dir)

    sys.exit(0)