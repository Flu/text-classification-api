import json
import re

jokes = []
with open('data/jokes.json') as f:
    jokes = json.load(f)

# Write the body of every joke to the jokes csv file
with open('data/jokes.csv', 'w') as f:
    for joke in jokes:
        body = re.sub(r'[\t\r\n]', '', joke['body'])
        f.write(body + '\t0\n')
