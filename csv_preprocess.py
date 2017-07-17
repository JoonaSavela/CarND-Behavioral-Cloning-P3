import csv

input = open('./data/driving_log.csv', 'rb')
output = open('./data/driving_log2.csv', 'wb')
reader = csv.reader(input)
writer = csv.writer(output)
next(reader, None)

for line in reader:
    if any(field.strip() for field in row):
        writer.writerow(row)

input.close()
output.close()

