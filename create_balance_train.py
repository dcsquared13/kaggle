__author__ = 'dan'

import csv

infile = 'csv/trainLabels.csv'
outfile = 'csv/trainLabels_4.csv'

i = 0
with open(infile,'r') as fin:
    with open (outfile,'w') as fout:
        writer = csv.writer(fout, delimiter=',')
        for row in csv.reader(fin, delimiter=','):
            if row[1] == '4':
                i += 1
                writer.writerow(row)
        writer.writerow("Total," + str(i))
fout.close()
fin.close()