__author__ = 'dan'

import random

infile = 'csv/trainLabelsBal.csv'
outfile = 'csv/trainLabelsBal1.csv'

with open(infile,'r') as source:
    data = [ (random.random(), line) for line in source ]
for i in range(10):
    data.sort()
with open(outfile,'w') as target:
    for _, line in data:
        target.write( line )