__author__ = 'dan'
import os

def mylistdir(directory):
    """A specialized version of os.listdir() that ignores files that
    start with a leading period."""
    filelist = os.listdir(directory)
    return [x for x in filelist
            if not (x.startswith('.'))]

test_dir = '/Users/dan/Google Drive/kaggle/retinopathy/test_normalized'

test_files = mylistdir(test_dir)

print "Number of test files to process: ", str(len(test_files))

num = 1
fhandle = open('csv/test.csv', 'w')
for f in test_files:
    fhandle.write(f + "\n")

fhandle.close()
