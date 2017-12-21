from os import listdir
from os.path import isfile, isdir, join

jeff = open("./candidate_name.txt", 'w')




name = listdir('./')

for i in name:
	jeff.write(i[:-12] +'\n')

jeff.close()