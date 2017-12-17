

from os import listdir
from os.path import isfile, isdir, join

bash_script = open("./trim.sh", 'w')

bash_script.write('#!/bin/bash\n\n')


people_path = "./voxceleb1_txt/"

people_folder = listdir(people_path)

for people in people_folder:
	recordings_path = people_path + people
	recordings_files = listdir(recordings_path)
	name = people.replace('.', '')
	for file in recordings_files:
		files_path = recordings_path + '/' + file
		mp3_file = file.split('.')[0]

		f = open(files_path, 'r')
		lines = f.readlines()
		lines = [x.strip() for x in lines]
		for i in range(5, len(lines)):
			temp_start_time = lines[i].split(' ')[1]
			temp_end_time = lines[i].split(' ')[2]
			bash_script.write('ffmpeg -i ' + mp3_file + '.mp3 -ss ' + temp_start_time + ' -to ' + temp_end_time + ' -c copy ./temp_output/' + mp3_file +str(i-4) + '.mp3\n')

	bash_script.write('mp3wrap ./output/' + name + '.mp3 ./temp_output/*.mp3\n')
	bash_script.write('rm -f ./temp_output/*.mp3\n')



