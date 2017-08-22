import os

for num in range(10, 100):
	cmdString = 'mv turing' + str('%d'%num)+ '.tfrecords turing'+ str('%.3d'%num) +'.tfrecords'
	print cmdString
