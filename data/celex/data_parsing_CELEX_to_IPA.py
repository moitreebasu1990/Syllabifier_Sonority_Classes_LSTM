import fileinput
import sys
import csv
import os

#processing line
def strip_line(line, dict_IPA):
	new_line = ""
	new_line_split_list=line.split("\\", 10)
	# new_line=new_line_split_list[1] + " " + new_line_split_list[6] + "\n"
	if not (" " in new_line_split_list[1]):
		final_list = list(new_line_split_list[6])
		IPA_list = []
		for item in final_list:
			if item in dict_IPA:
				IPA_list.append(dict_IPA[item])
			else:
				IPA_list.append(item)
		
		final_phoneme_string = ' '.join(IPA_list)
		new_line = new_line_split_list[1].lower() + " " + final_phoneme_string + "\n"
	return new_line

def read_IPA():
	reader = open('./CELEX_IPA_transition-en.csv', 'r', encoding='utf-8')
	dict_IPA = {}
	for line in reader.readlines():
		line = line.strip('\n')
		items = line.split(',')
		k = items[0]
		v = items[1]
		dict_IPA[k] = v
	reader.close()
	return dict_IPA

# reading the dictionary
def parse_file():
	
	#takes a dictionary filename as argument for parsing

	new_filename="CELEX_IPA_data.txt"
	fw = open(new_filename,"w", encoding='utf-8') #opens the file
	fr = open(sys.argv[1], "r")

	dict_IPA = read_IPA()
	for line in fr.readlines(): #for every line in the file
		new_line=strip_line(line, dict_IPA) #run the function strip_line on the line, save it in new_line
		fw.write(new_line) # write the new_line in the file
	fr.close()
	fw.close()



def main():
	parse_file()
	


if __name__ == '__main__':
	main()