import fileinput
import sys
import csv
import os

# global variables
dict_IPA = {}
flatten_IPA_classes = {}


def read_IPA_classes():
	reader = open('./IPA_phoneme_properties.csv', 'r', encoding='utf-8')
	for i, line in enumerate(reader):
		if i>8:
			line = line.strip('\n')
			items = line.split(',')
			value = items[0]
			key = items[1]
			flatten_IPA_classes[key] = value
	reader.close()
	return flatten_IPA_classes


def phonme_properties_mapping(flatten_IPA_classes, source, target):
	source_list = source.split()
	target_list = target.split()

	for item in source_list:
		if(item in flatten_IPA_classes):
			source_list[source_list.index(item)]=flatten_IPA_classes[item]

	for item in target_list:
		if(item in flatten_IPA_classes):
			target_list[target_list.index(item)]=flatten_IPA_classes[item]

	flatten_source="".join(source_list)
	flatten_target=" ".join(target_list) + "\n"

	return flatten_source + " , " + flatten_target


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

	new_filename="CELEX_IPA_flatten_data.txt"
	fw = open(new_filename,"w", encoding='utf-8') #opens the file
	fr = open(sys.argv[1], "r") # raw celex dataset

	dict_IPA = read_IPA()
	flatten_IPA_classes = read_IPA_classes()
	
	for line in fr.readlines(): #for every line in the file
		new_line=strip_line(line, dict_IPA) #run the function strip_line on the line, save it in new_line
		if new_line != "":
			phoneme_string=" ".join(new_line.split(" ")[1:]) #run the function strip_line on the line, save it in new_line
			phoneme_string_without_stress0 = phoneme_string.replace('ˈ', "")
			phoneme_string_without_stress0_stress1 = phoneme_string_without_stress0.replace('ˌ', "")
			phoneme_string_without_stress0_stress1_syllable = phoneme_string_without_stress0_stress1.replace('-', "")
			
			flatten_new_line = phonme_properties_mapping(flatten_IPA_classes, phoneme_string_without_stress0_stress1_syllable, phoneme_string_without_stress0_stress1)
			fw.write(flatten_new_line) # write the new_line in the file
	fr.close()
	fw.close()

# call the file with the celex raw dataset as argv[1]
if __name__ == '__main__':
	parse_file()