import fileinput
import sys
import csv
import os

def read_IPA():
	reader = open('./IPA_phoneme_properties.csv', 'r', encoding='utf-8')
	dict_IPA = {}
	for i, line in enumerate(reader):
		if i>8:
			line = line.strip('\n')
			items = line.split(',')
			value = items[0]
			key = items[1]
			dict_IPA[key] = value
	reader.close()
	return dict_IPA

def phonme_properties_mapping(dict_IPA, source, target):
	source_list = source.split()
	target_list = target.split()

	for item in source_list:
		if(item in dict_IPA):
			source_list[source_list.index(item)]=dict_IPA[item]

	for item in target_list:
		if(item in dict_IPA):
			target_list[target_list.index(item)]=dict_IPA[item]

	flatten_source=" ".join(source_list) + "\n"
	flatten_target=" ".join(target_list) + "\n"

	return flatten_source, flatten_target


# reading the dictionary
def parse_file():

	# line = "(\"abascal\" nil (((ae) 1) ((b ax) 0) ((s k ax l) 0)))"
	# new_line=strip_line(line)
	# print(new_line)

	dict_IPA=read_IPA()
	
	#takes a filename.dict as argument, takes the part before the . and changes it to filename_stripped.dict
	new_filename1 = "flatten_IPA_source.txt"
	new_filename2 = "flatten_IPA_target.txt"

	fw1 = open(new_filename1,"w") #opens the source file
	fw2 = open(new_filename2,"w") #opens the target file
	fr = open(sys.argv[1], "r")
	for line in fr.readlines(): #for every line in the file
		phoneme_string=" ".join(line.split(" ")[1:]) #run the function strip_line on the line, save it in new_line
		phoneme_string_without_stress0 = phoneme_string.replace('ˈ', "")
		phoneme_string_without_stress0_stress1 = phoneme_string_without_stress0.replace('ˌ', "")
		phoneme_string_without_stress0_stress1_syllable = phoneme_string_without_stress0_stress1.replace('-', "")

		flatten_source, flatten_target = phonme_properties_mapping(dict_IPA, phoneme_string_without_stress0_stress1_syllable, phoneme_string_without_stress0_stress1)

		fw1.write(flatten_source) # write the new_line in the file
		fw2.write(flatten_target)
	fr.close()
	fw1.close()
	fw2.close()



def main():
	parse_file()
	


if __name__ == '__main__':
	main()