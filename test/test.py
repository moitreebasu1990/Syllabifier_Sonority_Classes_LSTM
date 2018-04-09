import fileinput
import sys
import csv
import os
import numpy as np
import requests as req

def read_test_grapheme():
	fr = open("test_source.text","r", encoding='utf-8') #opens the input file
	fw = open("test_target.text","w", encoding='utf-8') #opens the output file
	for line in fr.readlines(): #for every line in the file
		new_line=run_test_g2p(line.strip("\n")) #run the function strip_line on the line, save it in new_line
		fw.write(new_line + "\n") # write the new_line in the file
	fr.close()
	fw.close()

def run_test_g2p(grapheme):
	response = req.get("http://localhost:5000/decode/" + grapheme)
	return (response.content.decode("utf-8").strip("\n")).strip('\"')

if __name__ == "__main__":
	read_test_grapheme()