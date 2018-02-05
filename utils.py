"""
Functions used in the main program
"""

def read_txt_file(file_path):
    """
    read the contents of a text file line by line into a list of lines
    each line is also a list of words
    file_path: the location of the text file
    returns: the lines list
    """
    with open(file_path, encoding='utf8') as file:
        text = file.readlines()
        lines_list = [line.strip() for line in text]
    return lines_list


def add_space(s, total_len):
    """ 
    Add spaces to the end of a string
    """
    while len(s) < total_len:
        s+= ' '
    return s