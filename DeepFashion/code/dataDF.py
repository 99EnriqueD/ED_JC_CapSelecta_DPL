
path_training_attributes = "../anno_fine/val_attr.txt"
path_custom = "../custom/val/"
# data = open(path_training_attributes, "r")

# data.readline()[0:-1]

def get_pattern(line):
    patterns = ['floral','graphic','striped','embroiderd','pleated','solid','lattice']
    code = line[0:14]
    for cindex in range(0,len(code)) :
        if code[cindex] == '1' :
            i = cindex // 2
            return patterns[i]


def get_sleeve(line):
    patterns = ['long_sleeve', 'short_sleeve','sleeveless']
    code = line[14:20]
    for cindex in range(0,len(code)) :
        if code[cindex] == '1' :
            i = (cindex // 2)
            return patterns[i]

def get_dlength(line):
    patterns = ['maxi_length', 'mini_length','no_dress']
    code = line[20:26]
    for cindex in range(0,len(code)) :
        if code[cindex] == '1' :
            i = (cindex // 2)
            return patterns[i]

def get_neck(line):
    patterns = ['crew_neckline', 'v_neckline','square_neckline','no_neckline']
    code = line[26:34]
    for cindex in range(0,len(code)) :
        if code[cindex] == '1' :
            i = (cindex // 2)
            return patterns[i]

def get_fabric(line):
    patterns = ['denim', 'chiffon','cotton','leather', 'faux', 'knit']
    code = line[34:46]
    for cindex in range(0,len(code)) :
        if code[cindex] == '1' :
            i = (cindex // 2)
            return patterns[i]

def get_fit(line):
    patterns = ['tight', 'loose','conventional']
    code = line[46:52]
    for cindex in range(0,len(code)) :
        if code[cindex] == '1' :
            i = (cindex // 2)
            return patterns[i]

def generateOutput(file_name, element):
    f = open(file_name, 'w')
    for elem in elements:
        f.write(elem + "\n")  
    f.close()

with open(path_training_attributes) as f:
    fp = open(path_custom + "patterns.txt", 'w')
    fs = open(path_custom + "sleeves.txt", 'w')
    fl = open(path_custom + "length.txt", 'w')
    fn = open(path_custom + "neck.txt", 'w')
    ffa = open(path_custom + "fabric.txt", 'w')
    ffi = open(path_custom + "fit.txt", 'w')
    for line in f:
       fp.write(get_pattern(line) + "\n")
       fs.write(get_sleeve(line) + "\n")
       fl.write(get_dlength(line) + "\n")
       fn.write(get_neck(line)+"\n")
       ffa.write(get_fabric(line)+ "\n")
       ffi.write(get_fit(line)+ "\n")
    fp.close()
    fs.close()
    fl.close()
    fn.close()
    ffa.close()
    ffi.close()

