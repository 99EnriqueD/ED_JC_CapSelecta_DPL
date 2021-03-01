
path_training_attributes = "DeepFashion/anno_fine/train_attr.txt"
# data = open(path_training_attributes, "r")

# data.readline()[0:-1]

def get_pattern(line):
    patterns = ['floral','graphic','striped','embroiderd','pleated','solid','lattice']
    code = line[0:14]
    for cindex in range(0,len(code)) :
        if code[cindex] == '1' :
            i = cindex // 2
            return patterns[i]

with open(path_training_attributes) as f:
   for line in f:
       print(line)
       print(line[0:7])
       print(get_pattern(line))
       break

# def get_sleeve(line):

# def get_length(line) :

# def get_neck(line):

# def get_fabric(line):

# def get_fit(line):
