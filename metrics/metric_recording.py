import matplotlib.pyplot as plt
import numpy as np

rel_path = "metrics/"

def clear_file(file_name) :
        try :
                if file_name[-4:] == ".txt":
                        with open(rel_path + file_name,"r+") as f:
                                f.truncate()
                else :
                        print("Illegal file to clear!")
        except IOError:
                print("No file to clear.")
        finally:
                return

def save_data(iteration,data,file_name):
        with open(rel_path + file_name,'a') as f:
                f.write(str(iteration) + ":" + str(data) + "\n")
        return

def save_cm(cm,file_name):
        with open(rel_path + file_name,'w+') as f:
                f.write(str(cm))
        return

