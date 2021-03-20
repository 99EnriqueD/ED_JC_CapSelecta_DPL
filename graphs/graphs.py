import matplotlib.pyplot as plt

rel_path = "graphs/"

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

def make_plot(dpl_data_file,baseline_data_file,label_base,label_dpl,output_file):
        baseline_data_file = rel_path + baseline_data_file
        dpl_data_file = rel_path + baseline_data_file
        iterations_base= []
        iterations_dpl= []
        to_plot_base= []
        to_plot_dpl= []
        with open(dpl_data_file) as f:
                line= f.readlines()
                splitted= line.split(":")
                iterations_dpl.append(int(splitted[0]))
                to_plot_dpl.append(float(splitted[1]))
        with open(baseline_data_file) as f:
                line= f.readlines()
                splitted= line.split(":")
                iterations_base.append(int(splitted[0]))
                to_plot_base.append(float(splitted[1]))
        
        plt.plot(iterations_base,to_plot_base, label=label_base)
        plt.plot(iterations_base,to_plot_base, label=label_dpl)
        plt.legend(loc='best')
        plt.savefig(output_file)
        return