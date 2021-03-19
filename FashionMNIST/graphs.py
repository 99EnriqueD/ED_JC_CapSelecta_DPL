import matplotlib.pyplot as plt

def save_data(iteration,data,file_name):
        with open(file_name) as f:
                f.write(iteration:data)
        return

def make_plot(dpl_data_file,baseline_data_file,label_base,label_dpl,output_file):
        iterations_base= []
        iterations_dpl= []
        to_plot_base= []
        to_plot_dpl= []
        with open(dpl_data_file) as f:
                line= f.readlines()
                splitted= line.split(":")
                iterations_dpl.append(int(splitted[0]))
                to_plot_dpl.append(float(splitted[1]))
        with open(base_data_file) as f:
                line= f.readlines()
                splitted= line.split(":")
                iterations_base.append(int(splitted[0]))
                to_plot_base.append(float(splitted[1]))
        
        plt.plot(iterations_base,to_plot_base, label=label_base)
        plt.plot(iterations_base,to_plot_base, label=label_dpl)
        plt.legend(loc='best')
        plt.savefig(output_file)
        return