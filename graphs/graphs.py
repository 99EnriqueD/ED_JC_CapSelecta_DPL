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

def make_plot(dpl_data_file,baseline_data_file,label_base,label_dpl,x_axis_name,y_axis_name,output_file):
        baseline_data_file = rel_path + baseline_data_file
        dpl_data_file = rel_path + dpl_data_file
        iterations_base= []
        iterations_dpl= []
        to_plot_base= []
        to_plot_dpl= []
        with open(dpl_data_file) as f:
                for line in f.readlines():
                        splitted= line.split(":")
                        iterations_dpl.append(int(splitted[0]))
                        to_plot_dpl.append(float(splitted[1]))
        with open(baseline_data_file) as f:
                for line in f.readlines():
                        splitted= line.split(":")
                        iterations_base.append(int(splitted[0]))
                        to_plot_base.append(float(splitted[1]))
        
        #print(sum(to_plot_base)/len(to_plot_base),"avg base")
        #print(sum(to_plot_dpl)/len(to_plot_dpl),"avg dpl")
        plt.plot(iterations_base,to_plot_base, label=label_base)
        plt.plot(iterations_dpl,to_plot_dpl, label=label_dpl)
        plt.xlabel(x_axis_name)
        plt.ylabel(y_axis_name)
        plt.legend(loc='best')
        plt.title(output_file)
        plt.savefig(rel_path +output_file)
        plt.clf()
        return


def calc_diff(dpl_data_file,baseline_data_file):
        baseline_data_file = rel_path + baseline_data_file
        dpl_data_file = rel_path + dpl_data_file
        iterations_base= []
        iterations_dpl= []
        to_plot_base= []
        to_plot_dpl= []
        with open(dpl_data_file) as f:
                for line in f.readlines():
                        splitted= line.split(":")
                        iterations_dpl.append(int(splitted[0]))
                        to_plot_dpl.append(float(splitted[1]))
        with open(baseline_data_file) as f:
                for line in f.readlines():
                        splitted= line.split(":")
                        iterations_base.append(int(splitted[0]))
                        to_plot_base.append(float(splitted[1]))
        
        return iterations_base, (to_plot_dpl-to_plot_base)

def make_diff_plot(dpl_data_file1,baseline_data_file1,dpl_data_file2,base_data_file2):
        amount_iterations1, diff1= calc_diff(dpl_data_file1,baseline_data_file1)
        amount_iterations2, diff2= calc_diff(dpl_data_file2,baseline_data_file2)
        print(diff1)
        print(diff2)
        return

#make_plot("outfit_acc.txt","outfit_baseline_acc.txt","Baseline outfit classifier","DeepProbLog outfit classifier","iterations","accuracy","outfit accuracy")
#make_plot("outfit_F1.txt","outfit_baseline_F1.txt","Baseline outfit classifier","DeepProbLog outfit classifier","iterations","F1- score","outfit F1- score")
#make_plot("budget_acc.txt","budget_baseline_acc.txt","Baseline budget classifier","DeepProbLog budget classifier","iterations","accuracy","budget accuracy")
#make_plot("budget_F1.txt","budget_baseline_F1.txt","Baseline budget classifier","DeepProbLog budget classifier","iterations","F1- score","budget F1- score")
#make_plot("outfit_dist.txt","outfit_baseline_dist.txt","Baseline budget classifier","DeepProbLog budget classifier","iterations","distance","budget distance function")
#make_diff_plot("outfit_F1.txt","outfit_baseline_F1.txt","budget_F1.txt","budget_baseline_F1.txt")