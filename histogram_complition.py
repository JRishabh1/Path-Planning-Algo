import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def make_histograms_compilation(world, csvnames):
    # Load the CSV file
    for i in range(len(csvnames)):
        data = pd.read_csv(csvnames[i] + ".csv", header=None)

        values = data.iloc[0].values
        
        plt.subplot(4, 2, i + 1)
        plt.hist(values, bins=30, alpha=0.5, color='steelblue', edgecolor='black')
        plt.title(csvnames[i])

        plt.xlabel('Branch Length')
        plt.ylabel('Frequency')

        plt.subplot(4, 2, 2*i + 1)
        plt.hist(values, bins=30, alpha=0.5, color='steelblue', edgecolor='black')
        plt.title(csvnames[i])

        plt.yscale('log')
        plt.xlabel('Branch Length')
        plt.ylabel('Frequency (log)')

    plt.tight_layout()
    plt.savefig(world + '_histograms.png', dpi=300, bbox_inches='tight')  

def make_dot_plot_compilation(world, csvnames):
    # Load the CSV file
    for i in range(len(csvnames)):
        data = pd.read_csv(csvnames[i] + ".csv", header=None)

        values = data.iloc[0].values
        
        plt.subplot(4, 2, i + 1)
        plt.plot(values, marker='o', linestyle=' ')
        plt.title(csvnames[i])
        plt.xlabel('Iterations')
        plt.ylabel('Length of Branch Added')

        plt.subplot(4, 2, 2*i + 1)
        plt.plot(values, marker='o', linestyle=' ')
        plt.title(csvnames[i])
        plt.xlabel('Iterations')
        plt.ylabel('Length of Branch Added (log)')
        plt.yscale('log')

    plt.tight_layout()
    plt.savefig(world + '_dotplot.png', dpi=300, bbox_inches='tight')  

def make_cum_sum_plot_compilation(world, csvnames):
    # Load the CSV file
    for i in range(len(csvnames)):
        data = pd.read_csv(csvnames[i] + ".csv", header=None)

        values = data.iloc[0].values
        
        prefix_sum = values.cumsum()
        
        plt.subplot(4, 2, i + 1)
        plt.plot(prefix_sum, marker='o', linestyle='-')
        plt.title(csvnames[i])
        plt.xlabel('Iterations')
        plt.ylabel('Prefix Sum of Length of Branch Added')
        slope = np.polyfit(range(len(prefix_sum)), prefix_sum, 1)[0]
        plt.text(0.05, 0.95, f'Slope: {slope:.2f}', transform=plt.gca().transAxes, verticalalignment='top')
        plt.subplot(4, 2, i + 1)
        plt.plot(prefix_sum, marker='o', linestyle='-')
        plt.title(csvnames[i])
        plt.xlabel('Iterations')
        plt.ylabel('Prefix Sum of Length of Branch Added (log)')
        plt.yscale('log')
        slope = np.polyfit(range(len(prefix_sum)), prefix_sum, 1)[0]
        plt.text(0.05, 0.95, f'Slope: {slope:.2f}', transform=plt.gca().transAxes, verticalalignment='top')

    plt.tight_layout()
    plt.savefig(world + '_dotplot.png', dpi=300, bbox_inches='tight') 

def make_cum_sum_plot_by_time_compilation(world, csvnames):
    # Load the CSV file
    for i in range(len(csvnames)):
        data = pd.read_csv(csvnames[i] + ".csv", header=None)

        values = data.iloc[0].values
        time = data.iloc[1].values
        
        prefix_sum = values.cumsum()
        
        plt.subplot(4, 2, i + 1)
        plt.plot(time, prefix_sum, marker='o', linestyle='-')
        plt.title(csvnames[i])
        plt.xlabel('Time')
        plt.ylabel('Prefix Sum')
        slope = np.polyfit(time, prefix_sum, 1)[0]
        plt.text(0.05, 0.95, f'Slope: {slope:.2f}', transform=plt.gca().transAxes, verticalalignment='top')
        plt.subplot(4, 2, 2*i + 1)
        plt.plot(prefix_sum, marker='o', linestyle='-')
        plt.title(csvnames[i])
        plt.xlabel('Iterations')
        plt.ylabel('Prefix Sum of Length of Branch Added (log)')
        plt.yscale('log')
        slope = np.polyfit(time, prefix_sum, 1)[0]
        plt.text(0.05, 0.95, f'Slope: {slope:.2f}', transform=plt.gca().transAxes, verticalalignment='top')


    plt.tight_layout()
    plt.savefig(world + '_dotplot.png', dpi=300, bbox_inches='tight') 


images_to_test = ["world1", "world2", "world3", "world4"]

laplace_at_start_to_test = [50, 500]
la_place_each_time_to_test = [100, 200, 300, 500]

for i in range(len(images_to_test)):
    image = images_to_test[i]
    image_graph_names = []
    for j in range(len(laplace_at_start_to_test)):
        la_place_at_start = laplace_at_start_to_test[j]
        for k in range(len(la_place_each_time_to_test)):
            len_per_iter = []
            time_per_iter = []
            node_list = []
            result_images = []

            la_place_each_time = la_place_each_time_to_test[k]
            output_path = str(image) + "_" + str(la_place_at_start) + "_start_" + str(la_place_each_time) + "_each_iter"
            image_graph_names.append(output_path)
    make_histograms_compilation(image, image_graph_names)
