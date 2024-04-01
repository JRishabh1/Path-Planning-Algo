# import pandas as pd
# import matplotlib.pyplot as plt
# import numpy as np

# def make_histograms_compilation(world, csvnames):
#     # Load the CSV file
#     for i in range(len(csvnames)):
#         data = pd.read_csv(csvnames[i] + ".csv", header=None)

#         values = data.iloc[0].values
        
#         plt.subplot(2, 5, i + 1)
#         plt.hist(values, bins=30, alpha=0.5, color='steelblue', edgecolor='black')
#         plt.title(csvnames[i])

#         plt.xlabel('Branch Length')
#         plt.ylabel('Frequency')

#         plt.subplot(2, 5, 2*i + 1)
#         plt.hist(values, bins=30, alpha=0.5, color='steelblue', edgecolor='black')
#         plt.title(csvnames[i])

#         plt.yscale('log')
#         plt.xlabel('Branch Length')
#         plt.ylabel('Frequency (log)')

#     plt.tight_layout()
#     plt.savefig(world + '_histograms.png', dpi=300, bbox_inches='tight')  

# def make_dot_plot_compilation(world, csvnames):
#     # Load the CSV file
#     for i in range(len(csvnames)):
#         data = pd.read_csv(csvnames[i] + ".csv", header=None)

#         values = data.iloc[0].values
        
#         plt.subplot(2, 5, i + 1)
#         plt.plot(values, marker='o', linestyle=' ')
#         plt.title(csvnames[i])
#         plt.xlabel('Iterations')
#         plt.ylabel('Length of Branch Added')

#         plt.subplot(2, 5, 2*i + 1)
#         plt.plot(values, marker='o', linestyle=' ')
#         plt.title(csvnames[i])
#         plt.xlabel('Iterations')
#         plt.ylabel('Length of Branch Added (log)')
#         plt.yscale('log')

#     plt.tight_layout()
#     plt.savefig(world + '_dotplot.png', dpi=300, bbox_inches='tight')  

# def make_cum_sum_plot_compilation(world, csvnames):
#     # Load the CSV file
#     for i in range(len(csvnames)):
#         data = pd.read_csv(csvnames[i] + ".csv", header=None)

#         values = data.iloc[0].values
        
#         prefix_sum = values.cumsum()
        
#         plt.subplot(2, 5, i + 1)
#         plt.plot(prefix_sum, marker='o', linestyle='-')
#         plt.title(csvnames[i])
#         plt.xlabel('Iterations')
#         plt.ylabel('Prefix Sum of Length of Branch Added')
#         slope = np.polyfit(range(len(prefix_sum)), prefix_sum, 1)[0]
#         plt.text(0.05, 0.95, f'Slope: {slope:.2f}', transform=plt.gca().transAxes, verticalalignment='top')
#         plt.subplot(2, 5, i + 1)
#         plt.plot(prefix_sum, marker='o', linestyle='-')
#         plt.title(csvnames[i])
#         plt.xlabel('Iterations')
#         plt.ylabel('Prefix Sum of Length of Branch Added (log)')
#         plt.yscale('log')
#         slope = np.polyfit(range(len(prefix_sum)), prefix_sum, 1)[0]
#         plt.text(0.05, 0.95, f'Slope: {slope:.2f}', transform=plt.gca().transAxes, verticalalignment='top')

#     plt.tight_layout()
#     plt.savefig(world + '_dotplot.png', dpi=300, bbox_inches='tight') 

# def make_cum_sum_plot_by_time_compilation(world, csvnames):
#     # Load the CSV file
#     for i in range(len(csvnames)):
#         data = pd.read_csv(csvnames[i] + ".csv", header=None)

#         values = data.iloc[0].values
#         time = data.iloc[1].values
        
#         prefix_sum = values.cumsum()
        
#         plt.subplot(2, 5, i + 1)
#         plt.plot(time, prefix_sum, marker='o', linestyle='-')
#         plt.title(csvnames[i])
#         plt.xlabel('Time')
#         plt.ylabel('Prefix Sum')
#         slope = np.polyfit(time, prefix_sum, 1)[0]
#         plt.text(0.05, 0.95, f'Slope: {slope:.2f}', transform=plt.gca().transAxes, verticalalignment='top')
#         plt.subplot(2, 5, 2*i + 1)
#         plt.plot(prefix_sum, marker='o', linestyle='-')
#         plt.title(csvnames[i])
#         plt.xlabel('Iterations')
#         plt.ylabel('Prefix Sum of Length of Branch Added (log)')
#         plt.yscale('log')
#         slope = np.polyfit(time, prefix_sum, 1)[0]
#         plt.text(0.05, 0.95, f'Slope: {slope:.2f}', transform=plt.gca().transAxes, verticalalignment='top')


#     plt.tight_layout()
#     plt.savefig(world + '_dotplot.png', dpi=300, bbox_inches='tight') 


# images_to_test = ["world1", "world2", "world3", "world4"]

# laplace_iters_to_test = [50, 100, 200, 300, 500]

# for i in range(len(images_to_test)):
#     image = images_to_test[i]
#     image_graph_names = []
#     for j in range(len(laplace_iters_to_test)):
#         len_per_iter = []
#         time_per_iter = []
#         node_list = []
#         result_images = []

#         la_place_each_time = laplace_iters_to_test[j]
#         output_path = str(image) + "_" + str(la_place_each_time) + "_each_iter"
#         image_graph_names.append(output_path)
#     plt.clf()
#     make_histograms_compilation(image, image_graph_names)
#     plt.clf()
#     make_dot_plot_compilation(image, image_graph_names)
#     plt.clf()
#     make_cum_sum_plot_compilation(image, image_graph_names)
#     plt.clf()
#     make_cum_sum_plot_by_time_compilation(image, image_graph_names)
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def make_histograms_compilation(world, csvnames):
    fig, axs = plt.subplots(2, len(csvnames), figsize=(20, 10))  # Adjust the figsize as necessary

    for i, csvname in enumerate(csvnames):
        data = pd.read_csv(csvname + ".csv", header=None)
        values = data.iloc[0].values

        # Normal Histogram
        axs[0, i].hist(values, bins=30, alpha=0.5, color='steelblue', edgecolor='black')
        axs[0, i].set_title(csvname)
        axs[0, i].set_xlabel('Branch Length')
        axs[0, i].set_ylabel('Frequency')

        # Histogram with Log Scale
        axs[1, i].hist(values, bins=30, alpha=0.5, color='steelblue', edgecolor='black')
        axs[1, i].set_yscale('log')
        axs[1, i].set_title(csvname + " (log scale)")
        axs[1, i].set_xlabel('Branch Length')
        axs[1, i].set_ylabel('Frequency (log)')

    plt.tight_layout()
    plt.savefig(world + '_histograms.png', dpi=300, bbox_inches='tight')
    plt.close(fig)  # Close the figure to free up memory

def make_dot_plot_compilation(world, csvnames):
    fig, axs = plt.subplots(2, len(csvnames), figsize=(20, 10))  # Adjust the figsize as necessary

    for i, csvname in enumerate(csvnames):
        data = pd.read_csv(csvname + ".csv", header=None)
        values = data.iloc[0].values

        # Normal Dot Plot
        axs[0, i].plot(values, marker='o', linestyle=' ')
        axs[0, i].set_title(csvname)
        axs[0, i].set_xlabel('Iterations')
        axs[0, i].set_ylabel('Length of Branch Added')

        # Dot Plot with Log Scale
        axs[1, i].plot(values, marker='o', linestyle=' ')
        axs[1, i].set_yscale('log')
        axs[1, i].set_title(csvname + " (log scale)")
        axs[1, i].set_xlabel('Iterations')
        axs[1, i].set_ylabel('Length of Branch Added (log)')

    plt.tight_layout()
    plt.savefig(world + '_dotplots.png', dpi=300, bbox_inches='tight')  # Use a different filename
    plt.close(fig)  # Close the figure to free up memory

def make_cum_sum_plot_compilation(world, csvnames):
    fig, axs = plt.subplots(2, len(csvnames), figsize=(20, 10))  # Adjust the figsize as necessary

    for i, csvname in enumerate(csvnames):
        data = pd.read_csv(csvname + ".csv", header=None)
        values = data.iloc[0].values

        prefix_sum = values.cumsum()
        slope = np.polyfit(range(len(prefix_sum)), prefix_sum, 1)[0]

        # Normal Cumulative Sum Plot
        axs[0, i].plot(prefix_sum, marker='o', linestyle='-')
        axs[0, i].set_title(csvname)
        axs[0, i].set_xlabel('Iterations')
        axs[0, i].set_ylabel('Prefix Sum of Length of Branch Added')
        axs[0, i].text(0.05, 0.95, f'Slope: {slope:.2f}', transform=axs[0, i].transAxes, verticalalignment='top')

        # Cumulative Sum Plot with Log Scale
        axs[1, i].plot(prefix_sum, marker='o', linestyle='-')
        axs[1, i].set_yscale('log')
        axs[1, i].set_title(csvname + " (log scale)")
        axs[1, i].set_xlabel('Iterations')
        axs[1, i].set_ylabel('Prefix Sum of Length of Branch Added (log)')
        axs[1, i].text(0.05, 0.95, f'Slope: {slope:.2f}', transform=axs[1, i].transAxes, verticalalignment='top')

    plt.tight_layout()
    plt.savefig(world + '_cumsum_plots.png', dpi=300, bbox_inches='tight')  # Use a different filename
    plt.close(fig)  # Close the figure to free up memory

def make_cum_sum_plot_by_time_compilation(world, csvnames):
    fig, axs = plt.subplots(2, len(csvnames), figsize=(20, 10))  # Adjust the figsize as necessary

    for i, csvname in enumerate(csvnames):
        data = pd.read_csv(csvname + ".csv", header=None)
        values = data.iloc[0].values
        time = data.iloc[1].values

        prefix_sum = values.cumsum()
        slope = np.polyfit(time, prefix_sum, 1)[0]

        # Normal Cumulative Sum Plot by Time
        axs[0, i].plot(time, prefix_sum, marker='o', linestyle='-')
        axs[0, i].set_title(csvname)
        axs[0, i].set_xlabel('Time')
        axs[0, i].set_ylabel('Prefix Sum')
        axs[0, i].text(0.05, 0.95, f'Slope: {slope:.2f}', transform=axs[0, i].transAxes, verticalalignment='top')

        # Cumulative Sum Plot by Time with Log Scale
        axs[1, i].plot(time, prefix_sum, marker='o', linestyle='-')
        axs[1, i].set_yscale('log')
        axs[1, i].set_title(csvname + " (log scale)")
        axs[1, i].set_xlabel('Time')
        axs[1, i].set_ylabel('Prefix Sum (log)')
        axs[1, i].text(0.05, 0.95, f'Slope: {slope:.2f}', transform=axs[1, i].transAxes, verticalalignment='top')

    plt.tight_layout()
    plt.savefig(world + '_cumsum_time_plots.png', dpi=300, bbox_inches='tight')  # Use a different filename
    plt.close(fig)  # Close the figure to free up memory


images_to_test = ["world1", "world2", "world3", "world4", "t_shape", "t_shape_other_way"]

laplace_iters_to_test = [50, 100, 200, 300, 500]

for i in range(len(images_to_test)):
    image = images_to_test[i]
    image_graph_names = []
    for j in range(len(laplace_iters_to_test)):
        len_per_iter = []
        time_per_iter = []
        node_list = []
        result_images = []

        la_place_each_time = laplace_iters_to_test[j]
        output_path = str(image) + "_" + str(la_place_each_time) + "_each_iter"
        image_graph_names.append(output_path)
    plt.clf()
    make_histograms_compilation(image, image_graph_names)
    plt.clf()
    make_dot_plot_compilation(image, image_graph_names)
    plt.clf()
    make_cum_sum_plot_compilation(image, image_graph_names)
    plt.clf()
    make_cum_sum_plot_by_time_compilation(image, image_graph_names)
    #200 points selected