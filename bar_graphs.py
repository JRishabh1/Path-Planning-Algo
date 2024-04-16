# import pandas as pd
# import matplotlib.pyplot as plt
# import numpy as np

# def make_bar_graphs(world, csvnames):
#     fig, axs = plt.subplots(3, int(len(csvnames)/3), figsize=(20, 10))  # Adjust the figsize as necessary

#     for i, csvname in enumerate(csvnames):
#         data = pd.read_csv(csvname + ".csv", header=None)
#         values = data.iloc[0].values
#         time = data.iloc[1].values

#         prefix_sum = values.cumsum()
#         slope = np.polyfit(time, prefix_sum, 1)[0]

#         # Normal Cumulative Sum Plot by Time
#         axs[i % 3, int(i / 3)].plot(time, prefix_sum, marker='o', linestyle='-')
#         axs[i % 3, int(i / 3)].set_title(csvname)
#         axs[i % 3, int(i / 3)].set_xlabel('Time')
#         axs[i % 3, int(i / 3)].set_ylabel('Prefix Sum')
#         axs[i % 3, int(i / 3)].text(0.05, 0.95, f'Slope: {slope:.2f}', transform=axs[i % 3, int(i / 3)].transAxes, verticalalignment='top')
#         axs[i % 3, int(i / 3)].set_xlim(0, 700)
#         axs[i % 3, int(i / 3)].set_ylim(0, 3500)

#     plt.tight_layout()
#     plt.savefig(world + '_time_bar_graphs.png', dpi=300, bbox_inches='tight')  # Use a different filename
#     plt.close(fig)  # Close the figure to free up memory


# images_to_test = ["world1", "world2", "world3", "world4", "t_shape", "t_shape_other_way"]

# laplace_iters_to_test = [50, 100, 200, 300, 500]
# branches_before_each_laplace = [1, 5, 10]

# for i in range(len(images_to_test)):
#     image = images_to_test[i]
#     image_graph_names = []
#     for j in range(len(laplace_iters_to_test)):
#         la_place_each_time = laplace_iters_to_test[j]
#         for k in range(len(branches_before_each_laplace)):
#             branch_each_time = branches_before_each_laplace[k]
#             output_path = str(image) + "_" + str(la_place_each_time) + "laplace_" + str(branch_each_time) + "branches_each_iter"
#             image_graph_names.append(output_path)
#     plt.clf()
#     make_bar_graphs(image, image_graph_names)
import pandas as pd
import matplotlib.pyplot as plt

# Load data from CSV
df = pd.read_csv('compile_times_without_drawing.csv')  # Adjust the filename as needed

# Set 'Test Name' as the row labels and transpose the DataFrame for easier plotting
df.set_index('Test Name', inplace=True)
# df = df.transpose()

# Print column names to check their format
print("Column names in DataFrame:")
print(df.columns.tolist())  # This will show you all the column names

def plot_world_times(df, world_prefix):
    # Adjust regex pattern based on the actual column names printed above
    pattern = f'{world_prefix}_\d+laplace_\d+branches_each_iter'
    world_data = df.filter(regex=pattern)

    if world_data.empty:
        print(f"No data found for {world_prefix}. Check regex and data.")
        return

    num_plots = world_data.shape[1]
    if num_plots == 0:
        print(f"No matching columns for {world_prefix}.")
        return

    num_cols = 3
    num_rows = (num_plots + num_cols - 1) // num_cols

    fig, axes = plt.subplots(num_rows, num_cols, figsize=(15, 5 * num_rows), sharey=True)
    axes = axes.flatten()

    for i, (column_name, data) in enumerate(world_data.items()):
        ax = axes[i]
        data.plot(kind='bar', ax=ax, title=column_name)
        ax.set_xticklabels(data.index, rotation=45, ha='right')
        ax.set_ylabel('Time (seconds)')

    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    plt.tight_layout()
    plt.savefig(world + '_time_bar_graphs.png', dpi=300, bbox_inches='tight')
    # plt.show()

# Example usage
worlds = ['world1', 'world2', 'world3', 'world4', 't_shape', 't_shape_other_way']
for world in worlds:
    plot_world_times(df, world)
