
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

object_type_labels = [
    'vehicle', 'pedestrian', 'motorcyclist', 'cyclist', 'bus',
    'static', 'background', 'construction', 'riderless_bicycle', 'unknown'
]

# for 1b
def plot_feature_correlations(train_data):
    """
    train_data: shape (10000, 50, 110, 6)
    """

    flat_data = train_data.reshape(-1, 6)
    df = pd.DataFrame(flat_data, columns=['position_x', 'position_y', 'velocity_x', 'velocity_y', 'heading','object_type'])
    corr_matrix = df.corr()

    sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap='coolwarm', vmin=-1, vmax=1)
    plt.title("Feature Correlation Heatmap")
    plt.savefig("feat_corr.png", dpi=200)
    plt.show()

# for 1b
def plot_all_agent_distributions_split_input_output(train_data):
    """
    train_data: shape (10000, 50, 110, 6)
    """
    px_in = train_data[:, :, :50, 0].flatten()
    py_in = train_data[:, :, :50, 1].flatten()
    px_out = train_data[:, :, :50, 0]
    py_out = train_data[:, :, :50, 1]

    fig, axs = plt.subplots(1, 2, figsize=(16, 6))

    # Input positions heatmap
    hist_in = axs[0].hist2d(px_in, py_in, bins=200, cmap='hot', vmax=1200)
    axs[0].set_title("Heatmap of Input Agent Positions")
    axs[0].set_xlabel("Position X")
    axs[0].set_ylabel("Position Y")
    cbar_in = fig.colorbar(hist_in[3], ax=axs[0])
    cbar_in.set_label('Frequency')

    # Output positions heatmap
    hist_out = axs[1].hist2d(px_out, py_out, bins=200, cmap='hot', vmax=1200)
    axs[1].set_title("Heatmap of Output Agent Positions")
    axs[1].set_xlabel("Position X")
    axs[1].set_ylabel("Position Y")
    cbar_out = fig.colorbar(hist_out[3], ax=axs[1])
    cbar_out.set_label('Frequency')

    plt.tight_layout()
    plt.savefig("1b_in_out_positions_heatmap.png", dpi=300)
    plt.show()

# for 1b
def plot_agent_distributions(train_data):
    """
    train_data: shape (10000, 50, 110, 6)
    """

    object_type = train_data[:, :, :, -1]
    px = train_data[:, :, :, 0]
    py = train_data[:, :, :, 1]

    fig, axs = plt.subplots(5,2, figsize=(9,14))
    axs = axs.flatten()

    for obj_id in range(len(object_type_labels)):
        mask = (object_type == obj_id)

        x_vals = px[mask].flatten()
        y_vals = py[mask].flatten()

        hist, xedges, yedges = np.histogram2d(x_vals, y_vals, bins=200)
        hist_prob = hist / hist.sum()

        ax = axs[obj_id]
        im = ax.imshow(hist_prob.T, origin='lower', extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]],
                      cmap='hot', vmax=0.001)

        # Force 1:1 aspect ratio for the data
        ax.set_aspect('equal')

        ax.set_title(object_type_labels[obj_id].capitalize(), fontsize=11)

    # Hide unused subplots
    for i in range(len(object_type_labels), len(axs)):
        fig.delaxes(axs[i])

    # Adjust layout and add colorbar
    fig.subplots_adjust(right=0.88, hspace=0.4, wspace=0.4)
    cbar_ax = fig.add_axes([0.9, 0.15, 0.02, 0.7])
    fig.colorbar(im, cax=cbar_ax, label='Probability Density')

    fig.suptitle("Position Probability Distribution by Object Type", fontsize=16)
    plt.savefig("all_position_prob_dist.png",dpi=300)
    plt.show()

# Visualize heading

def visualize_heading(train_data):
    """
    train_data: shape (10000, 50, 110, 6)
    """
    object_type = train_data[:, :, :, -1]

    heading = train_data[:, :, :, 4]

    import matplotlib.pyplot as plt
    import numpy as np

    plt.figure(figsize=(10, 5))

    bins = 72
    range_ = (-np.pi, np.pi)
    bin_edges = np.linspace(*range_, bins + 1)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    for obj_id in range(len(object_type_labels)):
        mask = (object_type == obj_id)
        headings = heading[mask].flatten()

        counts, _ = np.histogram(headings, bins=bin_edges, density=True)
        prob_dist = counts / counts.sum()
        plt.plot(bin_centers, prob_dist, label=object_type_labels[obj_id])

    plt.title("Heading per Object Type")
    plt.xlabel("Heading (radians)")
    plt.ylabel("Probability")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig("heading_by_obj_type.png", dpi=200)
    plt.show()

# Visualize Regression

def plot_preds_vs_real_paths(full_scene, ego_preds,fname, model_type):
    """
    Inputs:
        full_scene: one instance of a scene. shape (50, 110, 2)
        ego_preds: predicted (x,y) for the ego vehicle
        fname: filename to save the plot to
        model_type: Title for the plot ("Linear Regression" etc)
    """
    plt.figure(figsize=(10, 6))

    num_agents = full_scene.shape[0]
    colors = plt.cm.get_cmap('winter', num_agents)

    for agent_idx in range(full_scene.shape[0]):
        if agent_idx != 0:
            xs = full_scene[agent_idx, :, 0]
            ys = full_scene[agent_idx, :, 1]
            # trim all zeros
            xs = xs[xs != 0]
            ys = ys[ys != 0]
            # plot each line going from full to transparent
            for t in range(len(xs) - 1):
                alpha = (t + 1) / len(xs)
                plt.plot(xs[t:t+2], ys[t:t+2], color=colors(agent_idx), alpha=1-alpha)
        else:
            xs = full_scene[agent_idx, :, 0]
            ys = full_scene[agent_idx, :, 1]
            # trim all zeros
            xs = xs[xs != 0]
            ys = ys[ys != 0]
            # plot each line going from full to transparent
            for t in range(len(xs) - 1):
                alpha = (t + 1) / len(xs)
                plt.plot(xs[t:t+2], ys[t:t+2], color="black", alpha=1-alpha)

    xsp = ego_preds[ :, 0]
    ysp = ego_preds[ :, 1]

    for t in range(len(xsp) - 2):
        alpha = (t + 1) / len(xsp)
        plt.plot(xsp[t:t+2], ysp[t:t+2], color='red', alpha=alpha)
    for t in range(len(xsp) - 2, len(xsp) - 1):
        alpha = (t + 1) / len(xsp)
        plt.plot(xsp[t:t+2], ysp[t:t+2], color='red', alpha=alpha, label="Ego Vehicle Predicted Path")

    # put star on ego origin
    ego_xs = full_scene[0, :, 0]
    ego_ys = full_scene[0, :, 1]
    mask = (ego_xs != 0) | (ego_ys != 0)
    origin_x = ego_xs[mask][0]
    origin_y = ego_ys[mask][0]
    plt.plot(origin_x, origin_y, marker='*', color="black", markersize=10, label='Ego Vehicle Origin')
    plt.plot(ego_xs[mask][-1], ego_ys[mask][-1], marker='*', color="orange", markersize=10, label='Ego Vehicle Actual Endpoint')
    plt.plot(xsp[-1], ysp[-1], marker='*', color="red", markersize=10, label='Ego Vehicle Predicted Endpoint')

    plt.title(f"Predicted Trajectory: {model_type} Model")
    plt.xlabel("X Position")
    plt.ylabel("Y Position")
    plt.grid(True)
    plt.axis('equal')
    plt.legend()
    plt.savefig(fname, dpi=200)
    plt.show()