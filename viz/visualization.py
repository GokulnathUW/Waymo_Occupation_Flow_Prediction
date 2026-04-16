"""
Visualization utilities for Occupancy Flow Prediction.
Provides functions to:
- Generate step-by-step visualizations of agent trajectories
- Create animations from visualization frames
- Plot roadgraph and agent positions
"""

import uuid

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf


def create_figure_and_axes(size_pixels: int = 1000) -> tuple:
    """Initializes a unique figure and axes for plotting"""
    fig, ax = plt.subplots(1, 1, num=uuid.uuid4())

    # Set output image to pixel resolution
    dpi = 100
    size_inches = size_pixels / dpi
    fig.set_size_inches([size_inches, size_inches])
    fig.set_dpi(dpi)
    fig.set_facecolor("white")
    ax.set_facecolor("white")
    ax.xaxis.label.set_color("black")
    ax.tick_params(axis="x", colors="black")
    ax.yaxis.label.set_color("black")
    ax.tick_params(axis="y", colors="black")
    fig.set_tight_layout(True)
    ax.grid(False)

    return fig, ax


def fig_canvas_image(fig: plt.Figure) -> np.ndarray:
    """Returns a [H, W, 3] uint8 np.array image from fig.canvas.tostring_rgb()"""
    # Adjust margins to display ticks properly
    fig.subplots_adjust(
        left=0.08, bottom=0.08, right=0.98, top=0.98, wspace=0.0, hspace=0.0
    )
    fig.canvas.draw()
    data = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    return data.reshape(fig.canvas.get_width_height()[::-1] + (3,))


def get_colormap(num_agents: int) -> np.ndarray:
    """Compute a color map array of shape [num_agents, 4]"""
    try:
        # matplotlib 3.7+
        colors = plt.colormaps.get_cmap("jet").resampled(num_agents)
    except AttributeError:
        # matplotlib < 3.7 (deprecated but still works)
        colors = plt.cm.get_cmap("jet", num_agents)

    colors = colors(range(num_agents))
    np.random.shuffle(colors)
    return colors


def get_viewport(
    all_states: np.ndarray, all_states_mask: np.ndarray
) -> tuple[float, float, float]:
    """Gets the region containing the data for proper viewport framing"""
    valid_states = all_states[all_states_mask]
    all_y = valid_states[..., 1]
    all_x = valid_states[..., 0]

    center_y = (np.max(all_y) + np.min(all_y)) / 2
    center_x = (np.max(all_x) + np.min(all_x)) / 2

    range_y = np.ptp(all_y)
    range_x = np.ptp(all_x)

    width = max(range_y, range_x)

    return center_y, center_x, width


def visualize_one_step(
    states: np.ndarray,
    mask: np.ndarray,
    roadgraph: np.ndarray,
    title: str,
    center_y: float,
    center_x: float,
    width: float,
    color_map: np.ndarray,
    size_pixels: int = 1000,
) -> np.ndarray:
    """Generate visualization for a single timestep"""
    fig, ax = create_figure_and_axes(size_pixels=size_pixels)

    # Plot roadgraph
    rg_pts = roadgraph[:, :2].T
    ax.plot(rg_pts[0, :], rg_pts[1, :], "k.", alpha=1, ms=2)

    # Filter states by mask
    masked_x = states[:, 0][mask]
    masked_y = states[:, 1][mask]
    colors = color_map[mask]

    # Plot agent positions
    ax.scatter(masked_x, masked_y, marker="o", linewidths=3, color=colors)

    # Set title and axes
    ax.set_title(title)
    size = max(10, width * 1.0)
    ax.axis(
        [
            -size / 2 + center_x,
            size / 2 + center_x,
            -size / 2 + center_y,
            size / 2 + center_y,
        ]
    )
    ax.set_aspect("equal")

    # Convert to image
    image = fig_canvas_image(fig)
    plt.close(fig)

    return image


def visualize_all_agents_smooth(
    decoded_example: dict,
    size_pixels: int = 1000,
) -> list[np.ndarray]:
    """Visualizes all agent trajectories (past, current, future) as a series of images."""
    # Extract past states [num_agents, num_past_steps, 2]
    past_states = tf.stack(
        [decoded_example["state/past/x"], decoded_example["state/past/y"]],
        axis=-1,
    ).numpy()
    past_states_mask = decoded_example["state/past/valid"].numpy() > 0.0

    # Extract current states [num_agents, 1, 2]
    current_states = tf.stack(
        [
            decoded_example["state/current/x"],
            decoded_example["state/current/y"],
        ],
        axis=-1,
    ).numpy()
    current_states_mask = decoded_example["state/current/valid"].numpy() > 0.0

    # Extract future states [num_agents, num_future_steps, 2]
    future_states = tf.stack(
        [decoded_example["state/future/x"], decoded_example["state/future/y"]],
        axis=-1,
    ).numpy()
    future_states_mask = decoded_example["state/future/valid"].numpy() > 0.0

    # Extract roadgraph [num_points, 3]
    roadgraph_xyz = decoded_example["roadgraph_samples/xyz"].numpy()

    num_agents, num_past_steps, _ = past_states.shape
    num_future_steps = future_states.shape[1]

    # Generate color map for agents
    color_map = get_colormap(num_agents)

    # Concatenate all states for viewport calculation
    all_states = np.concatenate(
        [past_states, current_states, future_states], axis=1
    )
    all_states_mask = np.concatenate(
        [past_states_mask, current_states_mask, future_states_mask], axis=1
    )

    # Calculate viewport
    center_y, center_x, width = get_viewport(all_states, all_states_mask)

    images = []

    # Generate images for past timesteps
    for i, (s, m) in enumerate(
        zip(
            np.split(past_states, num_past_steps, axis=1),
            np.split(past_states_mask, num_past_steps, axis=1),
        )
    ):
        im = visualize_one_step(
            s[:, 0],
            m[:, 0],
            roadgraph_xyz,
            f"past: {num_past_steps - i}",
            center_y,
            center_x,
            width,
            color_map,
            size_pixels,
        )
        images.append(im)

    # Generate image for current timestep
    im = visualize_one_step(
        current_states[:, 0],
        current_states_mask[:, 0],
        roadgraph_xyz,
        "current",
        center_y,
        center_x,
        width,
        color_map,
        size_pixels,
    )
    images.append(im)

    # Generate images for future timesteps
    for i, (s, m) in enumerate(
        zip(
            np.split(future_states, num_future_steps, axis=1),
            np.split(future_states_mask, num_future_steps, axis=1),
        )
    ):
        im = visualize_one_step(
            s[:, 0],
            m[:, 0],
            roadgraph_xyz,
            f"future: {i + 1}",
            center_y,
            center_x,
            width,
            color_map,
            size_pixels,
        )
        images.append(im)

    return images


def create_animation(
    images: list[np.ndarray], interval: int = 100
) -> "matplotlib.animation.FuncAnimation":
    """Creates a Matplotlib animation from a list of images"""
    import matplotlib.animation as animation

    plt.ioff()
    fig, ax = plt.subplots()
    dpi = 100
    size_inches = 1000 / dpi
    fig.set_size_inches([size_inches, size_inches])
    plt.ion()

    def animate_func(i):
        ax.imshow(images[i])
        ax.set_xticks([])
        ax.set_yticks([])
        ax.grid("off")

    anim = animation.FuncAnimation(
        fig, animate_func, frames=len(images), interval=interval
    )
    plt.close(fig)

    return anim
