import numpy as np
import matplotlib.pyplot as plt


def plot_polar_heatmap(num_cakes, rad, z_data, first_cake_angle):
    """A method for plotting a polar heatmap."""
    azm = np.linspace(0, 2 * np.pi, num_cakes + 1)
    r, theta = np.meshgrid(rad, azm)
    plt.subplot(projection="polar", theta_direction=-1,
                theta_offset=np.deg2rad(360 / num_cakes / 2))
    plt.pcolormesh(theta, r, z_data.T)
    plt.plot(azm, r, ls='none')
    plt.grid()
    # Turn on theta grid lines at the cake edges
    plt.thetagrids([theta * 360 / num_cakes for theta in range(num_cakes)], labels=[])
    # Turn off radial grid lines
    plt.rgrids([])
    # Put the cake numbers in the right places
    ax = plt.gca()
    trans, _, _ = ax.get_xaxis_text1_transform(0)
    for label in range(1, num_cakes + 1):
        ax.text(np.deg2rad(label * 10 - 95 + first_cake_angle), -0.1, label,
                transform=trans, rotation=0, ha="center", va="center")
    plt.show()
