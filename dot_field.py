"""Random dot field utility"""

import numpy as np
import matplotlib.pyplot as plt

class DotField:
    """A simple random dot field."""

    def __init__(self, width, height, dot_density, rng=None):
        """Initialize the dot field.

        Parameters
        ----------
        width : int or float
            Width of the rectangular field.
        height : int or float
            Height of the rectangular field.
        dot_density : float
            Fraction of pixels occupied by dots (0..1). Determines the number of
            dots as ``width * height * dot_density``.
        rng : numpy.random.Generator, optional
            Random generator for reproducibility.
        """
        self.width = width
        self.height = height
        self.dot_density = dot_density
        self.rng = rng or np.random.default_rng()
        self.positions = self._initialize_positions()

    def plot(self, ax=None, **kwargs):
        """Display the current dot field using matplotlib.

        Parameters
        ----------
        ax : matplotlib.axes.Axes, optional
            Axes to draw on. A new figure and axes are created if ``None``.
        **kwargs : dict
            Additional keyword arguments passed to ``matplotlib.axes.Axes.scatter``.

        Returns
        -------
        matplotlib.axes.Axes
            The axes containing the plot.
        """
        if ax is None:
            fig, ax = plt.subplots()
        ax.scatter(self.positions[:, 0], self.positions[:, 1], s=2, **kwargs)
        ax.set_xlim(0, self.width)
        ax.set_ylim(0, self.height)
        ax.set_aspect('equal')
        return ax

    def _initialize_positions(self):
        count = int(self.width * self.height * self.dot_density)
        x = self.rng.uniform(0, self.width, count)
        y = self.rng.uniform(0, self.height, count)
        return np.vstack([x, y]).T

    def update(self, mode="translation", direction=(1, 0), speed=1.0):
        """Update dot positions for one frame.

        Parameters
        ----------
        mode : {"translation", "cw", "ccw"}
            Motion type. ``"cw"`` and ``"ccw"`` rotate dots clockwise or
            counterclockwise about the field center. ``"translation"`` shifts the
            dots by a vector.
        direction : tuple of float
            Direction vector for translation mode. Ignored when ``mode`` is
            ``"cw"`` or ``"ccw"``. The vector does not need to be normalized.
        speed : float
            Speed of motion. For translation, it scales the ``direction`` vector.
            For rotations, it is interpreted as radians per frame.
        """
        if mode == "translation":
            dx, dy = np.asarray(direction) * speed
            self.positions[:, 0] += dx
            self.positions[:, 1] += dy
        elif mode in {"cw", "ccw"}:
            angle = speed if mode == "ccw" else -speed
            cx, cy = self.width / 2.0, self.height / 2.0
            shifted = self.positions - [cx, cy]
            cos_a = np.cos(angle)
            sin_a = np.sin(angle)
            rot_matrix = np.array([[cos_a, -sin_a], [sin_a, cos_a]])
            rotated = shifted @ rot_matrix.T
            self.positions = rotated + [cx, cy]
        else:
            raise ValueError(f"Unsupported mode: {mode}")
        return self.positions.copy()

if __name__ == "__main__":
    # Basic demonstration
    field = DotField(width=200, height=150, dot_density=0.01)
    field.plot()
    plt.title("Random Dot Field")
    plt.show()
