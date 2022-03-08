import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import cv2


def rot(vec, r):
    R, _ = cv2.Rodrigues(r)
    vec = np.matmul(R, vec)
    return vec


def plot_cam(ax, t, r, rotation_fun=rot):
    x, y, z = t

    x_vec = rotation_fun(np.array([0.3, 0, 0]), r)
    y_vec = rotation_fun(np.array([0, 0.3, 0]), r)
    z_vec = rotation_fun(np.array([0, 0, 1]), r)

    UVW = np.array([x_vec, y_vec, z_vec]).T

    X = [x, x, x]
    Y = [y, y, y]
    Z = [z, z, z]
    U, V, W = UVW
    ax.quiver(X, Y, Z, U, V, W, color=["red", "green", "blue"])


def plot_cams(ts, rs, ponts3d=None, rotation_fun=rot):
    fig = plt.figure()
    plt.title("Camera")
    ax = fig.add_subplot(111, projection='3d')
    for t, r in zip(ts, rs):
        plot_cam(ax, t, r, rotation_fun=rotation_fun)

    if ponts3d is not None:
        xs, ys, zs = ponts3d.T
        ax.scatter(xs, ys, zs)

    ax.set_xlim([-1.2, 1.2])
    ax.set_ylim([-1.2, 1.2])
    ax.set_zlim([-1.2, 1.2])
    ax.view_init(elev=-25, azim=-90)
    plt.show()
