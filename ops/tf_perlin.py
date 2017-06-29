import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt


def make_canvas(h, w, vec):
    qh = h // 2
    qw = w // 2
    vec = tf.squeeze(vec)
    ul = vec[0]
    ur = vec[1]
    bl = vec[2]
    br = vec[3]
    tl = tf.ones((qh, qw)) * ul
    tr = tf.ones((qh, qw)) * ur
    bl = tf.ones((qh, qw)) * bl
    br = tf.ones((qh, qw)) * br
    tp = tf.concat([tl, tr], axis=1)
    bp = tf.concat([bl, br], axis=1)
    return tf.concat([tp, bp], axis=0)


def perlin(h, w, x, y, xi, yi, xf, yf, u, v, vectors, pn_range=256):
    # permutation table
    # np.random.seed(seed)
    pn = np.arange(pn_range, dtype=np.float32)
    shuff_p = tf.random_shuffle(pn)
    pn = tf.reshape((tf.stack([shuff_p, shuff_p])), [pn_range * 2])

    n00 = gradient(
        h=tf.gather(pn, tf.cast(tf.gather(pn, xi) + yi, tf.int32)),
        x=xf,
        y=yf,
        vectors=vectors)
    n01 = gradient(
        h=tf.gather(pn, tf.cast(tf.gather(pn, xi) + yi + 1, tf.int32)),
        x=xf,
        y=yf - 1,
        vectors=vectors)
    n11 = gradient(
        h=tf.gather(pn, tf.cast(tf.gather(pn, xi + 1) + yi + 1, tf.int32)),
        x=xf - 1,
        y=yf - 1,
        vectors=vectors)
    n10 = gradient(
        h=tf.gather(pn, tf.cast(tf.gather(pn, xi + 1) + yi, tf.int32)),
        x=xf - 1,
        y=yf,
        vectors=vectors)

    # combine noises
    x1 = lerp(n00, n10, u)
    x2 = lerp(n01, n11, u)  # FIX1: I was using n10 instead of n01
    return lerp(x1, x2, v)  # FIX2: I also had to reverse x1 and x2 here


def lerp(a, b, x):
    """linear interpolation"""
    return a + x * (b - a)


def fade(t):
    """6t^5 - 15t^4 + 10t^3"""
    return 6 * t**5 - 15 * t**4 + 10 * t**3


def gradient(h, x, y, vectors):
    """grad converts h to the right gradient vector
    and return the dot product with (x,y)"""
    g = tf.gather(vectors, tf.cast(tf.mod(h, 4), tf.int32))
    gs = tf.split(g, 2, axis=2)
    return tf.squeeze(gs[0]) * x + tf.squeeze(gs[1]) * y


def get_noise(h, w):
    vectors = tf.constant([[0, 1], [0, -1], [1, 0], [-1, 0]], dtype=tf.float32)
    lin_h = np.linspace(0, np.random.randint(2) + 1, h, endpoint=False)
    lin_w = np.linspace(0, np.random.randint(2) + 1, w, endpoint=False)
    x, y = np.meshgrid(lin_h, lin_w)
    # coordinates of the top-left
    xi = x.astype(int)
    yi = y.astype(int)
    # internal coordinates
    xf = x - xi
    yf = y - yi
    # fade factors
    u = fade(xf)
    v = fade(yf)
    # Perlin noise
    return perlin(h, 2, x, y, xi, yi, xf, yf, u, v, vectors)
    # adj_constant = tf.constant(1.) - tf.round(tf.random_uniform((), minval=0, maxval=1))
    # adj_pnoise = adj_constant - tf.abs(pnoise)
    # return adj_pnoise


def main():
    p = get_noise(h=240, w=320)
    sess = tf.Session()
    perlin = sess.run(p)
    plt.imshow(perlin)
    plt.show()


if __name__ == '__main__':
    main()

