import numpy as np
import cv2

# =====================================================================================
# TASK 1 -- FILTERING FOUNDATIONS
# =====================================================================================

def pad_image(img, ph, pw, padding):
    """Pad `img` by ph rows / pw cols on each side.
    padding = 'zero'    -> constant 0
    padding = 'reflect' -> mirror WITHOUT repeating the edge pixel (a b c | b a ...)"""
    if padding == 'zero':
        return np.pad(img, ((ph, ph), (pw, pw)), mode='constant', constant_values=0)
    if padding == 'reflect':
        return np.pad(img, ((ph, ph), (pw, pw)), mode='reflect')
    raise ValueError("padding must be 'zero' or 'reflect'")


def correlate2d(img, kernel, padding='reflect'):
    """2-D cross-correlation, any odd kernel size, 'same' output size.
    out(y,x) = sum_{i,j} K[i,j] * I(y + i - kh//2, x + j - kw//2)
    Vectorised: loop over the kernel offsets, each iteration is one whole-image operation."""
    img = np.asarray(img, dtype=np.float64)
    kernel = np.asarray(kernel, dtype=np.float64)
    kh, kw = kernel.shape
    if kh % 2 == 0 or kw % 2 == 0:
        raise ValueError("kernel dimensions must be odd")
    H, W = img.shape
    p = pad_image(img, kh // 2, kw // 2, padding)
    out = np.zeros((H, W))
    for i in range(kh):                      # loop over kernel offsets only
        for j in range(kw):
            out += kernel[i, j] * p[i:i + H, j:j + W]
    return out


def convolve2d(img, kernel, padding='reflect'):
    """True convolution = correlation with the kernel flipped in both axes."""
    return correlate2d(img, np.asarray(kernel)[::-1, ::-1], padding)


def correlate_separable(img, kx, ky, padding='reflect'):
    """Two 1-D passes: horizontal with kx then vertical with ky.
    Equivalent to correlate2d(img, np.outer(ky, kx)) but costs 2k instead of k^2 per pixel."""
    img = np.asarray(img, dtype=np.float64)
    kx = np.asarray(kx, dtype=np.float64)
    ky = np.asarray(ky, dtype=np.float64)
    H, W = img.shape
    ph, pw = len(ky) // 2, len(kx) // 2
    p = pad_image(img, ph, pw, padding)               # (H+2ph, W+2pw)
    tmp = np.zeros((H + 2 * ph, W))
    for j in range(len(kx)):                          # horizontal pass
        tmp += kx[j] * p[:, j:j + W]
    out = np.zeros((H, W))
    for i in range(len(ky)):                          # vertical pass
        out += ky[i] * tmp[i:i + H, :]
    return out


def gaussian_size(sigma, n_sigma=3.0):
    """Kernel size as a function of sigma: 2*ceil(3 sigma)+1  (truncate at +-3 sigma).
    Mass outside +-3 sigma is only 0.27 % (per axis), so truncation error is negligible,
    while the cost grows only linearly with sigma. Always odd."""
    return 2 * int(np.ceil(n_sigma * sigma)) + 1


def gaussian_kernel1d(sigma, size=None):
    """Normalised 1-D Gaussian (sums to 1)."""
    if size is None:
        size = gaussian_size(sigma)
    x = np.arange(size) - size // 2
    g = np.exp(-x ** 2 / (2.0 * sigma ** 2))
    return g / g.sum()


def gaussian_kernel2d(sigma, size=None):
    """2-D Gaussian = outer product of two 1-D Gaussians (separable by construction)."""
    g = gaussian_kernel1d(sigma, size)
    return np.outer(g, g)


def box_kernel1d(size):
    """Uniform (box) window, sums to 1."""
    return np.ones(size) / size


# Derivative kernels, all scaled so that they return the true slope dI/dx for a ramp.
CENTRAL_X = np.array([[0, 0, 0], [-1, 0, 1], [0, 0, 0]], dtype=float) / 2.0
SOBEL_X = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=float) / 8.0


def gaussian_derivative_kernel_x(sigma=1.0):
    """Derivative-of-Gaussian kernel (x direction): k(x,y) = (x / sigma^2) g(x) g(y),
    renormalised so that sum(x * k) = 1 (i.e. unit response to a unit-slope ramp)."""
    size = gaussian_size(sigma)
    x = np.arange(size) - size // 2
    g = gaussian_kernel1d(sigma, size)
    k = np.outer(g, x / sigma ** 2 * g)                 # rows = y, cols = x
    return k / (k * x[None, :]).sum()


def get_gradient_kernels(name='sobel', sigma_d=1.0):
    """Return (kx, ky) for 'central', 'sobel' or 'dog'."""
    if name == 'central':
        kx = CENTRAL_X
    elif name == 'sobel':
        kx = SOBEL_X
    elif name == 'dog':
        kx = gaussian_derivative_kernel_x(sigma_d)
    else:
        raise ValueError(name)
    return kx, kx.T


# =====================================================================================
# TASK 2 -- HARRIS DETECTOR
# =====================================================================================

def image_gradients(img, grad='sobel', sigma_d=1.0, padding='reflect'):
    """Ix, Iy of an image in [0,1] using the chosen derivative kernel."""
    kx, ky = get_gradient_kernels(grad, sigma_d)
    return correlate2d(img, kx, padding), correlate2d(img, ky, padding)


def window_smooth(a, sigma=1.5, window='gaussian', size=None, padding='reflect'):
    """Integration window w(u,v) applied to a map: Gaussian (sigma) or box of `size` x `size`."""
    if size is None:
        size = gaussian_size(sigma)
    k1 = gaussian_kernel1d(sigma, size) if window == 'gaussian' else box_kernel1d(size)
    return correlate_separable(a, k1, k1, padding)


def structure_tensor(img, sigma=1.5, grad='sobel', window='gaussian', size=None,
                     sigma_d=1.0, padding='reflect'):
    """Return (Sxx, Syy, Sxy): the entries of M = sum_w [[Ix^2, IxIy],[IxIy, Iy^2]] at every pixel."""
    I = np.asarray(img, dtype=np.float64) / 255.0
    Ix, Iy = image_gradients(I, grad, sigma_d, padding)
    Sxx = window_smooth(Ix * Ix, sigma, window, size, padding)
    Syy = window_smooth(Iy * Iy, sigma, window, size, padding)
    Sxy = window_smooth(Ix * Iy, sigma, window, size, padding)
    return Sxx, Syy, Sxy


def harris_R(Sxx, Syy, Sxy, k=0.04):
    """R = det(M) - k * trace(M)^2."""
    det = Sxx * Syy - Sxy ** 2
    tr = Sxx + Syy
    return det - k * tr ** 2


def harris_response(img, sigma=1.5, k=0.04, grad='sobel', window='gaussian', size=None,
                    sigma_d=1.0, padding='reflect'):
    """Full pipeline up to R. Returns R and the tensor entries (Sxx, Syy, Sxy)."""
    Sxx, Syy, Sxy = structure_tensor(img, sigma, grad, window, size, sigma_d, padding)
    return harris_R(Sxx, Syy, Sxy, k), (Sxx, Syy, Sxy)


def nms3x3(R, thr):
    """Fully vectorised 3x3 non-maximum suppression + threshold.
    A pixel survives if R > thr and it is the maximum of its 3x3 neighbourhood.
    Ties (plateaus) are broken by raster order so a plateau yields exactly one pixel:
    neighbours that come earlier in raster order must be strictly smaller."""
    H, W = R.shape
    p = np.pad(R, 1, mode='constant', constant_values=-np.inf)
    keep = R > thr
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            if dy == 0 and dx == 0:
                continue
            nb = p[1 + dy:1 + dy + H, 1 + dx:1 + dx + W]
            if dy < 0 or (dy == 0 and dx < 0):
                keep &= R > nb
            else:
                keep &= R >= nb
    return keep


def corners_from_R(R, rel_thr=0.01, abs_thr=None):
    """Threshold (relative to max R, or absolute) + NMS. Returns (N,2) array of (x,y)."""
    if R.max() <= 0:
        return np.zeros((0, 2))
    thr = abs_thr if abs_thr is not None else rel_thr * R.max()
    ys, xs = np.nonzero(nms3x3(R, thr))
    return np.stack([xs, ys], axis=1).astype(float)


def detect_corners(img, sigma=1.5, k=0.04, rel_thr=0.01, abs_thr=None, **kw):
    """Harris corners of a [0,255] image. Returns (corners (N,2) [x,y], R)."""
    R, _ = harris_response(img, sigma, k, **kw)
    return corners_from_R(R, rel_thr, abs_thr), R


def refine_subpixel(R, pts):
    """Sub-pixel refinement of corner positions: fit a parabola to R along x and along y
    through the 3 samples around each detected pixel and move to its vertex (|shift| <= 0.5)."""
    pts = np.asarray(pts, dtype=float)
    out = pts.copy()
    if len(pts) == 0:
        return out
    x, y = pts[:, 0].astype(int), pts[:, 1].astype(int)
    ok = (x >= 1) & (x < R.shape[1] - 1) & (y >= 1) & (y < R.shape[0] - 1)
    x, y, idx = x[ok], y[ok], np.nonzero(ok)[0]
    c = R[y, x]
    for axis, (a, b) in enumerate((((y, x - 1), (y, x + 1)), ((y - 1, x), (y + 1, x)))):
        lo, hi = R[a], R[b]
        den = lo - 2 * c + hi
        shift = np.where(den < 0, 0.5 * (lo - hi) / np.where(den < 0, den, -1.0), 0.0)
        out[idx, axis] += np.clip(shift, -0.5, 0.5)
    return out


# =====================================================================================
# TASK 3 -- EIGENVALUE ANALYSIS
# =====================================================================================

def eigenvalues_2x2(Sxx, Syy, Sxy):
    """Closed form for the symmetric matrix [[a,b],[b,c]]:
         lam = (a+c)/2 +- sqrt(((a-c)/2)^2 + b^2),    lam1 >= lam2."""
    half_tr = 0.5 * (Sxx + Syy)
    root = np.sqrt((0.5 * (Sxx - Syy)) ** 2 + Sxy ** 2)
    return half_tr + root, half_tr - root


def classify_R(R, rel_thr=0.01):
    """0 = flat (|R| small), 1 = edge (R very negative), 2 = corner (R large positive)."""
    t = rel_thr * R.max()
    cls = np.zeros(R.shape, dtype=int)
    cls[R > t] = 2
    cls[R < -t] = 1
    return cls


def rho_min(k):
    """Smallest lam2/lam1 (lam1>=lam2) for which R>0:  k r^2 + (2k-1) r + k < 0
       -> r > [(1-2k) - sqrt(1-4k)] / (2k)."""
    return ((1 - 2 * k) - np.sqrt(1 - 4 * k)) / (2 * k)


def R_level_curve(lam1, k, c):
    """lam2 on the curve  lam1*lam2 - k (lam1+lam2)^2 = c  (lower branch, lam2 <= lam1).
    Solves k lam2^2 - (1-2k) lam1 lam2 + k lam1^2 + c = 0."""
    disc = (1 - 4 * k) * lam1 ** 2 - 4 * k * c
    lam2 = ((1 - 2 * k) * lam1 - np.sqrt(np.where(disc >= 0, disc, np.nan))) / (2 * k)
    return lam2


# =====================================================================================
# TASK 4 -- TRANSFORMATIONS AND REPEATABILITY
# =====================================================================================

def rotate_image(img, deg):
    """Rotate about the image centre (same canvas size). Returns (B, map_points).
    Border = reflect, so no artificial black wedges (which would create fake strong corners)."""
    h, w = img.shape
    c = np.array([(w - 1) / 2.0, (h - 1) / 2.0])
    th = np.deg2rad(deg)
    Rm = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])
    M = np.hstack([Rm, (c - Rm @ c)[:, None]])             # 2x3 forward map  src -> dst
    B = cv2.warpAffine(img.astype(np.float32), M, (w, h), flags=cv2.INTER_LINEAR,
                       borderMode=cv2.BORDER_REFLECT_101).astype(np.float64)
    return B, (lambda P: P @ Rm.T + (c - Rm @ c)), B.shape


def scale_image(img, s):
    """Resize by factor s. Pixel-centre convention of cv2.resize: x' = (x+.5)*sx - .5."""
    h, w = img.shape
    wn, hn = int(round(w * s)), int(round(h * s))
    B = cv2.resize(img.astype(np.float32), (wn, hn), interpolation=cv2.INTER_LINEAR).astype(np.float64)
    sx, sy = wn / w, hn / h
    return B, (lambda P: np.stack([(P[:, 0] + .5) * sx - .5, (P[:, 1] + .5) * sy - .5], 1)), B.shape


def add_brightness(img, b):
    return np.clip(img + b, 0, 255), (lambda P: P), img.shape


def mul_contrast(img, c):
    return np.clip(img * c, 0, 255), (lambda P: P), img.shape


def add_noise(img, sigma, rng):
    return np.clip(img + rng.normal(0, sigma, img.shape), 0, 255), (lambda P: P), img.shape


def match_fraction(P, Q, tol=2.0):
    """Fraction of points in P having a point of Q within `tol` px (and the boolean mask)."""
    if len(P) == 0:
        return np.nan, np.zeros(0, bool)
    if len(Q) == 0:
        return 0.0, np.zeros(len(P), bool)
    hit = np.zeros(len(P), bool)
    for s in range(0, len(P), 512):                          # chunked pairwise distances
        d2 = ((P[s:s + 512, None, :] - Q[None, :, :]) ** 2).sum(-1)
        hit[s:s + 512] = d2.min(1) <= tol ** 2
    return hit.mean(), hit


def repeatability(cornersA, cornersB, mapfn, shapeB, tol=2.0):
    """Fraction of A's corners that, mapped into B, lie within `tol` px of a B corner.
    Corners whose image falls outside B are excluded. Returns (rep, n_inside)."""
    if len(cornersA) == 0:
        return np.nan, 0
    P = mapfn(cornersA)
    hB, wB = shapeB
    inside = (P[:, 0] >= 0) & (P[:, 0] <= wB - 1) & (P[:, 1] >= 0) & (P[:, 1] <= hB - 1)
    P = P[inside]
    if len(P) == 0:
        return np.nan, 0
    frac, _ = match_fraction(P, cornersB, tol)
    return frac, int(inside.sum())


# =====================================================================================
# TEST IMAGES
# =====================================================================================

def make_checkerboard(n=8, sq=32):
    """Anti-aliased checkerboard whose inner corners are EXACTLY at pixel centres.
    Returns (image [0,255], ground-truth corners (x,y)); corner (i,j) is at (sq*i, sq*j).
    Each pixel is the exact area average of the ideal pattern: the integral of a separable
    pattern over a pixel square is the product of the 1-D integrals, so a pixel centred on
    an edge gets 0.5 and a pixel centred on a corner also gets 0.5."""
    N = n * sq
    sub = (np.arange(1000) + 0.5) / 1000.0 - 0.5              # sub-pixel sample offsets
    t = np.arange(N)[:, None] + sub[None, :]                   # (N, 1000) positions
    wave = np.where(np.floor(t / sq).astype(int) % 2 == 0, 1.0, -1.0)
    u = wave.mean(1)                                           # integrated square wave in [-1,1]
    img = 255.0 * (0.5 + 0.5 * np.outer(u, u))
    gt = np.array([(sq * i, sq * j) for i in range(1, n) for j in range(1, n)], dtype=float)
    return img, gt


# =====================================================================================
# PART B -- EXPERIMENTS (Tasks 1-5, bonus, failure case)
# =====================================================================================
import os, sys, json, time
import numpy as np
import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import SymLogNorm, ListedColormap
from matplotlib.lines import Line2D
import scipy.ndimage as ndi            # verification only

ROOT = os.path.dirname(os.path.abspath(__file__)) if '__file__' in globals() else os.getcwd()   # works as a script and in a notebook cell
DATA, OUT = os.path.join(ROOT, 'data'), os.path.join(ROOT, 'results')
SIGMA, K, REL = 1.5, 0.04, 0.01        # default detector parameters (window sigma, k, rel. threshold)
plt.rcParams.update({'font.size': 7, 'axes.titlesize': 7.5, 'axes.labelsize': 7,
                     'legend.fontsize': 6, 'xtick.labelsize': 6.5, 'ytick.labelsize': 6.5,
                     'figure.dpi': 100, 'savefig.dpi': 220, 'lines.linewidth': 1.0})
RESULTS = {}


# ------------------------------------------------------------------------------------
# data
# ------------------------------------------------------------------------------------
def prepare_data():
    """Create the three grayscale test images (once) from scikit-image sample data and save
    them as PNG. Replace data/*.png by your own images if you wish -- nothing else changes."""
    os.makedirs(DATA, exist_ok=True)
    if not all(os.path.exists(os.path.join(DATA, f)) for f in ('brick.png', 'cat.png', 'moon.png')):
        import skimage.data as sd                    # only used to obtain the sample pictures
        for name, src in (('brick', 'brick'), ('cat', 'chelsea'), ('moon', 'moon')):
            im = getattr(sd, src)()
            if im.ndim == 3:                         # colour -> gray, centre-crop to square
                im = cv2.cvtColor(im, cv2.COLOR_RGB2GRAY)
                h, w = im.shape; s = min(h, w)
                im = im[(h - s) // 2:(h - s) // 2 + s, (w - s) // 2:(w - s) // 2 + s]
            cv2.imwrite(os.path.join(DATA, name + '.png'), cv2.resize(im, (256, 256), interpolation=cv2.INTER_AREA))
    cb, _ = make_checkerboard()
    cv2.imwrite(os.path.join(DATA, 'checkerboard_generated.png'), np.round(cb).astype(np.uint8))


def load_images():
    """Return dict name -> float image in [0,255], plus the checkerboard ground truth.
    If the instructor's checkerboard is placed at data/checkerboard.png it replaces the
    generated one in Tasks 2-5 (ground truth is only known for the generated one)."""
    imgs = {}
    for name in ('brick', 'cat', 'moon'):
        imgs[name] = cv2.imread(os.path.join(DATA, name + '.png'), cv2.IMREAD_GRAYSCALE).astype(float)
    cb_path = os.path.join(DATA, 'checkerboard.png')
    gen, gt = make_checkerboard()
    imgs['checker'] = (cv2.imread(cb_path, cv2.IMREAD_GRAYSCALE).astype(float)
                       if os.path.exists(cb_path) else gen)
    return imgs, gen, gt


TITLES = {'brick': 'brick (man-made)', 'cat': 'cat (natural, textured)',
          'moon': 'moon (low contrast)', 'checker': 'checkerboard'}
COL = {'brick': 'C0', 'cat': 'C2', 'moon': 'C1', 'checker': 'C3'}


def save(fig, name):
    os.makedirs(OUT, exist_ok=True)
    fig.savefig(os.path.join(OUT, name), bbox_inches='tight')
    plt.close(fig)


def best_time(f, n=3):
    ts = []
    for _ in range(n):
        t0 = time.perf_counter(); f(); ts.append(time.perf_counter() - t0)
    return min(ts)


# ------------------------------------------------------------------------------------
# TASK 1
# ------------------------------------------------------------------------------------
def task1(imgs):
    rng = np.random.default_rng(0)
    img = imgs['brick']
    R1 = {}
    # --- (a) verification of correlate2d / convolve2d against OpenCV and SciPy ----------
    bmap_cv = {'zero': cv2.BORDER_CONSTANT, 'reflect': cv2.BORDER_REFLECT_101}
    bmap_sp = {'zero': 'constant', 'reflect': 'mirror'}
    kernels = [('Gaussian s=1.5 (11x11)', gaussian_kernel2d(1.5)), ('Gaussian s=3 (19x19)', gaussian_kernel2d(3.0)),
               ('Sobel-x', SOBEL_X), ('Central diff-x', CENTRAL_X),
               ('random 5x5', rng.standard_normal((5, 5))), ('random 7x3', rng.standard_normal((7, 3)))]
    ver = []
    for name, ker in kernels:
        for pad in ('zero', 'reflect'):
            mine = correlate2d(img, ker, pad)
            e_cv = np.abs(mine - cv2.filter2D(img, -1, ker, borderType=bmap_cv[pad])).max()
            e_sp = np.abs(mine - ndi.correlate(img, ker, mode=bmap_sp[pad])).max()
            ver.append(dict(kernel=name, padding=pad, err_cv2=float(e_cv), err_scipy=float(e_sp)))
    ker = kernels[4][1]
    for pad in ('zero', 'reflect'):                                   # true convolution vs scipy
        e = np.abs(convolve2d(img, ker, pad) - ndi.convolve(img, ker, mode=bmap_sp[pad])).max()
        ver.append(dict(kernel='random 5x5 (convolution)', padding=pad, err_cv2=None, err_scipy=float(e)))
    g1 = gaussian_kernel1d(1.5)
    e = np.abs(correlate_separable(img, g1, g1, 'reflect') -
               cv2.GaussianBlur(img, (11, 11), 1.5, borderType=cv2.BORDER_REFLECT_101)).max()
    ver.append(dict(kernel='separable Gaussian vs cv2.GaussianBlur', padding='reflect', err_cv2=float(e), err_scipy=None))
    R1['verification'] = ver
    R1['max_err_overall'] = float(max(max(v['err_cv2'] or 0, v['err_scipy'] or 0) for v in ver))

    # --- (b) Gaussian size rule: truncation error ---------------------------------------
    from math import erf, sqrt
    trunc = []
    for n in (2, 3, 4):
        tail = 1 - erf(n / sqrt(2))                                  # mass outside +-n sigma (per axis)
        ref = correlate_separable(img, gaussian_kernel1d(2.0, gaussian_size(2.0, 6)), gaussian_kernel1d(2.0, gaussian_size(2.0, 6)))
        g = gaussian_kernel1d(2.0, gaussian_size(2.0, n))
        trunc.append(dict(n_sigma=n, size_s2=gaussian_size(2.0, n), tail_mass=tail,
                          max_err_vs_6sigma=float(np.abs(correlate_separable(img, g, g) - ref).max())))
    R1['truncation'] = trunc

    # --- (c) separability ---------------------------------------------------------------
    G = gaussian_kernel2d(1.5)
    sv = np.linalg.svd(G, compute_uv=False)
    R1['singular_values'] = sv[:4].tolist()
    R1['sep_err_vs_2d'] = float(np.abs(correlate2d(img, G, 'reflect') -
                                       correlate_separable(img, g1, g1, 'reflect')).max())
    big = rng.random((512, 512))
    sizes = [3, 5, 7, 11, 15, 21, 31, 41]
    t2, ts = [], []
    for ks in sizes:
        s = (ks - 1) / 6.0                                           # => gaussian_size(s) == ks
        gk = gaussian_kernel1d(s, ks); G2 = np.outer(gk, gk)
        t2.append(best_time(lambda: correlate2d(big, G2, 'reflect')))
        ts.append(best_time(lambda: correlate_separable(big, gk, gk, 'reflect')))
    R1['speed'] = dict(sizes=sizes, t_2d=t2, t_sep=ts, speedup=[a / b for a, b in zip(t2, ts)],
                       theory=[k / 2 for k in sizes], image='512x512')
    RESULTS['task1'] = R1

    fig, ax = plt.subplots(1, 3, figsize=(7.3, 1.9))
    ax[0].loglog(sizes, np.array(t2) * 1e3, 'o-', label='2-D  (k$^2$ offsets)')
    ax[0].loglog(sizes, np.array(ts) * 1e3, 's-', label='separable (2k offsets)')
    ax[0].set(xlabel='kernel size k', ylabel='time (ms), 512x512', title='(a) run time'); ax[0].legend()
    ax[1].plot(sizes, R1['speed']['speedup'], 'o-', label='measured')
    ax[1].plot(sizes, [k / 2 for k in sizes], 'k--', label='theory k/2')
    ax[1].set(xlabel='kernel size k', ylabel='speed-up', title='(b) separable speed-up'); ax[1].legend()
    ax[2].semilogy(range(1, 5), sv[:4] + 1e-20, 'o-')
    ax[2].set(xlabel='singular value index', ylabel='$\\sigma_i$ of 2-D Gaussian', title='(c) rank of Gaussian kernel = 1')
    ax[2].set_xticks(range(1, 5))
    fig.tight_layout(); save(fig, 't1_filtering.png')


# ------------------------------------------------------------------------------------
# TASK 2
# ------------------------------------------------------------------------------------
CLS_CMAP = ListedColormap(['#e6e6e6', '#2b6cb0', '#d62728'])        # flat, edge, corner


def heatmap_figure(imgs, name_list, fname, picks_fixed=None):
    fig, axes = plt.subplots(len(name_list), 3, figsize=(7.3, 2.45 * len(name_list)))
    axes = np.atleast_2d(axes)
    info = {}
    for r, name in enumerate(name_list):
        img = imgs[name]
        R, (Sxx, Syy, Sxy) = harris_response(img, SIGMA, K)
        Rmax = R.max(); cls = classify_R(R, REL)
        if name == 'checker' and picks_fixed:
            pts = picks_fixed
        else:
            tr = Sxx + Syy
            inner = np.zeros(R.shape, bool); inner[10:-10, 10:-10] = True      # ignore the border band
            yx = lambda a: np.unravel_index(a, R.shape)
            c = yx(np.argmax(np.where(inner, R, -np.inf)))
            e = yx(np.argmin(np.where(inner, R, np.inf)))
            f = yx(np.argmin(np.where(inner, tr, np.inf)))
            pts = dict(corner=(c[1], c[0]), edge=(e[1], e[0]), flat=(f[1], f[0]))
        mk = dict(corner=('o', 'C3'), edge=('s', 'C0'), flat=('^', 'C2'))
        info[name] = {kk: dict(xy=[int(v[0]), int(v[1])], R_over_Rmax=float(R[v[1], v[0]] / Rmax),
                               trace=float((Sxx + Syy)[v[1], v[0]])) for kk, v in pts.items()}
        info[name]['mean_R_over_Rmax'] = {lab: float(R[cls == i].mean() / Rmax) if (cls == i).any() else None
                                          for i, lab in enumerate(['flat', 'edge', 'corner'])}
        info[name]['fraction_pixels'] = {lab: float((cls == i).mean()) for i, lab in enumerate(['flat', 'edge', 'corner'])}
        absmax = np.abs(R).max()
        a0, a1, a2 = axes[r]
        a0.imshow(img, cmap='gray', vmin=0, vmax=255); a0.set_title(f'{TITLES[name]}')
        im = a1.imshow(R, cmap='RdBu_r', norm=SymLogNorm(linthresh=REL * Rmax, vmin=-absmax, vmax=absmax))
        a1.set_title('R heat map (symlog scale)')
        cb = fig.colorbar(im, ax=a1, fraction=0.046, pad=0.02); cb.ax.tick_params(labelsize=5.5)
        a2.imshow(cls, cmap=CLS_CMAP, vmin=0, vmax=2, interpolation='nearest')
        a2.set_title(f'classification of R (|R| vs {REL:g}·Rmax)')
        for kk, v in pts.items():
            for a in (a0, a1):
                a.plot(v[0], v[1], mk[kk][0], mfc='none', mec=mk[kk][1], ms=7, mew=1.4)
        h = [Line2D([0], [0], marker=mk[kk][0], ls='', mfc='none', mec=mk[kk][1], mew=1.2,
                    label=f'{kk}: R/Rmax={info[name][kk]["R_over_Rmax"]:+.3f}') for kk in pts]
        a1.legend(handles=h, loc='lower center', bbox_to_anchor=(0.5, -0.30), fontsize=5.5, ncol=1, frameon=False)
        a2.legend(handles=[plt.Rectangle((0, 0), 1, 1, fc=c_, label=l_) for c_, l_ in
                           zip(['#d62728', '#2b6cb0', '#e6e6e6'], ['corner (R>0, large)', 'edge (R<0, large)', 'flat (|R|~0)'])],
                  loc='lower center', bbox_to_anchor=(0.5, -0.30), fontsize=5.5, frameon=False)
        for a in axes[r]: a.set_xticks([]); a.set_yticks([])
    fig.tight_layout(); save(fig, fname)
    return info


def task2(imgs):
    info = heatmap_figure(imgs, ['checker', 'cat'], 't2_heatmaps.png',
                          picks_fixed=dict(corner=(96, 96), edge=(96, 80), flat=(80, 80)))
    RESULTS['task2'] = dict(params=dict(sigma=SIGMA, k=K, rel_thr=REL, grad='sobel/8', window_size=gaussian_size(SIGMA)),
                            heatmap_points=info)
    names = ['brick', 'cat', 'moon', 'checker']
    fig, ax = plt.subplots(1, 4, figsize=(7.3, 1.95))
    counts = {}
    for a, n in zip(ax, names):
        c, R = detect_corners(imgs[n], SIGMA, K, REL)
        counts[n] = len(c)
        a.imshow(imgs[n], cmap='gray', vmin=0, vmax=255)
        a.scatter(c[:, 0], c[:, 1], s=9, facecolors='none', edgecolors='r', linewidths=0.6)
        a.set_title(f'{TITLES[n]}\n{len(c)} corners'); a.set_xticks([]); a.set_yticks([])
        f2, a2 = plt.subplots(figsize=(3.2, 3.2))
        a2.imshow(imgs[n], cmap='gray', vmin=0, vmax=255)
        a2.scatter(c[:, 0], c[:, 1], s=14, facecolors='none', edgecolors='r', linewidths=0.8)
        a2.axis('off'); a2.set_title(f'{TITLES[n]}: {len(c)} corners'); save(f2, f'corners_{n}.png')
    fig.tight_layout(); save(fig, 't2_corners_overlay.png')
    RESULTS['task2']['corner_counts'] = counts


# ------------------------------------------------------------------------------------
# TASK 3
# ------------------------------------------------------------------------------------
def task3(imgs):
    R3 = {}
    # numerical identities on all four images
    ident = {}
    for n, img in imgs.items():
        Sxx, Syy, Sxy = structure_tensor(img, SIGMA)
        l1, l2 = eigenvalues_2x2(Sxx, Syy, Sxy)
        det = Sxx * Syy - Sxy ** 2; tr = Sxx + Syy
        ident[n] = dict(pixels=int(det.size), max_abs_det_err=float(np.abs(det - l1 * l2).max()),
                        max_abs_trace_err=float(np.abs(tr - (l1 + l2)).max()), max_det=float(det.max()), max_trace=float(tr.max()),
                        min_lambda2=float(l2.min()), max_lambda1=float(l1.max()),
                        R_formula_err=float(np.abs((det - K * tr ** 2) - (l1 * l2 - K * (l1 + l2) ** 2)).max()))
    R3['identities'] = ident
    # minimum ratio for corner: closed form, polynomial roots, brute force
    mr = {}
    for k in (0.04, 0.06):
        roots = np.roots([k, 2 * k - 1, k])
        rho = np.linspace(0, 1, 2_000_001)
        Rv = rho - k * (1 + rho) ** 2                                    # R / lam1^2
        mr[str(k)] = dict(closed_form=float(rho_min(k)), polynomial_root=float(roots.min()),
                          brute_force=float(rho[np.argmax(Rv > 0)]), reciprocal=float(1 / rho_min(k)),
                          R_over_l1sq_at_rho1=float(1 - 4 * k))
    R3['min_ratio'] = mr
    # scatter plots
    names = ['checker', 'brick', 'cat']
    fig, ax = plt.subplots(1, 3, figsize=(7.3, 2.45))
    for a, n in zip(ax, names):
        Sxx, Syy, Sxy = structure_tensor(imgs[n], SIGMA)
        l1, l2 = eigenvalues_2x2(Sxx, Syy, Sxy)
        R = harris_R(Sxx, Syy, Sxy, K); cls = classify_R(R, REL)
        lim = l1.max() * 1.03
        order = np.argsort(cls.ravel(), kind='stable')                   # corners drawn last
        a.scatter(l1.ravel()[order], np.clip(l2.ravel()[order], 0, None), s=1.2, c=cls.ravel()[order],
                  cmap=CLS_CMAP, vmin=0, vmax=2, rasterized=True, linewidths=0)
        x = np.linspace(1e-9, lim, 600)
        a.plot(x, rho_min(0.04) * x, 'k-', lw=0.9)                       # R=0, k=0.04
        a.plot(x, rho_min(0.06) * x, 'k--', lw=0.9)                      # R=0, k=0.06
        a.plot(x, R_level_curve(x, K, REL * R.max()), color='#d62728', ls=':', lw=1.1)
        a.plot(x, R_level_curve(x, K, -REL * R.max()), color='#2b6cb0', ls=':', lw=1.1)
        a.plot(x, x, color='gray', lw=0.5)
        a.set_xlim(0, lim); a.set_ylim(0, lim * 0.6)
        a.set_xlabel('$\\lambda_1$ (larger)'); a.set_ylabel('$\\lambda_2$ (smaller)')
        a.set_title(f'{TITLES[n]}')
    h = [Line2D([0], [0], marker='o', ls='', color=c_, ms=3, label=l_) for c_, l_ in
         zip(['#d62728', '#2b6cb0', '#bbbbbb'], ['corner', 'edge', 'flat'])]
    h += [Line2D([0], [0], color='k', label='R=0, k=0.04 ($\\lambda_2/\\lambda_1$=0.0436)'),
          Line2D([0], [0], color='k', ls='--', label='R=0, k=0.06 (0.0685)'),
          Line2D([0], [0], color='#d62728', ls=':', label='R=+0.01 Rmax'),
          Line2D([0], [0], color='#2b6cb0', ls=':', label='R=-0.01 Rmax')]
    fig.tight_layout(rect=(0, 0.1, 1, 1))
    fig.legend(handles=h, loc='lower center', fontsize=5.8, frameon=False, ncol=7, columnspacing=1.0, handlelength=1.6)
    save(fig, 't3_eigen_scatter.png')
    RESULTS['task3'] = R3


# ------------------------------------------------------------------------------------
# TASK 4
# ------------------------------------------------------------------------------------
ROT = list(range(0, 181, 15))
SCALE = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.25, 1.5, 1.75, 2.0]
BRIGHT = list(range(0, 81, 10))
CONTR = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.25, 1.5, 1.75, 2.0]
NOISE = [0, 5, 10, 15, 20, 25, 30]


def transform(A, kind, p, rng):
    if kind == 'rot':    return rotate_image(A, p)
    if kind == 'scale':  return scale_image(A, p)
    if kind == 'bright': return add_brightness(A, p)
    if kind == 'contr':  return mul_contrast(A, p)
    if kind == 'noise':  return add_noise(A, p, rng)
    raise ValueError(kind)


def rep_curve(A, kind, params, mode='fixed', det_kw=None, ntrials=1, k=K, sigma=SIGMA):
    """Repeatability of corners of A under a family of transformations.
    mode='fixed'    : absolute R threshold T = REL * Rmax(A) used for A *and* B
    mode='relative' : B is thresholded with REL * Rmax(B)
    Returns (repeatability list, mean #corners in B list, mean #A corners kept inside B)."""
    det_kw = det_kw or {}
    cA, RA = detect_corners(A, sigma, k, REL, **det_kw)
    T = REL * RA.max()
    reps, nBs, nIn = [], [], []
    for p in params:
        rr, nb, ni = [], [], []
        for t in range(ntrials):
            B, mp, shp = transform(A, kind, p, np.random.default_rng(1000 + t))
            if mode == 'fixed':
                cB, _ = detect_corners(B, sigma, k, abs_thr=T, **det_kw)
            else:
                cB, _ = detect_corners(B, sigma, k, REL, **det_kw)
            r, n_in = repeatability(cA, cB, mp, shp, tol=2.0)
            rr.append(r); nb.append(len(cB)); ni.append(n_in)
        reps.append(float(np.nanmean(rr))); nBs.append(float(np.mean(nb))); nIn.append(float(np.mean(ni)))
    return reps, nBs, nIn, len(cA)


def task4(imgs):
    names = ['brick', 'cat', 'moon', 'checker']
    fam = [('rot', ROT, 'rotation angle (deg)', 1), ('scale', SCALE, 'scale factor', 1),
           ('bright', BRIGHT, 'additive brightness b', 1), ('contr', CONTR, 'contrast factor c', 1),
           ('noise', NOISE, 'noise $\\sigma_n$', 3)]
    R4 = {'params': {k_: v for k_, v, _, _ in fam}, 'fixed': {}, 'relative': {}, 'nA': {}}
    for n in names:
        R4['fixed'][n], R4['relative'][n] = {}, {}
        for kind, params, _, nt in fam:
            if kind in ('rot', 'contr'):
                modes = ('fixed', 'relative')
            else:
                modes = ('fixed',)
            for mode in modes:
                rep, nB, nIn, nA = rep_curve(imgs[n], kind, params, mode, ntrials=nt)
                R4[mode][n][kind] = dict(rep=rep, nB=nB, n_inside=nIn)
        R4['nA'][n] = nA
    fig, ax = plt.subplots(2, 4, figsize=(7.3, 3.55))
    A = ax.ravel()
    for i, (kind, params, xl, _) in enumerate(fam):
        for n in names:
            A[i].plot(params, R4['fixed'][n][kind]['rep'], 'o-', ms=2.5, color=COL[n], label=TITLES[n].split(' ')[0])
        A[i].set(xlabel=xl, ylabel='repeatability', ylim=(-0.02, 1.03),
                 title=f'({"abcde"[i]}) {dict(rot="rotation", scale="scale", bright="brightness", contr="contrast", noise="noise")[kind]}')
    A[0].legend(loc='lower left', ncol=2, handlelength=1.2, columnspacing=0.8)
    for j, (kind, params, xl) in enumerate([('rot', ROT, 'rotation angle (deg)'), ('contr', CONTR, 'contrast factor c')]):
        a = A[5 + j]
        for mode, ls in (('fixed', '-'), ('relative', '--')):
            M = np.array([R4[mode][n][kind]['rep'] for n in names])
            a.plot(params, np.nanmean(M, 0), ls, marker='o', ms=2.5, label=f'{mode} threshold')
            a.fill_between(params, np.nanmin(M, 0), np.nanmax(M, 0), alpha=0.15)
        a.set(xlabel=xl, ylabel='repeatability (mean, range)', ylim=(-0.02, 1.03),
              title=f'({"fg"[j]}) {"rotation" if kind == "rot" else "contrast"}: fixed vs rel. thr.')
        a.legend(loc='lower left' if kind == 'rot' else 'lower right')
    a = A[7]
    for mode, ls in (('fixed', '-'), ('relative', '--')):
        M = np.array([np.array(R4[mode][n]['contr']['nB']) / R4['nA'][n] for n in names])
        a.semilogy(CONTR, np.mean(M, 0), ls, marker='o', ms=2.5, label=f'{mode} threshold')
    a.axhline(1, color='gray', lw=0.5)
    a.set(xlabel='contrast factor c', ylabel='corners in B / in A', title='(h) corner count vs contrast')
    a.legend()
    fig.tight_layout(); save(fig, 't4_invariance.png')
    RESULTS['task4'] = R4


# ------------------------------------------------------------------------------------
# TASK 5
# ------------------------------------------------------------------------------------
def time_fn(f, n=15):
    f()
    ts = []
    for _ in range(n):
        t0 = time.perf_counter(); f(); ts.append(time.perf_counter() - t0)
    return float(np.median(ts)) * 1e3                                         # ms


def task5(imgs, gen, gt):
    R5 = {}
    names = ['brick', 'cat', 'moon', 'checker']
    sig_l, k_l, thr_l = [1.0, 1.5, 2.0, 3.0], [0.02, 0.04, 0.06], [0.001, 0.01, 0.1]
    # --- (a) grid + counts ---------------------------------------------------------------
    table = []
    for s in sig_l:
        for k in k_l:
            for t in thr_l:
                row = dict(sigma=s, k=k, thr=t)
                for n in names:
                    row[n] = len(detect_corners(imgs[n], s, k, t)[0])
                table.append(row)
    R5['counts'] = table
    import csv
    with open(os.path.join(OUT, 't5_param_counts.csv'), 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=['sigma', 'k', 'thr'] + names); w.writeheader(); w.writerows(table)
    rows = [1.0, 1.5, 3.0]
    cols = [('k=0.02', 1.5, 0.02, 0.01), ('k=0.04 (baseline)', 1.5, 0.04, 0.01), ('k=0.06', 1.5, 0.06, 0.01),
            ('thr=0.001', 1.5, 0.04, 0.001), ('thr=0.1', 1.5, 0.04, 0.1)]
    fig, ax = plt.subplots(3, 5, figsize=(7.3, 4.5))
    for i, s in enumerate(rows):
        for j, (lab, _, k, t) in enumerate(cols):
            c, _ = detect_corners(imgs['brick'], s, k, t)
            a = ax[i, j]; a.imshow(imgs['brick'], cmap='gray', vmin=0, vmax=255)
            a.scatter(c[:, 0], c[:, 1], s=4, facecolors='none', edgecolors='r', linewidths=0.4)
            a.set_xticks([]); a.set_yticks([])
            a.set_title(f'$\\sigma$={s}, {lab.split(" ")[0]}  (n={len(c)})', fontsize=6)
    fig.tight_layout(pad=0.3); save(fig, 't5_param_grid.png')
    # --- (b) Gaussian vs box window of equal size -------------------------------------------
    size = gaussian_size(SIGMA)
    gb = {}
    for wname, kw in (('gaussian', dict(window='gaussian')), ('box', dict(window='box', size=size))):
        d = dict(counts={n: len(detect_corners(imgs[n], SIGMA, K, REL, **kw)[0]) for n in names})
        # localisation on the generated checkerboard (ground truth known); tol = 8 px so that the
        # box window's displaced plateau picks are still counted as (badly localised) hits
        c, Rw = detect_corners(gen, SIGMA, K, REL, **kw)
        d['localisation'] = localisation_stats(c, gt, gen.shape, tol=8.0)
        d['localisation_tol3'] = localisation_stats(c, gt, gen.shape, tol=3.0)
        # plateau: pixels within 1 % of the corner peak in the 21x21 neighbourhood of a corner
        pk = Rw[96 - 10:96 + 11, 96 - 10:96 + 11]
        d['plateau_pixels'] = int((pk >= 0.99 * pk.max()).sum())
        # rotation repeatability averaged over images and angles 15..180 (fixed threshold)
        rep = np.array([rep_curve(imgs[n], 'rot', ROT[1:], 'fixed', det_kw=kw)[0] for n in names])
        d['rot_rep_mean'] = float(np.nanmean(rep)); d['rot_rep_45'] = float(np.nanmean(rep[:, ROT[1:].index(45)]))
        # noise repeatability (sigma_n = 10)
        nz = np.array([rep_curve(imgs[n], 'noise', [10], 'fixed', det_kw=kw, ntrials=3)[0][0] for n in names])
        d['noise10_rep_mean'] = float(np.nanmean(nz))
        # peak side-lobe of the 1-D window's frequency response (relative to DC), in dB
        win = gaussian_kernel1d(SIGMA, size) if wname == 'gaussian' else box_kernel1d(size)
        Hf = np.abs(np.fft.rfft(win, 4096)); Hf /= Hf[0]
        first_min = np.nonzero((np.diff(Hf)[:-1] < 0) & (np.diff(Hf)[1:] >= 0))[0][0] + 1
        d['peak_sidelobe_dB'] = float(20 * np.log10(Hf[first_min:].max()))
        gb[wname] = d
    R5['window'] = gb
    # --- (c) comparison with cv2.cornerHarris -----------------------------------------------
    cmp_ = {}
    for n in names:
        img = imgs[n]; im32 = (img / 255.0).astype(np.float32)
        R_cv = cv2.cornerHarris(im32, size, 3, K)
        c_cv = corners_from_R(R_cv, REL)
        out = {'n_cv2': len(c_cv)}
        for lab, kw in (('gauss', dict(window='gaussian')), ('box', dict(window='box', size=size))):
            c_me, R_me = detect_corners(img, SIGMA, K, REL, **kw)
            m1, _ = match_fraction(c_me, c_cv, 2.0); m2, _ = match_fraction(c_cv, c_me, 2.0)
            out[lab] = dict(n_mine=len(c_me), mine_in_cv2=float(m1), cv2_in_mine=float(m2),
                            corr_R=float(np.corrcoef(R_me.ravel(), R_cv.ravel())[0, 1]))
        out['t_cv2_ms'] = time_fn(lambda: cv2.cornerHarris(im32, size, 3, K))
        out['t_mine_ms'] = time_fn(lambda: harris_response(img, SIGMA, K))
        out['t_mine_box_ms'] = time_fn(lambda: harris_response(img, SIGMA, K, window='box', size=size))
        out['t_mine_total_ms'] = time_fn(lambda: detect_corners(img, SIGMA, K, REL))
        cmp_[n] = out
    R5['cv2'] = cmp_
    RESULTS['task5'] = R5


def localisation_stats(c, gt, shape, tol=3.0, border=8, valid=None):
    """Ground-truth based statistics: recall (GT corners with a detection within `tol` px),
    false positives (detections not near any GT; ignoring the image-border band where the
    finite image creates genuine T-junctions, and detections for which valid(c) is False),
    mean / RMS distance of the matched detections to their GT corner."""
    if len(c) == 0:
        return dict(recall=0.0, false_pos=0, mean_err=None, rms_err=None)
    D = np.sqrt(((gt[:, None] - c[None]) ** 2).sum(-1))                      # (n_gt, n_det)
    dmin = D.min(1); hit = dmin <= tol
    h, w = shape
    keep = (c[:, 0] >= border) & (c[:, 0] < w - border) & (c[:, 1] >= border) & (c[:, 1] < h - border)
    if valid is not None:
        keep &= valid(c)
    fp = int(((D.min(0) > tol) & keep).sum())
    return dict(recall=float(hit.mean()), false_pos=fp,
                mean_err=float(dmin[hit].mean()) if hit.any() else None,
                rms_err=float(np.sqrt((dmin[hit] ** 2).mean())) if hit.any() else None)


# ------------------------------------------------------------------------------------
# BONUS
# ------------------------------------------------------------------------------------
def bonus(imgs, gen, gt):
    """Central difference vs Sobel vs derivative-of-Gaussian (sigma_d=1) as gradient operator.
    Localisation errors are reported for integer-pixel detections and after parabolic
    sub-pixel refinement (integer positions are quantisation limited to ~0.4 px)."""
    ops = [('central', dict(grad='central')), ('sobel', dict(grad='sobel')), ('dog', dict(grad='dog', sigma_d=1.0))]
    RB = {'localisation': {}, 'noise': {}}
    th = 22.5
    Brot, mp, shp = rotate_image(gen, th)
    gt_rot = mp(gt)
    ins = (gt_rot[:, 0] > 12) & (gt_rot[:, 0] < 243) & (gt_rot[:, 1] > 12) & (gt_rot[:, 1] < 243)
    gt_rot = gt_rot[ins]
    # detections are only judged inside the footprint of the original (un-reflected) image
    th_r = np.deg2rad(th); Rm = np.array([[np.cos(th_r), -np.sin(th_r)], [np.sin(th_r), np.cos(th_r)]])
    ctr = np.array([127.5, 127.5])
    inside_orig = lambda P: np.all(np.abs((P - ctr) @ Rm) < 127.5 - 8, axis=1)   # inverse-rotate, test footprint
    for name, kw in ops:
        d = {}
        def run(img, G, valid=None, nt=1, noise=0):
            res = []
            for t in range(nt):
                im = add_noise(img, noise, np.random.default_rng(50 + t))[0] if noise else img
                c, R = detect_corners(im, SIGMA, K, REL, **kw)
                res.append((localisation_stats(c, G, img.shape, valid=valid),
                            localisation_stats(refine_subpixel(R, c), G, img.shape, valid=valid)))
            f = lambda i, key: float(np.nanmean([r[i][key] if r[i][key] is not None else np.nan for r in res]))
            return dict(recall=f(1, 'recall'), false_pos=f(1, 'false_pos'),
                        err_int=f(0, 'mean_err'), err_sub=f(1, 'mean_err'))
        d['aligned'] = run(gen, gt)
        d['rot22.5'] = run(Brot, gt_rot, valid=inside_orig)
        d['noise10'] = run(gen, gt, nt=5, noise=10)
        RB['localisation'][name] = d
        RB['noise'][name] = {n: rep_curve(imgs[n], 'noise', NOISE, 'fixed', det_kw=kw, ntrials=3)[0] for n in ('brick', 'checker')}
    fig, ax = plt.subplots(1, 2, figsize=(4.6, 1.9))
    for a, n in zip(ax, ('brick', 'checker')):
        for name, _ in ops:
            a.plot(NOISE, RB['noise'][name][n], 'o-', ms=2.5, label={'central': 'central diff', 'sobel': 'Sobel', 'dog': 'derivative of Gaussian'}[name])
        a.set(xlabel='noise $\\sigma_n$', ylabel='repeatability', ylim=(-0.02, 1.03), title=TITLES[n])
    ax[0].legend(loc='lower left'); fig.tight_layout(); save(fig, 'bonus_gradient_operators.png')
    RESULTS['bonus'] = RB


# ------------------------------------------------------------------------------------
# FAILURE CASE: rounded corners
# ------------------------------------------------------------------------------------
def rounded_square(r, N=96, half=30):
    """Square of half-size `half` with corner radius r, anti-aliased via signed distance."""
    y, x = np.mgrid[0:N, 0:N].astype(float)
    c = (N - 1) / 2.0
    qx, qy = np.abs(x - c) - (half - r), np.abs(y - c) - (half - r)
    d = np.hypot(np.maximum(qx, 0), np.maximum(qy, 0)) + np.minimum(np.maximum(qx, qy), 0) - r
    img = 40 + 175 * np.clip(0.5 - d, 0, 1)
    off = (half - r) + r / np.sqrt(2)                                          # arc mid-points
    gt = np.array([(c + sx * off, c + sy * off) for sx in (-1, 1) for sy in (-1, 1)])
    return img, gt


def failure_case():
    radii = [0, 4, 8, 16, 28]
    scales = [1.5, 3.0, 6.0]
    # thresholds calibrated on the ideal right-angle corner at each scale
    ideal, _ = rounded_square(0)
    T = {s: REL * harris_response(ideal, s, K)[0].max() for s in scales}
    out = dict(radii=radii, detected_single=[], detected_multi=[], rho_s1_5=[], rho_s6=[], R_over_T_single=[])
    fig = plt.figure(figsize=(7.3, 2.3))
    gs = fig.add_gridspec(2, 6, width_ratios=[1, 1, 1, 1, 1, 2.3], wspace=0.08, hspace=0.12)
    sc_col = {1.5: 'r', 3.0: 'orange', 6.0: 'm'}
    for j, r in enumerate(radii):
        img, gt = rounded_square(r)
        tol = 3 + r / 2.0
        found = {}
        for s in scales:
            R, (Sxx, Syy, Sxy) = harris_response(img, s, K)
            pts = corners_from_R(R, abs_thr=T[s])
            found[s] = pts
            l1, l2 = eigenvalues_2x2(Sxx, Syy, Sxy)
            if s in (1.5, 6.0):
                rho = [float((l2 / np.maximum(l1, 1e-12))[int(round(g[1])), int(round(g[0]))]) for g in gt]
                out['rho_s1_5' if s == 1.5 else 'rho_s6'].append(float(np.mean(rho)))
            if s == 1.5:
                out['R_over_T_single'].append(float(np.mean([R[int(round(g[1])), int(round(g[0]))] / T[s] for g in gt])))
        def hits(P):
            return int(sum((np.sqrt(((P - g) ** 2).sum(1)).min() <= tol) if len(P) else False for g in gt))
        out['detected_single'].append(hits(found[1.5]))
        out['detected_multi'].append(hits(np.vstack([found[s] for s in scales])))
        for row in range(2):
            a = fig.add_subplot(gs[row, j]); a.imshow(img, cmap='gray', vmin=0, vmax=255)
            if row == 0:
                a.scatter(found[1.5][:, 0], found[1.5][:, 1], s=18, facecolors='none', edgecolors='r', linewidths=0.9)
                a.set_title(f'r={r}: {out["detected_single"][-1]}/4', fontsize=6.5)
            else:
                for s in scales[::-1]:
                    a.scatter(found[s][:, 0], found[s][:, 1], s=18 + 5 * (s > 2), facecolors='none', edgecolors=sc_col[s], linewidths=0.9)
                a.set_xlabel(f'multi-scale: {out["detected_multi"][-1]}/4', fontsize=6.5, labelpad=1)
            a.set_xticks([]); a.set_yticks([])
            if j == 0:
                a.set_ylabel('$\\sigma$=1.5 only' if row == 0 else '$\\sigma\\in${1.5,3,6}', fontsize=6.5)
    a = fig.add_subplot(gs[:, 5])
    rr = np.array(radii)
    a.plot(rr, out['rho_s1_5'], 'o-', color='r', label='$\\sigma$=1.5')
    a.plot(rr, out['rho_s6'], 's-', color='m', label='$\\sigma$=6')
    a.axhline(rho_min(K), color='k', ls='--', lw=0.8); a.text(rr[0] + 0.5, rho_min(K) * 0.62, '$\\rho_{min}$(k=0.04)=0.0436', ha='left', fontsize=5.8)
    a.set(yscale='log', xlabel='corner radius r (px)', ylabel='$\\lambda_2/\\lambda_1$ at the corner', title='eigenvalue ratio at arc centre')
    a.yaxis.set_label_position('right'); a.yaxis.tick_right(); a.legend(loc='lower left')
    save(fig, 'failure_rounded_corners.png')
    RESULTS['failure'] = out


# ------------------------------------------------------------------------------------
def main(tasks):
    prepare_data()
    imgs, gen, gt = load_images()
    os.makedirs(OUT, exist_ok=True)
    jobs = dict(t1=lambda: task1(imgs), t2=lambda: task2(imgs), t3=lambda: task3(imgs), t4=lambda: task4(imgs),
                t5=lambda: task5(imgs, gen, gt), bonus=lambda: bonus(imgs, gen, gt), failure=failure_case)
    path = os.path.join(OUT, 'results.json')
    if os.path.exists(path):
        RESULTS.update(json.load(open(path)))
    tasks = [t for t in (tasks or []) if t in jobs]          # ignore unknown arguments (e.g. from Jupyter/Colab)
    for t in (tasks or list(jobs)):
        t0 = time.time(); jobs[t](); print(f'{t} done in {time.time() - t0:.1f}s', flush=True)
        json.dump(RESULTS, open(path, 'w'), indent=1)


if __name__ == '__main__':
    main(sys.argv[1:])