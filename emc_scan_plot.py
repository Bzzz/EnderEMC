#!/usr/bin/env python3
import argparse
import numpy as np
import matplotlib.pyplot as plt

# -------------------------------
# Helper functions
# -------------------------------

def parse_freq_label(label: str) -> float:
    """
    Convert frequency label to Hz.
    Accepts numbers in MHz/kHz/GHz or plain Hz.
    Ignores non-numeric labels.
    """
    try:
        # Remove commas, whitespace
        f_clean = label.replace(',', '.').strip()
        # If it ends with a unit, handle it
        if f_clean.lower().endswith("ghz"):
            return float(f_clean[:-3]) * 1e9
        elif f_clean.lower().endswith("mhz"):
            return float(f_clean[:-3]) * 1e6
        elif f_clean.lower().endswith("khz"):
            return float(f_clean[:-3]) * 1e3
        else:
            return float(f_clean)
    except ValueError:
        # Not a numeric label
        return np.nan


def load_emc_scan(filename):
    """
    Load EMC ASCII scan file with variable-length rows.
    Returns xs, ys, zs, amp_data (n_points x n_freqs), freqs_hz
    """
    xs, ys, zs = [], [], []
    freq_labels = []
    amp_data_dict = {}  # key = freq label, value = list of amplitudes

    with open(filename, 'r', encoding='utf-8') as f:
        lines = [l.strip() for l in f if l.strip()]

    # Parse X, Y, Z
    for line in lines:
        if line.startswith("X:"):
            xs = [float(x.replace(',', '.')) for x in line.split(';')[1:]]
        elif line.startswith("Y:"):
            ys = [float(y.replace(',', '.')) for y in line.split(';')[1:]]
        elif line.startswith("Z:"):
            zs = [float(z.replace(',', '.')) for z in line.split(';')[1:]]

    # Parse amplitude rows
    for line in lines:
        if line.startswith(("X:", "Y:", "Z:")):
            continue
        parts = line.split(';')
        freq = parts[0].replace(',', '.')
        freq_labels.append(freq)
        amps = []
        for a in parts[1:]:
            try:
                amps.append(float(a.replace(',', '.')))
            except ValueError:
                amps.append(np.nan)
        amp_data_dict[freq] = amps

    # Make a rectangular array: n_points x n_freqs
    n_points = len(xs)
    n_freqs = len(freq_labels)
    amp_data = np.full((n_points, n_freqs), np.nan)
    for j, f in enumerate(freq_labels):
        row = amp_data_dict[f]
        length = min(len(row), n_points)
        amp_data[:length, j] = row[:length]

    freqs_hz = np.array([parse_freq_label(f) for f in freq_labels])
    # Keep only valid frequencies
    valid_idx = ~np.isnan(freqs_hz)
    freqs_hz = freqs_hz[valid_idx]
    amp_data = amp_data[:, valid_idx]

    xs = np.array(xs)
    ys = np.array(ys)
    zs = np.array(zs)

    print(f"Loaded {n_points} points with {n_freqs} frequencies")
    return xs, ys, zs, amp_data, freqs_hz


def make_grid(xs, ys, data):
    """Convert list of points into 2D grid for plotting"""
    x_unique = np.unique(xs)
    y_unique = np.unique(ys)
    nx, ny = len(x_unique), len(y_unique)
    grid = np.full((ny, nx, data.shape[1]), np.nan)

    # Detect serpentine scan pattern
    y_idx_map = {y:i for i,y in enumerate(y_unique)}
    x_idx_map = {x:i for i,x in enumerate(x_unique)}

    # Fill grid
    for i, (x, y) in enumerate(zip(xs, ys)):
        xi = x_idx_map[x]
        yi = y_idx_map[y]
        grid[yi, xi, :] = data[i]

    return grid, x_unique, y_unique

def freq_label(f):
    """Pretty print frequency"""
    if f >= 1e9:
        return f"{f/1e9:.2f} GHz"
    elif f >= 1e6:
        return f"{f/1e6:.2f} MHz"
    elif f >= 1e3:
        return f"{f/1e3:.2f} kHz"
    else:
        return f"{f:.2f} Hz"

def make_log_blocks(freqs, n_blocks):
    """Return index ranges for logarithmic bins"""
    edges = np.logspace(np.log10(freqs[0]), np.log10(freqs[-1]), n_blocks+1)
    blocks = []
    start_idx = 0
    for i in range(n_blocks):
        end_idx = len(freqs) if i==n_blocks-1 else np.searchsorted(freqs, edges[i+1], side='left')
        if end_idx <= start_idx:
            end_idx = start_idx + 1
        blocks.append((start_idx, end_idx))
        start_idx = end_idx
    return blocks

def save_image(grid2d, filename, title, vmin, vmax, canvas_size=1000, rotation=0):
    """Save a single 2D grid image with colorbar and title"""
    TITLE_HEIGHT = 200
    IMAGE_SIZE = 800
    BAR_WIDTH = 25
    TEXT_SPACE = 75
    SHIFT_RIGHT = 100
    MARGIN = 10

    # Rotate if needed
    if rotation != 0:
        grid2d = np.rot90(grid2d, k=rotation//90)

    masked = np.ma.masked_invalid(grid2d)
    factor = max(1, IMAGE_SIZE // masked.shape[0], IMAGE_SIZE // masked.shape[1])
    scaled = np.kron(masked, np.ones((factor, factor), dtype=masked.dtype))

    fig = plt.figure(figsize=(10,10), dpi=100)

    ax_img = fig.add_axes([0, 0, IMAGE_SIZE/canvas_size, IMAGE_SIZE/canvas_size])
    cmap = plt.cm.viridis.copy()
    cmap.set_bad("white")
    ax_img.imshow(scaled, origin='lower', cmap=cmap, vmin=vmin, vmax=vmax)
    ax_img.axis('off')

    # Title
    ax_title = fig.add_axes([0, IMAGE_SIZE/canvas_size, IMAGE_SIZE/canvas_size, TITLE_HEIGHT/canvas_size])
    ax_title.axis("off")
    ax_title.text(0.5, 0.5, title, ha="center", va="center", fontsize=16)

    # Colorbar
    cbar_ax = fig.add_axes([(IMAGE_SIZE+SHIFT_RIGHT)/canvas_size, 0, BAR_WIDTH/canvas_size, IMAGE_SIZE/canvas_size])
    grad = np.linspace(vmin, vmax, 256).reshape(-1,1)
    cbar_ax.imshow(grad, origin="lower", aspect="auto", cmap=cmap, extent=[0,BAR_WIDTH,MARGIN,IMAGE_SIZE-MARGIN])
    cbar_ax.axis("off")

    # Text
    ax_text = fig.add_axes([(IMAGE_SIZE+BAR_WIDTH+SHIFT_RIGHT)/canvas_size, 0, TEXT_SPACE/canvas_size, IMAGE_SIZE/canvas_size])
    ax_text.axis("off")
    n_ticks = 6
    for i in range(n_ticks):
        frac = i/(n_ticks-1)
        val = vmin + frac*(vmax-vmin)
        y = MARGIN + frac*(IMAGE_SIZE-2*MARGIN)
        ax_text.text(0, y/IMAGE_SIZE, f"{val:.2f}", va="center", ha="left", transform=ax_text.transAxes, fontsize=10)
    ax_text.text(0.5, 0.5, "Amplitude", rotation=270, va="center", ha="center", transform=ax_text.transAxes, fontsize=12)

    plt.savefig(filename, dpi=100)
    plt.close()

# -------------------------------
# Main
# -------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("file")
    parser.add_argument("--out_prefix", default="out")
    parser.add_argument("--blocks", type=int, default=None, help="Number of log-frequency blocks")
    parser.add_argument("--scale", choices=["global","local"], default="local")
    parser.add_argument("--rotate", type=int, choices=[0,90,180,270], default=0)
    parser.add_argument("--freq_start", type=float, default=None,
                    help="Minimum frequency in Hz (inclusive)")
    parser.add_argument("--freq_end", type=float, default=None,
                    help="Maximum frequency in Hz (inclusive)")

    args = parser.parse_args()

    xs, ys, zs, amp_data, freqs_hz = load_emc_scan(args.file)

    # Apply optional frequency limits
    if args.freq_start is not None or args.freq_end is not None:
        freq_start_hz = args.freq_start if args.freq_start is not None else freqs_hz.min()
        freq_end_hz   = args.freq_end   if args.freq_end   is not None else freqs_hz.max()

        mask = (freqs_hz >= freq_start_hz) & (freqs_hz <= freq_end_hz)
        freqs_hz = freqs_hz[mask]
        amp_data = amp_data[:, mask]

    grid, x_unique, y_unique = make_grid(xs, ys, amp_data)

    if grid.size == 0:
        raise ValueError("Grid is empty!")

    # Compute global vmin/vmax
    valid = grid[~np.isnan(grid)]
    global_min = valid.min() if valid.size>0 else 0
    global_max = valid.max() if valid.size>0 else 1

    # Full average over all frequencies
    avg_all = np.nanmean(grid, axis=2)
    save_image(avg_all, f"{args.out_prefix}_avg.png", "Average", global_min, global_max, rotation=args.rotate)
    print(f"Saved full average image: {args.out_prefix}_avg.png")

    # Logarithmic blocks
    if args.blocks:
        blocks = make_log_blocks(freqs_hz, args.blocks)
        for i,(start,end) in enumerate(blocks):
            block_data = grid[:,:,start:end]
            mask = ~np.all(np.isnan(block_data), axis=2)
            avg_block = np.full(block_data.shape[:2], np.nan)
            if np.any(mask):
                avg_block[mask] = np.nanmean(block_data[mask], axis=1)
            if args.scale=="global":
                vmin,vmax = global_min, global_max
            else:
                valid_block = avg_block[~np.isnan(avg_block)]
                vmin,vmax = (valid_block.min(), valid_block.max()) if valid_block.size>0 else (0,1)
            title = f"Freq: {freq_label(freqs_hz[start])} – {freq_label(freqs_hz[end-1])}"
            save_image(avg_block, f"{args.out_prefix}_block{i+1}.png", title, vmin, vmax, rotation=args.rotate)
            print(f"Saved block {i+1}: {args.out_prefix}_block{i+1}.png")

if __name__=="__main__":
    main()
