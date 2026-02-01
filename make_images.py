#!/usr/bin/env python3
import os
import glob
import numpy as np
import matplotlib.pyplot as plt
import argparse
import re

#
# example: python make_images.py ./20260131_210248_X100-250_Y190-310_Z76_S1/ --blocks 20 --scale global

#
def parse_filename(fname):
    m = re.search(r'X([-+]?\d*\.?\d+)_Y([-+]?\d*\.?\d+)_Z([-+]?\d*\.?\d+)', fname)
    if not m:
        raise ValueError(f"Cannot parse XYZ from filename: {fname}")
    return map(float, m.groups())

def load_files(folder):
    files = glob.glob(os.path.join(folder, 'scan_X*_Y*_Z*.csv'))
    data, xs, ys, zs = [], [], [], []
    freqs = None

    for f in files:
        x, y, z = parse_filename(os.path.basename(f))
        xs.append(x); ys.append(y); zs.append(z)

        tmp = np.loadtxt(f, delimiter=',')
        if freqs is None:
            freqs = tmp[:, 0]  # Frequency column
        data.append(tmp[:, 1])  # Amplitude column

    return np.array(xs), np.array(ys), np.array(zs), np.array(data), freqs


def make_grid(xs, ys, data):
    x_unique = np.unique(xs)
    y_unique = np.unique(ys)
    grid = np.full((len(y_unique), len(x_unique), data.shape[1]), np.nan)  # <-- NaN

    for i, (x, y) in enumerate(zip(xs, ys)):
        xi = np.where(x_unique == x)[0][0]
        yi = np.where(y_unique == y)[0][0]
        grid[yi, xi, :] = data[i]

    return grid




def freq_label(f):
    if f >= 1e9:
        return f"{f/1e9:.2f} GHz"
    elif f >= 1e6:
        return f"{f/1e6:.2f} MHz"
    elif f >= 1e3:
        return f"{f/1e3:.2f} kHz"
    else:
        return f"{f:.2f} Hz"

def scale_grid(grid_2d, canvas_size):
    h, w = grid_2d.shape
    factor = max(1, min(canvas_size // h, canvas_size // w))
    return np.kron(grid_2d, np.ones((factor, factor)))


def save_image(grid_2d, filename, title, vmin, vmax, canvas_size=1000):
    import numpy as np
    import matplotlib.pyplot as plt

    TITLE_HEIGHT = 200
    IMAGE_SIZE = 800
    TOTAL_RIGHT = 150
    BAR_WIDTH = 25
    TEXT_SPACE = TOTAL_RIGHT - BAR_WIDTH
    SHIFT_RIGHT = 100  # px shift for bar+text only
    MARGIN = 10        # px margin at top/bottom so labels aren’t cut

    # --- Mask NaNs ---
    masked_grid = np.ma.masked_invalid(grid_2d)

    # --- Integer scaling ---
    factor = max(1, min(IMAGE_SIZE // masked_grid.shape[0],
                         IMAGE_SIZE // masked_grid.shape[1]))
    scaled = np.kron(masked_grid, np.ones((factor, factor), dtype=masked_grid.dtype))

    # --- Figure ---
    fig = plt.figure(figsize=(10, 10), dpi=100)

    # --- Image axes (unchanged) ---
    ax_img = fig.add_axes([0, 0, IMAGE_SIZE / canvas_size, IMAGE_SIZE / canvas_size])
    cmap = plt.cm.viridis.copy()
    cmap.set_bad("white")
    ax_img.imshow(scaled, origin='lower', cmap=cmap, vmin=vmin, vmax=vmax)
    ax_img.axis('off')

    # --- Title axes (unchanged) ---
    ax_title = fig.add_axes([0, IMAGE_SIZE / canvas_size,
                             IMAGE_SIZE / canvas_size, TITLE_HEIGHT / canvas_size])
    ax_title.axis("off")
    ax_title.text(0.5, 0.5, title, ha="center", va="center", fontsize=16)

    # --- Colorbar axes (shifted 100 px right) ---
    cbar_ax = fig.add_axes([(IMAGE_SIZE + SHIFT_RIGHT) / canvas_size, 0,
                            BAR_WIDTH / canvas_size, IMAGE_SIZE / canvas_size])
    # --- Bar image with margin ---
    grad = np.linspace(vmin, vmax, 256).reshape(-1, 1)
    y0 = MARGIN
    y1 = IMAGE_SIZE - MARGIN
    cbar_ax.imshow(grad, origin="lower", aspect="auto", cmap=cmap,
                   extent=[0, BAR_WIDTH, y0, y1])
    cbar_ax.axis("off")

    # --- Text axes (remaining space) ---
    ax_text = fig.add_axes([(IMAGE_SIZE + BAR_WIDTH + SHIFT_RIGHT) / canvas_size, 0,
                            TEXT_SPACE / canvas_size, IMAGE_SIZE / canvas_size])
    ax_text.axis("off")

    n_ticks = 6
    for i in range(n_ticks):
        frac = i / (n_ticks - 1)
        val = vmin + frac * (vmax - vmin)
        # map frac from 0→1 to margin→IMAGE_SIZE-MARGIN
        y = MARGIN + frac * (IMAGE_SIZE - 2 * MARGIN)
        ax_text.text(0, y / IMAGE_SIZE, f"{val:.2f}", va="center", ha="left",
                     transform=ax_text.transAxes, fontsize=10)

    # --- Bar label ---
    ax_text.text(0.5, 0.5, "Amplitude", rotation=270,
                 va="center", ha="center", transform=ax_text.transAxes, fontsize=12)

    plt.savefig(filename, dpi=100)
    plt.close()



def make_log_blocks(freqs, n_blocks, debug=False):
    """
    Create logarithmically spaced frequency bins and return index ranges
    covering the samples inside each bin.
    """
    f_min = freqs[0]
    f_max = freqs[-1]

    # Logarithmic bin edges in Hz
    edges = np.logspace(np.log10(f_min), np.log10(f_max), n_blocks + 1)

    if debug:
        print("Log bin edges (Hz):")
        for e in edges:
            if e >= 1e9:
                print(f"  {e/1e9:.3f} GHz")
            elif e >= 1e6:
                print(f"  {e/1e6:.3f} MHz")
            elif e >= 1e3:
                print(f"  {e/1e3:.3f} kHz")
            else:
                print(f"  {e:.3f} Hz")
        print()

    blocks = []
    start_idx = 0

    for i in range(n_blocks):
        # Last bin takes all remaining
        if i == n_blocks - 1:
            end_idx = len(freqs)
        else:
            # Find first index >= next edge
            end_idx = np.searchsorted(freqs, edges[i+1], side='left')

        # Guarantee at least one sample per bin
        if end_idx <= start_idx:
            end_idx = start_idx + 1

        blocks.append((start_idx, end_idx))

        if debug:
            print(f"Block {i+1}: indices {start_idx} – {end_idx-1}, "
                  f"freqs {freqs[start_idx]:.2f} Hz – {freqs[end_idx-1]:.2f} Hz")

        start_idx = end_idx

    if debug:
        print(f"\nTotal blocks returned: {len(blocks)}\n")

    return blocks



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("folder")
    parser.add_argument("--freq_start", type=int)
    parser.add_argument("--freq_count", type=int)
    parser.add_argument("--blocks", type=int)
    parser.add_argument("--out_prefix", default="out")
    parser.add_argument("--canvas_size", type=int, default=1000)
    parser.add_argument("--scale", choices=["global", "local"], default="local", help="Color scaling mode: 'global' uses one scale for all images, 'local' scales each image individually")

    args = parser.parse_args()

    xs, ys, zs, data, freqs = load_files(args.folder)
    grid = make_grid(xs, ys, data)


    # Compute global min/max across entire dataset, ignoring NaNs
    all_valid = grid[~np.isnan(grid)]
    if all_valid.size == 0:
        global_vmin, global_vmax = 0, 1
    else:
        global_vmin, global_vmax = all_valid.min(), all_valid.max()

    print(f"Global scale: vmin={global_vmin:.2f}, vmax={global_vmax:.2f}")

    global_min = np.min(grid)
    global_max = np.max(grid)

    # --- Full average image ---
    avg_all = np.mean(grid, axis=2)
    save_image(avg_all,
               f"{args.out_prefix}_avg.png",
               f"Average: {freq_label(freqs[0])} – {freq_label(freqs[-1])}",
               global_min, global_max, args.canvas_size)
    print("Saved average image")

    # --- Block mode ---
    if args.blocks:
        blocks = make_log_blocks(freqs, args.blocks, debug=True)

        for b, (start, end) in enumerate(blocks):
            # Compute per-block average safely
            block_slice = grid[:, :, start:end]

            avg_block = np.full(block_slice.shape[:2], np.nan)
            mask = ~np.all(np.isnan(block_slice), axis=2)
            if np.any(mask):
                avg_block[mask] = np.nanmean(block_slice[mask], axis=1)

            # Choose color scale mode
            if args.scale == "global":
                vmin, vmax = global_vmin, global_vmax
            else:  # local
                valid_data = avg_block[~np.isnan(avg_block)]
                if valid_data.size == 0:
                    vmin, vmax = 0, 1
                else:
                    vmin, vmax = valid_data.min(), valid_data.max()


            # Build title for the frequency block
            title = f"Frequency: {freq_label(freqs[start])} – {freq_label(freqs[end-1])}"

            # Save the image; masked NaNs will appear white
            save_image(avg_block,
                    f"{args.out_prefix}_block{b+1}.png",
                    title=title,
                    vmin=vmin,
                    vmax=vmax,
                    canvas_size=args.canvas_size)

            print(f"Saved block image: {args.out_prefix}_block{b+1}.png")




    # --- Specific frequency range mode ---
    elif args.freq_count is not None:
        start = args.freq_start
        end = start + args.freq_count

        avg_block = np.mean(grid[:, :, start:end], axis=2)
        title = f"Frequency: {freq_label(freqs[start])} – {freq_label(freqs[end-1])}"

        save_image(avg_block,
                   f"{args.out_prefix}_freq_{start}_{end}.png",
                   title, global_min, global_max, args.canvas_size)

        print("Saved selected frequency range")

if __name__ == "__main__":
    main()
