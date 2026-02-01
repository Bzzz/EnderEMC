import sys
import time
import subprocess
import requests
import argparse
import json
import numpy as np
import argparse
import vxi11
import sys
import os

from datetime import datetime
#
# example: python3 printermove.py --xstart 100 --ystart 190 --xend 250 --yend 310 --step 1 --z 76 --start 100e3 --stop 800e6 --decades=1
#

# ------------------ DEFAULTS ------------------
BASE_URL = "http://192.168.0.149:7125"

FEEDRATE = None  # Use printer default feedrate if not set

# ------------------ UTILS ------------------
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="Spiral raster printer moves")
    parser.add_argument("--xstart", type=float, default=0)
    parser.add_argument("--ystart", type=float, default=0)
    parser.add_argument("--xend", type=float, default=360)
    parser.add_argument("--yend", type=float, default=388)
    parser.add_argument("--step", type=float, default=10)
    parser.add_argument("--z", type=float, default=50)
    parser.add_argument("--start", type=float, default=1e6)
    parser.add_argument("--stop", type=float, default=3e9)
    #parser.add_argument("--rbw", type=float, default=1000)
    #parser.add_argument("--points", type=int, default=1001)
    parser.add_argument("--decades", type=float, default=1)
    return vars(parser.parse_args())  # returns a dict


def send_gcode(script: str):
    url = f"{BASE_URL}/printer/gcode/script"
    r = requests.post(url, json={"script": script})
    r.raise_for_status()

_last_pos = {"x": None, "y": None, "z": None}
_first_print = True

def move_and_wait(x, y, z, folder, args, feedrate=None):

    global _last_pos, _first_print  # <--- important!
    if feedrate:
        move_line = f"G1 X{x:.3f} Y{y:.3f} Z{z:.3f} F{feedrate}"
    else:
        move_line = f"G1 X{x:.3f} Y{y:.3f} Z{z:.3f}"

    gcode = f"""
{move_line}
M400
G4 P1000
"""
    send_gcode(gcode)
    #print(f"Reached: X{x:.2f} Y{y:.2f} Z{z:.2f}")
    # Prepare a string that only shows changes
    changes = []
    if _last_pos["x"] is None or _last_pos["x"] != x:
        changes.append(f"X{x:.2f}")
    if _last_pos["y"] is None or _last_pos["y"] != y:
        changes.append(f"Y{y:.2f}")
    if _last_pos["z"] is None or _last_pos["z"] != z:
        changes.append(f"Z{z:.2f}")

    if changes:
        if _first_print:
            # first print
            print("Reached:", " ".join(changes), end="", flush=True)
            _first_print = False
        else:
            # append changes on the same line
            print(" → " + " ".join(changes), end="", flush=True)

    # update last position
    _last_pos["x"] = x
    _last_pos["y"] = y
    _last_pos["z"] = z


    on_point_reached(x, y, z, folder, args)

def on_point_reached(x, y, z, folder, args):

    # Build a clean filename that encodes the coordinates
    outfile = f"scan_X{x:.1f}_Y{y:.1f}_Z{z:.1f}.csv"
    start = float(args["start"])
    stop = float(args["stop"])
    decades = int(args.get("decades", 1))  # Number of log-splits
    host = "192.168.0.181"

    # Generate the split frequencies (log scale)
    freqs = np.logspace(np.log10(start), np.log10(stop), decades + 1)

    for i in range(decades):
        split_start = freqs[i]
        split_stop = freqs[i + 1]
        #print(f"{split_start} to {split_stop}")
        runscan(host, split_start, split_stop, folder, outfile)

    # Optional pause after sampling
    #time.sleep(1)  # placeholder for custom logic


def generate_raster(x_min, x_max, y_min, y_max, step):
    """
    Generate coordinates in a serpentine (boustruphedon) raster scan.
    Minimizes travel by reversing x direction every other row.
    """
    y = y_min
    row_index = 0
    while y <= y_max + 1e-6:
        if row_index % 2 == 0:
            # left → right
            x = x_min
            while x <= x_max + 1e-6:
                yield x, y
                x += step
        else:
            print("")
            # right → left
            x = x_max
            while x >= x_min - 1e-6:
                yield x, y
                x -= step

        y += step
        row_index += 1



# ------------------ MACHINE LIMITS ------------------
X_LIMIT = 360
Y_LIMIT = 388
Z_LIMIT = 410

def clamp(val, min_v, max_v):
    return max(min_v, min(val, max_v))

def fmt_si(f):
    if f >= 1e9:
        return f"{f/1e9:.3g}G"
    elif f >= 1e6:
        return f"{f/1e6:.3g}M"
    elif f >= 1e3:
        return f"{f/1e3:.3g}k"
    else:
        return f"{f:.3g}"

def runscan(host, start, stop, folder, outfile):

    # Connect to the instrument via VXI-11
    try:
        instr = vxi11.Instrument(host)
        instr.timeout = 5000  # milliseconds
    except Exception as e:
        print(f"Failed to connect to {host}: {e}")
        sys.exit(1)

    # Reset / clear errors
    instr.write("*CLS")
    instr.write("FORM ASC")

    # Sweep configuration
    instr.write(f"FREQ:STAR {start}")
    instr.write(f"FREQ:STOP {stop}")
    #instr.write(f"BAND:RES {rbw}")    # pointless
    #instr.write(f"SWE:POIN {points}") # pointless

    # Single sweep
    instr.write("INIT:CONT OFF")
    instr.write("INIT; *WAI")

    # Query trace 1
    data = instr.ask("TRAC? TRACE1")
    values = [float(x) for x in data.split(",")]

    # Generate frequency axis
    step = (stop - start) / (len(values) - 1)

    if outfile:
        # Use folder if provided
        filename = outfile
        if folder:
            os.makedirs(folder, exist_ok=True)
            filename = os.path.join(folder, os.path.basename(outfile))

        with open(filename, "a") as f:
            # Do NOT write the title linef.write("Frequency_Hz,Power_dBm\n")
            for i, pwr in enumerate(values):
                freq = start + i * step
                f.write(f"{freq:.2f},{pwr:.2f}\n")

        #print(f"Sweep saved to {filename}")

    else:
        print("# Frequency_Hz, Power_dBm")
        for i, pwr in enumerate(values):
            freq = start + i * step
            print(f"{freq:.2f}, {pwr:.2f}")

# ------------------ MAIN ------------------

def main():

    # 1 Parse arguments first
    args = parse_args()  # now args exists

    # 2️ Create a unique output folder using args and timestamp
    from datetime import datetime
    import os
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f"{timestamp}_X{int(args['xstart'])}-{int(args['xend'])}_Y{int(args['ystart'])}-{int(args['yend'])}_Z{int(args['z'])}_S{int(args['step'])}"
    os.makedirs(output_dir, exist_ok=True)

    # 3️ Dump CLI parameters into a file
    params_file = os.path.join(output_dir, "params.txt")
    with open(params_file, "w") as f:
        for key, val in args.items():
            f.write(f"{key} = {val}\n")

    x_min = clamp(args["xstart"], 0, X_LIMIT)
    x_max = clamp(args["xend"], 0, X_LIMIT)
    y_min = clamp(args["ystart"], 0, Y_LIMIT)
    y_max = clamp(args["yend"], 0, Y_LIMIT)
    step = args["step"]
    z = clamp(args["z"], 0, Z_LIMIT)

    print(f"Starting spiral raster from ({x_min},{y_min}) → ({x_max},{y_max}) step {step}")

    start = float(args["start"])
    stop = float(args["stop"])
    decades = int(args.get("decades", 1))  # Number of log-splits

    # Generate the split frequencies (log scale)
    freqs = np.logspace(np.log10(start), np.log10(stop), decades + 1)

    # Print all sweeps in one line
    sweep_strs = [f"{fmt_si(freqs[i])}→{fmt_si(freqs[i+1])}" for i in range(decades)]
    print(f"Frequency sweeps: 1183 points per block, frequencies: {'; '.join(sweep_strs)}")



    print("Sending moves to printer...")

    for x, y in generate_raster(x_min, x_max, y_min, y_max, step):
        move_and_wait(x, y, z, output_dir, args, FEEDRATE)



if __name__ == "__main__":
    main()



