#!/usr/bin/env python3
"""MiniSLAM - Minimal Visual SLAM implementation."""

import os
import sys
import time
import argparse

import cv2
import yaml
import numpy as np

from minislam.camera import Camera
from minislam.dataset import ImageLoader, VideoLoader
from minislam.display import Display
from minislam.odometry import VisualOdometry

np.set_printoptions(suppress=True)

__version__ = "0.1.0"


# ANSI color codes - Neon green theme
class Colors:
  # Neon green (bright green)
  NEON = "\033[38;5;46m"
  # Darker green for accents
  GREEN = "\033[38;5;40m"
  # Dim green for subtle text
  DIM_GREEN = "\033[38;5;34m"
  # Standard colors
  YELLOW = "\033[93m"
  RED = "\033[91m"
  WHITE = "\033[97m"
  BOLD = "\033[1m"
  DIM = "\033[2m"
  RESET = "\033[0m"


def print_banner():
  """Print the MiniSLAM banner."""
  banner = f"""
{Colors.NEON}{Colors.BOLD}
  ███╗   ███╗██╗███╗   ██╗██╗███████╗██╗      █████╗ ███╗   ███╗
  ████╗ ████║██║████╗  ██║██║██╔════╝██║     ██╔══██╗████╗ ████║
  ██╔████╔██║██║██╔██╗ ██║██║███████╗██║     ███████║██╔████╔██║
  ██║╚██╔╝██║██║██║╚██╗██║██║╚════██║██║     ██╔══██║██║╚██╔╝██║
  ██║ ╚═╝ ██║██║██║ ╚████║██║███████║███████╗██║  ██║██║ ╚═╝ ██║
  ╚═╝     ╚═╝╚═╝╚═╝  ╚═══╝╚═╝╚══════╝╚══════╝╚═╝  ╚═╝╚═╝     ╚═╝
{Colors.RESET}{Colors.DIM_GREEN}  Minimal Visual SLAM • v{__version__}{Colors.RESET}
"""
  print(banner)


def print_info(msg: str):
  print(f"{Colors.DIM_GREEN}[INFO]{Colors.RESET} {msg}")


def print_success(msg: str):
  print(f"{Colors.NEON}[OK]{Colors.RESET} {msg}")


def print_warn(msg: str):
  print(f"{Colors.YELLOW}[WARN]{Colors.RESET} {msg}")


def print_error(msg: str):
  print(f"{Colors.RED}[ERROR]{Colors.RESET} {msg}")


def print_section(title: str):
  print(f"\n{Colors.BOLD}{Colors.NEON}▸ {title}{Colors.RESET}")


def load_config(path: str) -> dict:
  """Load configuration file."""
  if not os.path.exists(path):
    return {"datasets": {}}
  with open(path, "r") as fp:
    return yaml.safe_load(fp)


def list_datasets(config_path: str):
  """List all available datasets in config."""
  cfg = load_config(config_path)
  datasets = cfg.get("datasets", {})

  print_section("Available Datasets")

  if not datasets:
    print_warn("No datasets configured. Add datasets to config.yaml")
    return

  for name, info in datasets.items():
    path = info.get("path", "N/A")
    exists = os.path.exists(path)
    status = f"{Colors.NEON}✓{Colors.RESET}" if exists else f"{Colors.RED}✗{Colors.RESET}"
    dims = f"{info.get('w', '?')}x{info.get('h', '?')}"
    print(f"  {status} {Colors.BOLD}{Colors.WHITE}{name}{Colors.RESET}")
    print(f"      Path: {Colors.DIM_GREEN}{path}{Colors.RESET}")
    print(f"      Resolution: {dims}, fx={info.get('fx', '?')}, fy={info.get('fy', '?')}")


def run_slam(
  dataset: str | None = None,
  path: str | None = None,
  config_path: str = "config.yaml",
  width: int | None = None,
  height: int | None = None,
  fx: float | None = None,
  fy: float | None = None,
  cx: float | None = None,
  cy: float | None = None,
  headless: bool = False,
):
  """Run the SLAM pipeline."""

  # Determine source and camera params
  if path:
    # Direct path mode
    if not os.path.exists(path):
      print_error(f"Path not found: {path}")
      sys.exit(1)

    # Require camera params for direct path
    if not all([width, height, fx, fy]):
      print_error("When using --path, you must specify --width, --height, --fx, --fy")
      print_info("Or use --dataset to load from config.yaml")
      sys.exit(1)

    # Type narrowing - we know these are not None after the check above
    assert width is not None and height is not None and fx is not None and fy is not None

    cx = cx if cx is not None else width / 2
    cy = cy if cy is not None else height / 2
    source_path = path

  else:
    # Dataset mode
    cfg = load_config(config_path)
    datasets = cfg.get("datasets", {})

    if not dataset:
      dataset = "test2"  # default

    if dataset not in datasets:
      print_error(f"Dataset '{dataset}' not found in {config_path}")
      list_datasets(config_path)
      sys.exit(1)

    ds = datasets[dataset]
    source_path = ds["path"]
    width = width if width is not None else int(ds["w"])
    height = height if height is not None else int(ds["h"])
    fx = fx if fx is not None else float(ds["fx"])
    fy = fy if fy is not None else float(ds["fy"])
    cx = cx if cx is not None else float(ds.get("cx", width / 2))
    cy = cy if cy is not None else float(ds.get("cy", height / 2))

  # At this point, all values are guaranteed to be set
  assert width is not None and height is not None
  assert fx is not None and fy is not None and cx is not None and cy is not None

  # Validate source
  if not os.path.exists(source_path):
    print_error(f"Source path not found: {source_path}")
    sys.exit(1)

  # Print configuration
  print_section("Configuration")
  print(f"  Source:     {Colors.WHITE}{source_path}{Colors.RESET}")
  print(f"  Resolution: {Colors.DIM_GREEN}{width} x {height}{Colors.RESET}")
  print(f"  Focal:      {Colors.DIM_GREEN}fx={fx:.2f}, fy={fy:.2f}{Colors.RESET}")
  print(f"  Principal:  {Colors.DIM_GREEN}cx={cx:.2f}, cy={cy:.2f}{Colors.RESET}")

  # Create camera and loader
  camera = Camera(width, height, fx, fy, cx, cy)

  if os.path.isdir(source_path):
    loader = ImageLoader(source_path)
    print(f"  Type:       Image sequence")
  else:
    loader = VideoLoader(source_path)
    print(f"  Type:       Video file")

  # Run SLAM
  print_section("Running SLAM")
  print(f"  {Colors.DIM}Press ESC to stop{Colors.RESET}\n")

  display = Display(width, height)
  vo = VisualOdometry(camera)

  start_time = time.time()
  frame_count = 0

  try:
    for i, img in enumerate(loader):
      img = cv2.resize(img, (width, height))
      vo.process_frame(img, i)
      frame_count = i + 1

      key = display.show(vo)
      if key == 27:  # ESC
        print_warn("Interrupted by user")
        break

      # Print progress every 100 frames
      if frame_count % 100 == 0:
        elapsed = time.time() - start_time
        fps = frame_count / elapsed if elapsed > 0 else 0
        kf = vo.num_keyframes
        lc = len(vo.loop_closures)
        print(f"\r  Frame {frame_count:5d} | {fps:5.1f} FPS | KF: {kf:3d} | Loops: {lc:2d}", end="", flush=True)

  except KeyboardInterrupt:
    print_warn("\nInterrupted by user")

  finally:
    display.close()

  # Summary
  elapsed = time.time() - start_time
  fps = frame_count / elapsed if elapsed > 0 else 0

  print_section("Summary")
  print(f"  Frames processed: {frame_count}")
  print(f"  Time elapsed:     {elapsed:.2f}s")
  print(f"  Average FPS:      {fps:.1f}")
  print(f"  Keyframes:        {vo.num_keyframes}")
  print(f"  Loop closures:    {len(vo.loop_closures)}")
  print()
  print_success("Done!")


def main():
  parser = argparse.ArgumentParser(
    prog="minislam",
    description="Minimal Visual SLAM - Real-time monocular visual odometry and mapping",
    formatter_class=argparse.RawDescriptionHelpFormatter,
    epilog=f"""
{Colors.BOLD}Examples:{Colors.RESET}
  %(prog)s                          Run with default dataset
  %(prog)s -d kitti                 Run with 'kitti' dataset from config
  %(prog)s -l                       List available datasets
  %(prog)s -p video.mp4 -W 1280 -H 720 --fx 700 --fy 700
                                    Run on video with custom camera params

{Colors.BOLD}Controls:{Colors.RESET}
  ESC                               Quit
  Left-drag on 3D view              Rotate camera
  Scroll on 3D view                 Zoom in/out
  Right-click on 3D view            Reset to auto-follow
""",
  )

  # Mode selection
  mode_group = parser.add_argument_group("Mode")
  mode_group.add_argument(
    "-l",
    "--list",
    action="store_true",
    help="List available datasets and exit",
  )
  mode_group.add_argument(
    "-V",
    "--version",
    action="version",
    version=f"%(prog)s {__version__}",
  )

  # Source selection
  source_group = parser.add_argument_group("Source")
  source_group.add_argument(
    "-d",
    "--dataset",
    metavar="NAME",
    type=str,
    help="Dataset name from config.yaml (default: test2)",
  )
  source_group.add_argument(
    "-p",
    "--path",
    metavar="PATH",
    type=str,
    help="Direct path to video file or image directory",
  )
  source_group.add_argument(
    "-c",
    "--config",
    metavar="FILE",
    type=str,
    default="config.yaml",
    help="Config file path (default: config.yaml)",
  )

  # Camera parameters
  cam_group = parser.add_argument_group("Camera parameters (required with --path)")
  cam_group.add_argument("-W", "--width", type=int, metavar="PX", help="Image width")
  cam_group.add_argument("-H", "--height", type=int, metavar="PX", help="Image height")
  cam_group.add_argument("--fx", type=float, metavar="PX", help="Focal length X")
  cam_group.add_argument("--fy", type=float, metavar="PX", help="Focal length Y")
  cam_group.add_argument("--cx", type=float, metavar="PX", help="Principal point X (default: width/2)")
  cam_group.add_argument("--cy", type=float, metavar="PX", help="Principal point Y (default: height/2)")

  args = parser.parse_args()

  # Print banner
  print_banner()

  # Handle modes
  if args.list:
    list_datasets(args.config)
    sys.exit(0)

  # Run SLAM
  run_slam(
    dataset=args.dataset,
    path=args.path,
    config_path=args.config,
    width=args.width,
    height=args.height,
    fx=args.fx,
    fy=args.fy,
    cx=args.cx,
    cy=args.cy,
  )


if __name__ == "__main__":
  main()
