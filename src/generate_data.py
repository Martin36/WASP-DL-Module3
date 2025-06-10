# Generates Sokoban images using PDDL gym 

import argparse
import os
import shutil
import pddlgym
import imageio
import numpy as np
from tqdm import tqdm

arg_parser = argparse.ArgumentParser(description="Generate Sokoban images using PDDL gym")
arg_parser.add_argument("--output_folder", type=str, required=True, help="Directory to save the generated images")
arg_parser.add_argument("--num_images", type=int, default=100, help="Number of images to generate")
arg_parser.add_argument("--actions_per_env", type=int, default=10, help="Number of actions to take before creating a new environment")
args = arg_parser.parse_args()

shutil.rmtree(args.output_folder, ignore_errors=True)
os.makedirs(args.output_folder, exist_ok=True)

gen_images_count = 0
pbar = tqdm(total=args.num_images, desc="Generating images", unit="image")
env = pddlgym.make("PDDLEnvSokoban-v0")
while gen_images_count < args.num_images:
  obs, debug_info = env.reset()
  img = env.render()
  for i in range(args.actions_per_env):
    assert isinstance(img, np.ndarray), "Rendered image should be a numpy array"
    imageio.imsave(f"{args.output_folder}/frame_{gen_images_count:05d}.png", img)
    gen_images_count += 1
    pbar.update(1)
    if gen_images_count >= args.num_images:
      break
    action = env.action_space.sample(obs)
    obs, reward, done, truncated, debug_info = env.step(action)
    img = env.render()
