# Generates Sokoban images using PDDL gym

import argparse
import os
import shutil
import pddlgym
import imageio
import numpy as np
from tqdm import tqdm
from pddlgym_planners.fd import FD

arg_parser = argparse.ArgumentParser(description="Generate Sokoban images using PDDL gym")
arg_parser.add_argument("--output_folder", type=str, required=True, help="Directory to save the generated images")
arg_parser.add_argument("--num_images", type=int, default=100, help="Number of images to generate")
arg_parser.add_argument("--actions_per_env", type=int, default=10, help="Number of actions to take before creating a new environment")
# arg_parser.add_argument("--num_domains",type=int,)
args = arg_parser.parse_args()

shutil.rmtree(args.output_folder, ignore_errors=True)
os.makedirs(args.output_folder, exist_ok=True)

problem_index = 0
gen_images_count = 0
# pbar = tqdm(total=args.num_images, desc="Generating images", unit="image")
env = pddlgym.make("PDDLEnvSokoban-v0")
# assert isinstance(env, pddlgym.core.PDDLEnv), "Environment should be an instance of PDDLEnv"

while problem_index < 5: # There are only 5 different Sokoban instances in pddlgym #gen_images_count < args.num_images:
  print(f"Generating images for problem index: {problem_index}")
  env.fix_problem_index(problem_index)
  problem_index += 1
  obs, debug_info = env.reset()
  img = env.render()
  assert isinstance(img, np.ndarray), "Rendered image should be a numpy array"
  imageio.imsave(f"{args.output_folder}/frame_{gen_images_count:05d}.png", img)
  gen_images_count += 1
  planner = FD(alias_flag="--alias lama-first")
  plan = planner(env.domain ,obs, timeout=60)
  print(f"found plan of length: {len(plan)}")
  for act in tqdm(plan):
    #pbar.update(1)
    #if gen_images_count >= args.num_images:
    #  break
    action = env.action_space.sample(obs)
    obs, reward, done, truncated, debug_info = env.step(act)
    img = env.render()
    assert isinstance(img, np.ndarray), "Rendered image should be a numpy array"
    imageio.imsave(f"{args.output_folder}/frame_{gen_images_count:05d}.png", img)
    gen_images_count += 1
