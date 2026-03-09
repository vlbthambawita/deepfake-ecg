#!/usr/bin/env python
# -*- coding: utf-8 -*-
# ==========================================================================
#         ____                   __       _          _____ ____ ____
#        |  _ \  ___  ___ _ __  / _| __ _| | _____  | ____/ ___/ ___|
#        | | | |/ _ \/ _ \ '_ \| |_ / _` | |/ / _ \ |  _|| |  | |  _
#        | |_| |  __/  __/ |_) |  _| (_| |   <  __/ | |__| |__| |_| |
#        |____/ \___|\___| .__/|_|  \__,_|_|\_\___| |_____\____\____|
#                        |_|
#
#                       --- Deepfake ECG Generator ---
#                https://github.com/vlbthambawita/deepfake-ecg
# ==========================================================================
#
# Deepfake ECG Example
# Copyright (C) 2025 by Turtle <erencemayez@gmail.com>
# Copyright (C) 2025-2026 by Thomas Dreibholz
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.
#
# Contact:
# * Turtle <erencemayez@gmail.com>
# * Thomas Dreibholz <dreibh@simula.no>

import sys
sys.path.append('..')

import deepfakeecg
import os

# Create output directory if it doesn't exist
output_dir = "generated_ecgs"
os.makedirs(output_dir, exist_ok=True)

# Generate 5 ECG samples starting from ID 0
deepfakeecg.generate(
    num_of_sample = 5,
    out_dir       = output_dir,
    start_id      = 0
)

print(f"Generated 5 ECG samples in {output_dir}")
print("Files generated:")
for file in os.listdir(output_dir)[:5]:
    print(f" - {file}")
