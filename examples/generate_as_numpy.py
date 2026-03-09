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
import matplotlib.pyplot
import numpy

# Generate a single ECG sample
ecg_data = deepfakeecg.generate_as_numpy()

# Plot the first lead (Lead I)
matplotlib.pyplot.figure(figsize=(15, 3))
matplotlib.pyplot.plot(ecg_data[:, 0], label="Lead I")
matplotlib.pyplot.plot(ecg_data[:, 1], label="Lead II")
matplotlib.pyplot.plot(ecg_data[:, 2], label="V1")
matplotlib.pyplot.plot(ecg_data[:, 3], label="V2")
matplotlib.pyplot.plot(ecg_data[:, 4], label="V3")
matplotlib.pyplot.plot(ecg_data[:, 5], label="V4")
matplotlib.pyplot.plot(ecg_data[:, 6], label="V5")
matplotlib.pyplot.plot(ecg_data[:, 7], label="V6")
matplotlib.pyplot.legend()
matplotlib.pyplot.title("Generated ECG")
matplotlib.pyplot.xlabel("Time [s]")
matplotlib.pyplot.ylabel("Amplitude [μV]")
matplotlib.pyplot.grid(True)

# Print shape and basic stats
print(f"ECG data shape: {ecg_data.shape}")
print(f"Value range: [{numpy.min(ecg_data)}, {numpy.max(ecg_data)}]")
print("\nLead order: [I, II, V1, V2, V3, V4, V5, V6]")

matplotlib.pyplot.show()
