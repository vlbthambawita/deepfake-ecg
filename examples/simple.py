#!/usr/bin/env python3

import deepfakeecg

deepfakeecg.generate(5, ".", start_id=0)  # Generate 5 ECGs to the current folder starting from id=0
