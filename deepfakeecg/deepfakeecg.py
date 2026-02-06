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
# Generator Library
# Copyright (C) 2021-2025 by Vajira Thambawita
# Copyright (C) 2021-2025 by Turtle <erencemayez@gmail.com>
# Copyright (C) 2025 by Thomas Dreibholz
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
# * Vajira Thambawita <vlbthambawita@gmail.com>
# * Turtle <erencemayez@gmail.com>
# * Thomas Dreibholz <dreibh@simula.no>

import ecg_plot
import matplotlib
import matplotlib.backends.backend_pdf
import neurokit2
import numpy
import os
import pathlib
import sys
import torch
import tqdm
import typing

from . import Generator


# ------ Constants ----------------------------------------
ECG_SAMPLING_RATE             = 500   # in Hz
ECG_DEFAULT_LENGTH_IN_SECONDS = 10
ECG_DEFAULT_SCALE_FACTOR      = 6000

# ------ ECG types ----------------------------------------
DATA_ECG8           = 8
DATA_ECG12          = 12

# ------ Output formats -----------------------------------
OUTPUT_NUMPY        = 1
OUTPUT_TENSOR       = 2
OUTPUT_ASC          = 10
OUTPUT_CSV          = 11
OUTPUT_PDF          = 20
OUTPUT_PDF_ANALYSIS = 21

ECG_LEADS = {
   'I':   [  1, 'Lead I',   DATA_ECG8  ],
   'II':  [  2, 'Lead II',  DATA_ECG8  ],
   'V1':  [  3, 'V1',       DATA_ECG8  ],
   'V2':  [  4, 'V2',       DATA_ECG8  ],
   'V3':  [  5, 'V3',       DATA_ECG8  ],
   'V4':  [  6, 'V4',       DATA_ECG8  ],
   'V5':  [  7, 'V5',       DATA_ECG8  ],
   'V6':  [  8, 'V6',       DATA_ECG8  ],
   'III': [  9, 'Lead III', DATA_ECG12 ],
   'aVL': [ 10, 'aVL',      DATA_ECG12 ],
   'aVR': [ 11, 'aVR',      DATA_ECG12 ],
   'aVF': [ 12, 'aVF',      DATA_ECG12 ]
}


# ###### Produce ECG ASCII file from Tensor #################################
def dataToASCII(ecgResult, outputFileName):
   # Convert to NumPy, and remove the Timestamp column (0):
   data = ecgResult.detach().cpu().numpy()[1:]
   numpy.savetxt(outputFileName, data, fmt = '%i')


# ###### Produce ECG CSV file from Tensor ###################################
def dataToCSV(ecgResult, ecgType, outputFileName):

   data = ecgResult.detach().cpu().numpy()

   if ecgType == DATA_ECG8:
      header = 'Timestamp,LeadI,LeadII,V1,V2,V3,V4,V5,V6'
   elif ecgType == DATA_ECG12:
      header = 'Timestamp,LeadI,LeadII,V1,V2,V3,V4,V5,V6,LeadIII,aVL,aVR,aVF'
   else:
      raise Exception('Invalid ECG type!')

   numpy.savetxt(outputFileName, data,
                 header    = header,
                 comments  = '',
                 delimiter = ',',
                 fmt       = '%i')


# ###### Produce ECG PDF file from Tensor ###################################
def dataToPDF(ecgResult, ecgType, outputLeads, outputFileName,
              outputFormat = OUTPUT_PDF, idNumber = None):

   # 1. Convert to NumPy
   # 2. Remove the Timestamp column (0)
   # 3. Convert from µV to mV
   data = ecgResult.t().detach().cpu().numpy()[1:] / 1000
   # print(data)

   if idNumber != None:
      titleExtension = ' — ID ' + str(idNumber)
   else:
      titleExtension = ''
   titleExtension = titleExtension + ' — 25 mm/sec, 1 mV/10 mm'

   pdf = matplotlib.backends.backend_pdf.PdfPages(outputFileName)

   # ------ ECG-12 -------------------------------------------------------
   if ecgType == DATA_ECG12:
      ecg_plot.plot(data,
                    title       = 'ECG-12' + titleExtension,
                    sample_rate = ECG_SAMPLING_RATE,
                    lead_index  = [ 'I', 'II', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6', 'III', 'aVR', 'aVL', 'aVF' ],
                    lead_order  = [0, 1, 8, 9, 10, 11, 2, 3, 4, 5, 6, 7],
                    show_grid   = True)
   # ------ ECG-8 --------------------------------------------------------
   else:
      ecg_plot.plot(data,
                    title       = 'ECG-8' + titleExtension,
                    sample_rate = ECG_SAMPLING_RATE,
                    lead_index  = [ 'I', 'II', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6' ],
                    lead_order  = [0, 1, 2, 3, 4, 5, 6, 7],
                    show_grid   = True)

   pdf.savefig(matplotlib.pyplot.gcf())
   matplotlib.pyplot.close()

   if outputFormat == OUTPUT_PDF_ANALYSIS:
      leadI = data[0]

      signals, info = neurokit2.ecg_process(leadI, sampling_rate = ECG_SAMPLING_RATE)
      neurokit2.ecg_plot(signals, info)

      # DIN A4 landscape: w=11.7, h=8.27
      w = 508/25.4   # mm to inch
      h = 122/25.4   # mm to inch
      matplotlib.pyplot.gcf().set_size_inches(w, h, forward=True)
      pdf.savefig(matplotlib.pyplot.gcf())
      matplotlib.pyplot.close()

   pdf.close()


# ###### Generate Deepfake ECGs #############################################
def generateDeepfakeECGs(numberOfECGs:       int = 1,
                         ecgType:            int = DATA_ECG8,
                         ecgLengthInSeconds: int = ECG_DEFAULT_LENGTH_IN_SECONDS,
                         ecgScaleFactor:     int = ECG_DEFAULT_SCALE_FACTOR,
                         outputFormat:       int = OUTPUT_NUMPY,
                         outputFilePattern:  typing.Union[str, pathlib.Path] = None,
                         outputStartID:      int = 0,
                         outputLeads:        list = [ 'I' ],
                         showProgress:       bool = True,
                         runOnDevice:        typing.Literal['cpu', 'cuda'] = 'cuda' if torch.cuda.is_available() else 'cpu'):
   """Generate ECG waveforms using deepfakeecg model, with configurable
      data type (8-lead or 12-lead ECG) and output type (numpy, file).

   Args:
      numberOfECGs (int): The number of ECGs to generate
      ecgLengthInSeconds (int): The ECG length in seconds
      outputFormat (int): The format of the output
         OUTPUT_NUMPY: list of NumPy numpy.ndarray objects
         OUTPUT_TENSOR: list of PyTorch torch.Tensor objects
         OUTPUT_ASC: text, as in the original code
         OUTPUT_CSV: CSV, with additional column for time stamp in milliseconds
         OUTPUT_PDF: PDF, with plot of the output
         OUTPUT_PDF_ANALYSIS: PDF, with plot of the output and NeuroKit2 analysis
      outputFilePattern: Pattern for naming output files, with format() placeholder 'number', e.g. 'ecg-{number:06d}.csv'
      outputStartID: Start ID for file numbering
      outputLeads: List of output leads for PDF plotting (from: [ 'I', 'II', 'III', 'aVL', 'aVR', 'aVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6' ])
      runOnDevice (str): Device to run generation on ('cpu' or 'cuda')

   Returns:
      In case of outputFormat OUTPUT_NUMPY or OUTPUT_TENSOR:
         List of arrays of shape (ecgLength, n) containing the ECG data.
         For ECG type DATA_ECG8:
            numpy.ndarray/torch.Tensor: [I, II, V1, V2, V3, V4, V5, V6]
         For ECG type DATA_ECG12:
            numpy.ndarray/torch.Tensor: [I, II, V1, V2, V3, V4, V5, V6, III, aVL, aVR, aVF]
   """

   # ====== Initialise generator ============================================
   root_dir = pathlib.Path(__file__).parent
   device = torch.device(runOnDevice)

   generator = Generator()
   checkpoint = torch.load(
      os.path.join(root_dir, 'checkpoints/g_stat.pt'),
      map_location = device,
      weights_only = True
   )
   generator.load_state_dict(checkpoint['stat_dict'])
   generator.to(device)
   generator.eval()

   # ====== Make milliseconds time stamp tensor =============================
   ecgLengthInSamples = ecgLengthInSeconds * ECG_SAMPLING_RATE
   timeStamp = torch.arange(0, 1000000 * ecgLengthInSeconds,
                            1000000 / ECG_SAMPLING_RATE,
                            dtype = torch.int32, device = device)
   # Timestamp shape is [ ecgLengthInSamples ]
   timeStamp = torch.t(timeStamp.reshape(1, ecgLengthInSamples))
   # Now, shape is [ ecgLengthInSamples, 1 ]

   # ====== Generate ECGs ===================================================
   results  = [ ]

   ecgRange = range(outputStartID, outputStartID + numberOfECGs)
   if showProgress:
      ecgRange = tqdm.tqdm(ecgRange)
   for i in ecgRange:
      # ------ Create random noise  -----------------------------------------
      noise = torch.empty(1, 8, ecgLengthInSamples, device = device).uniform_(-1, 1)

      # ------ Generate ECG -------------------------------------------------
      generatedECG = generator(noise)
      # Output shape is [1, 8, ecgLengthInSamples].

      # ------ Rescale and convert to integer -------------------------------
      generatedECG = generatedECG * ecgScaleFactor
      # generatedECG = generatedECG.int()
      generatedECG = torch.transpose(generatedECG.squeeze(), 0, 1)
      # Now, shape is [ecgLengthInSamples, 8].

      # ------ EGC12 computations -------------------------------------------
      if ecgType == DATA_ECG12:
         # Details and formulae:
         # https://ecgwaves.com/topic/ekg-ecg-leads-electrodes-systems-limb-chest-precordial/

         leadI   = generatedECG[:,0]
         leadII  = generatedECG[:,1]

         # Computations:
         # Lead III = Lead II - Lead I
         # aVL      = (Lead I - Lead III) / 2
         # aVR      = -(Lead I + Lead II) / 2
         # aVF      = (Lead II + Lead III) / 2
         leadIII = leadII - leadI
         aVL     = (leadI - leadIII) / 2
         aVR     = -(leadI + leadII) / 2
         aVF     = (leadII + leadIII) / 2
         # Shape is [ ecgLengthInSamples ]

         # Reshape to [ ecgLengthInSamples, 1 ] and combine with generatedECG:
         generatedECG = torch.cat( ( generatedECG,
                                     leadIII.reshape(ecgLengthInSamples, 1),
                                     aVL.reshape(ecgLengthInSamples, 1),
                                     aVR.reshape(ecgLengthInSamples, 1),
                                     aVF.reshape(ecgLengthInSamples, 1)
                                   ) , 1 )

      # ------ Add time stamp for CSV output --------------------------------
      # Combine time stamp with generated ECG samples.
      # Now, shape is [ecgLengthInSamples, 1+8].
      generatedECG = torch.cat( ( timeStamp, generatedECG ), 1 )
      # print(generatedECG[:,0])

      # ------ Write output file --------------------------------------------
      if outputFormat in [ OUTPUT_ASC, OUTPUT_CSV, OUTPUT_PDF, OUTPUT_PDF_ANALYSIS ]:
        outputFileName = outputFilePattern.format(number = i)

        # ------ ASCII text -------------------------------------------------
        if outputFormat == OUTPUT_ASC:
           dataToASCII(generatedECG, outputFileName)

        # ------ CSV --------------------------------------------------------
        elif outputFormat == OUTPUT_CSV:
           dataToCSV(generatedECG, ecgType, outputFileName)

        # ------ PDF --------------------------------------------------------
        elif ( (outputFormat == OUTPUT_PDF) or (outputFormat == OUTPUT_PDF_ANALYSIS) ):
           dataToPDF(generatedECG, ecgType, outputLeads, outputFileName, outputFormat)

      # ------ Collect data in array ----------------------------------------
      elif outputFormat == OUTPUT_TENSOR:
         results.append(generatedECG)

      # ------ Collect data in array ----------------------------------------
      elif outputFormat == OUTPUT_NUMPY:
         results.append(generatedECG.detach().cpu().numpy())

   return results


# ###### Generate Deepfake ECG as files #####################################
def generate(num_of_sample: int,
             out_dir:       typing.Union[str, pathlib.Path],
             start_id:      int = 0,
             runOnDevice:   typing.Literal['cpu', 'cuda'] = 'cuda' if torch.cuda.is_available() else 'cpu') -> None:
   """Generate multiple 8-lead ECG waveforms and save them as ASCII files

   Args:
      num_of_sample (int): Number of ECG samples to generate
      out_dir (typing.Union[str, pathlib.Path]): Output directory path where files will be saved
      start_id (int): Starting ID for the generated samples
      runOnDevice (typing.Literal['cpu', 'cuda']): Device to run generation on ('cpu' or 'cuda')

   Returns:
      None: Files are saved to the specified output directory with names {start_id}.asc to {start_id + num_of_sample - 1}.asc
      Each file contains ECG data in ASCII format with shape (5000, 8) for leads [I, II, V1, V2, V3, V4, V5, V6]
    """

   generateDeepfakeECGs(num_of_sample,
                        ecgType            = DATA_ECG8,
                        ecgLengthInSeconds = int(5000 / ECG_SAMPLING_RATE),
                        outputFormat       = OUTPUT_ASC,
                        outputFilePattern  = os.path.join(out_dir, '{number}.asc'),
                        outputStartID      = 0)


# ###### Generate Deepfake ECG as NumPy object ##############################
def generate_as_numpy(runOnDevice: typing.Literal['cpu', 'cuda'] = 'cuda' if torch.cuda.is_available() else 'cpu') -> numpy.ndarray:
   """Generate a single 8-lead ECG waveform using deepfakeecg model

   Args:
       runOnDevice (str): Device to run generation on ('cpu' or 'cuda')

   Returns:
       numpy.ndarray: Array of shape (5000, 8) containing the ECG data for leads [I, II, V1, V2, V3, V4, V5, V6]
    """

   results = generateDeepfakeECGs(1,
                                  ecgType            = DATA_ECG8,
                                  ecgLengthInSeconds = int(5000 / ECG_SAMPLING_RATE),
                                  outputFormat       = OUTPUT_NUMPY)
   return results[0]
