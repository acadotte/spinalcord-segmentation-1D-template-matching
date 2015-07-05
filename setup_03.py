"""
A general setup file to prepare the system for running abvi.
"""

import sys
import os

sys.path.append("packages")
sys.path.append("cpp/src/spline")

labels = {"cord": 1,
          "spine": 2, #This is the CSF
          "PMJ": 3, # Ponto-Medullary Junction
          "spine end": 4,
          "dorsal rootlets": 5,
          "ventral rootlets": 6,
          "hole angle range": 7,
          "hole region": 8,
          "basion": 9,
          "C2-CB": 10, 
          "C3-RB": 11,
          "C3-CB": 12,
          "C4-RB": 13,
          "C4-CB": 14,
          "C5-RB": 15,
          "C5-CB": 16,
          "C6-RB": 17,
          "C6-CB": 18,
          "C7-RB": 19,
          "C7-CB": 20,

          "C2": 21,
          "C3": 22,
          "C4": 23,
          "C5": 24,
          "C6": 25,
          "C7": 26,
          "T1": 27,
          "T2": 28,
          "T3": 29,
          "T4": 30,
          "T5": 31,
          "T6": 32,
          "T7": 33,
          "T8": 34,
          "T9": 35,
          "T10": 36,
          "T11": 37,
          "T12": 38,
          }

# Global definitions
data_path = os.path.abspath(os.path.join(os.path.dirname(__file__),
                                         '../SpineExtraction/data2/nrrd/'))
image_path = os.path.abspath(os.path.join(os.path.dirname(__file__),
                                         '../SpineExtraction/data2/images/'))
dicom_path = os.path.abspath(os.path.join(os.path.dirname(__file__),
                                         '../SpineExtraction/data2/data/'))
data_file_extension = 'nrrd'
