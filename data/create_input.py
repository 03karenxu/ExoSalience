# Copyright 2018 The Exoplanet ML Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

r"""Script to preprocesses data from the Kepler space telescope.

This script produces training, validation and test sets of labeled Kepler
Threshold Crossing Events (TCEs). A TCE is a detected periodic event on a
particular Kepler target star that may or may not be a transiting planet. Each
TCE in the output contains local and global views of its light curve; auxiliary
features such as period and duration; and a label indicating whether the TCE is
consistent with being a transiting planet. The data sets produced by this script
can be used to train and evaluate models that classify Kepler TCEs.

The input TCEs and their associated labels are specified by the DR24 TCE Table,
which can be downloaded in CSV format from the NASA Exoplanet Archive at:

  https://exoplanetarchive.ipac.caltech.edu/cgi-bin/TblView/nph-tblView?app=ExoTbls&config=q1_q17_dr24_tce

The downloaded CSV file should contain at least the following column names:
  rowid: Integer ID of the row in the TCE table.
  kepid: Kepler ID of the target star.
  tce_plnt_num: TCE number within the target star.
  tce_period: Orbital period of the detected event, in days.
  tce_time0bk: The time corresponding to the center of the first detected
      transit in Barycentric Julian Day (BJD) minus a constant offset of
      2,454,833.0 days.
  tce_duration: Duration of the detected transit, in hours.
  av_training_set: Autovetter training set label; one of PC (planet candidate),
      AFP (astrophysical false positive), NTP (non-transiting phenomenon),
      UNK (unknown).

The Kepler light curves can be downloaded from the Mikulski Archive for Space
Telescopes (MAST) at:

  http://archive.stsci.edu/pub/kepler/lightcurves.

The Kepler data is assumed to reside in a directory with the same structure as
the MAST archive. Specifically, the file names for a particular Kepler target
star should have the following format:

    .../${kep_id:0:4}/${kep_id}/kplr${kep_id}-${quarter_prefix}_${type}.fits,

where:
  kep_id is the Kepler id left-padded with zeros to length 9;
  quarter_prefix is the file name quarter prefix;
  type is one of "llc" (long cadence light curve) or "slc" (short cadence light
    curve).

The output TFRecord file contains one serialized tensorflow.train.Example
protocol buffer for each TCE in the input CSV file. Each Example contains the
following light curve representations:
  global_view: Vector of length 2001; the Global View of the TCE.
  local_view: Vector of length 201; the Local View of the TCE.

In addition, each Example contains the value of each column in the input TCE CSV
file. Some of these features may be useful as auxiliary features to the model.
The columns include:
  rowid: Integer ID of the row in the TCE table.
  kepid: Kepler ID of the target star.
  tce_plnt_num: TCE number within the target star.
  av_training_set: Autovetter training set label.
  tce_period: Orbital period of the detected event, in days.
  ...
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import preprocess

class FLAGS:
  input_tce_csv_file = "../dr24-tce.csv"

  kepler_data_dir = "../tmp"

  output_dir = "real_output/"

  num_train_shards = 1
  num_worker_processes = 1

# label column name and allowed values
_LABEL_COLUMN = "av_training_set"
_ALLOWED_LABELS = {"PC", "AFP", "NTP"}

def _process_tce(tce):
  """Processes the light curve for a Kepler TCE and returns an Example proto.

  Args:
    tce: Row of the input TCE table.

  Returns:
    dict containing global view, local view, label, kepid, tabular data
  """
  all_time, all_flux = preprocess.read_light_curve(tce.kepid, FLAGS.kepler_data_dir)
  time, flux = preprocess.process_light_curve(all_time, all_flux)
  return preprocess.generate_dict_for_tce(time, flux, tce)


def _process_set(tce_table, file_name):
  """Processes a single set (test or train)
  Args:
    tce_table: A Pandas DateFrame containing the TCEs in the shard.
    file_name: The output .npz record
  """

  # containers
  global_views = []
  local_views = []
  kepid = []
  label = []
  tabular = []

  for _, tce in tce_table.iterrows():
    result = _process_tce(tce)
    
    if result is not None:
      global_views.append(result["global_view"])
      local_views.append(result["local_view"])
      kepid.append(result["kepid"])
      label.append(result["label"])
      tabular.append(result["tabular"])
      
  # convert to stacked np array
  global_views_arr = np.stack(global_views)
  local_views_arr = np.stack(local_views)

  # map string labels to ints
  label_map = {name: i for i, name in enumerate(sorted(list(_ALLOWED_LABELS)))}
  labels_int = np.array([label_map[l] for l in label], dtype=float)

  # save data to npz record
  output_path = file_name + ".npz"
  np.savez_compressed(
    output_path,
    global_view=global_views_arr,
    local_view=local_views_arr,
    kepid=kepid,
    label=labels_int
  )


def main():

  # read TCE CSV file
  tce_table = pd.read_csv(
      FLAGS.input_tce_csv_file, index_col="loc_rowid", comment="#")
  tce_table["tce_duration"] /= 24  # Convert hours to days.

  # filter TCE table for only allowed labels
  allowed_tces = tce_table[_LABEL_COLUMN].apply(lambda l: l in _ALLOWED_LABELS)
  tce_table = tce_table[allowed_tces]
  num_tces = len(tce_table)

  # randomly shuffle
  np.random.seed(123)
  tce_table = tce_table.iloc[np.random.permutation(num_tces)]

  # partition the table
  train_cutoff = int(0.70 * num_tces)
  train_tces = tce_table[0:train_cutoff]
  test_tces = tce_table[train_cutoff:]

  # process train
  train_file = os.path.join(FLAGS.output_dir, "train.npz")
  _process_set(train_tces, train_file[:-4])

  # process test
  test_file = os.path.join(FLAGS.output_dir, "test.npz")
  _process_set(test_tces, test_file[:-4])


if __name__ == "__main__":
  main()