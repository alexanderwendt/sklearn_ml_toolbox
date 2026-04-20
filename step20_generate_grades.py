#!/usr/bin/env python
# -*- coding: utf-8 -*-

# This script generates feature and outcome data from the Student_performance_data.csv file.

"""
License_info: ISC
ISC License

Copyright (c) 2020, Alexander Wendt

Permission to use, copy, modify, and/or distribute this software for any
purpose with or without fee is hereby granted, provided that the above
copyright notice and this permission notice appear in all copies.

THE SOFTWARE IS PROVIDED "AS IS" AND THE AUTHOR DISCLAIMS ALL WARRANTIES
WITH REGARD TO THIS SOFTWARE INCLUDING ALL IMPLIED WARRANTIES OF
MERCHANTABILITY AND FITNESS. IN NO EVENT SHALL THE AUTHOR BE LIABLE FOR
ANY SPECIAL, DIRECT, INDIRECT, OR CONSEQUENTIAL DAMAGES OR ANY DAMAGES
WHATSOEVER RESULTING FROM LOSS OF USE, DATA OR PROFITS, WHETHER IN AN
ACTION OF CONTRACT, NEGLIGENCE OR OTHER TORTIOUS ACTION, ARISING OUT OF
OR IN CONNECTION WITH THE USE OR PERFORMANCE OF THIS SOFTWARE.
"""
import argparse

import pandas as pd
import configparser
import os
import argparse

__author__ = 'Alexander Wendt'
__copyright__ = 'Copyright 2020, Christian Doppler Laboratory for ' \
                'Embedded Machine Learning'
__credits__ = ['']
__license__ = 'ISC'
__version__ = '0.2.0'
__maintainer__ = 'Alexander Wendt'
__email__ = 'alexander.wendt@tuwien.ac.at'
__status__ = 'Experiental'

import argparse

# Configuration
parser = argparse.ArgumentParser(description='Generate features and outcomes for the grades dataset.')
parser.add_argument("-conf", '--config_path', default='samples/grades/config/grades.ini',
                    help='Path to the configuration file')
args = parser.parse_args()
config_file = args.config_path

# Read configuration with interpolation
config = configparser.ConfigParser(interpolation=configparser.ExtendedInterpolation())
config.read(config_file)

source_path = config.get('Paths', 'source_path')
features_out = config.get('Generation', 'features_out')
outcomes_out = config.get('Generation', 'outcomes_out')
source_out = config.get('Generation', 'source_out', fallback=None)

# Get the absolute path of the project root
project_root = os.path.dirname(os.path.abspath(__file__))

# Construct absolute paths
source_csv_path = os.path.join(project_root, source_path)
features_out_path = os.path.join(project_root, features_out)
outcomes_out_path = os.path.join(project_root, outcomes_out)
if source_out:
    source_out_path = os.path.join(project_root, source_out)
else:
    source_out_path = None


# Create output directories if they don't exist
os.makedirs(os.path.dirname(features_out_path), exist_ok=True)
os.makedirs(os.path.dirname(outcomes_out_path), exist_ok=True)
if source_out_path:
    os.makedirs(os.path.dirname(source_out_path), exist_ok=True)

# Load the dataset
df = pd.read_csv(source_csv_path)

# Define feature and outcome columns
outcome_column = config.get('Common', 'class_name')
feature_columns = [col for col in df.columns if col not in ['StudentID', 'GPA', 'GradeClass']]

# Separate features and outcomes
features_df = df[feature_columns].copy()
features_df['id'] = df['StudentID']
features_df.set_index('id', inplace=True)

outcomes_df = df[[outcome_column]].copy()
outcomes_df['id'] = df['StudentID']
outcomes_df.set_index('id', inplace=True)

# Save the feature and outcome data
features_df.to_csv(features_out_path, index=True, sep=';')
outcomes_df.to_csv(outcomes_out_path, index=True, sep=';')
if source_out_path:
    # Add id column to the source dataframe
    df['id'] = df['StudentID']
    df.to_csv(source_out_path, index=False, sep=';')

print(f"Features saved to {features_out_path}")
print(f"Outcomes saved to {outcomes_out_path}")
if source_out_path:
    print(f"Source saved to {source_out_path}")
