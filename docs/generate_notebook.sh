#!/bin/bash

# Define notebook file name
NOTEBOOK_FILE="3d_gre_example.ipynb"

# Create notebook structure
cat <<EOF > $NOTEBOOK_FILE
{
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# MRI Sequence Design Package: User Guide\\n",
    "## Building a 3D GRE (Gradient-Recalled Echo) Sequence Example\\n",
    "\\n",
    "In this tutorial, we will guide you through building a 3D Gradient Echo (3D GRE) sequence using the MRI sequence design package.\\n",
    "We will demonstrate the high-level functionalities as well as show how to customize the sequence using lower-level routines.\\n",
    "Let’s get started!"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Step 0: Define Sequence Parameters\\n",
    "import numpy as np\\n",
    "\\n",
    "# Define sequence parameters\\n",
    "fov = [256e-3, 256e-3, 128e-3]  # Field of view in meters (x, y, z)\\n",
    "matrix_size = [256, 256, 64]  # Matrix size (x, y, z)\\n",
    "flip_angle = 15  # Flip angle in degrees\\n",
    "tr = 8e-3  # Repetition time (TR) in seconds\\n",
    "te = 4e-3  # Echo time (TE) in seconds\\n",
    "slice_thickness = 5e-3  # Slab thickness in meters\\n",
    "bandwidth = 200e3  # Bandwidth in Hz"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Step 1: Instantiate System Limits\\n",
    "import pypulseq as pp\\n",
    "\\n",
    "# Instantiate system limits\\n",
    "system_limits = pp.Opts(max_grad=32, grad_unit='mT/m', max_slew=130, slew_unit='T/m/s', \\n",
    "                        rf_ringdown_time=20e-6, rf_dead_time=100e-6, adc_dead_time=10e-6)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Step 2: Instantiate the Sequence Object\\n",
    "# Instantiate the sequence object\\n",
    "seq = MySequence(system_limits)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Step 3: Define Building Blocks of the Sequence\\n",
    "\\n",
    "# 3.1: Slab-selective RF pulse\\n",
    "rf_duration = 3e-3  # RF pulse duration\\n",
    "flip_angle_rad = np.deg2rad(flip_angle)\\n",
    "\\n",
    "rf, gz = seq.make_sinc_pulse(flip_angle=flip_angle_rad, duration=rf_duration, \\n",
    "                             slice_thickness=slice_thickness, system_limits=system_limits)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# 3.2: Refocusing Gradient\\n",
    "gz_refocus = seq.make_trapezoid(channel='z', area=-gz.area/2, duration=rf_duration, system_limits=system_limits)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# 3.3: Phase Encoding Gradients\\n",
    "gx, gy, gz_pe = seq.make_phase_encodings(fov=fov, matrix_size=matrix_size)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# 3.4: Readout Gradient and ADC\\n",
    "gx_readout, adc = seq.make_readout(fov=fov[0], matrix_size=matrix_size[0], \\n",
    "                                  duration=3.2e-3, system_limits=system_limits)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# 3.5: Rewinders and Spoilers\\n",
    "gx_rew = seq.make_trapezoid(channel='x', area=-gx.area/2, duration=2e-3, system_limits=system_limits)\\n",
    "gy_rew = seq.make_trapezoid(channel='y', area=-gy.area/2, duration=2e-3, system_limits=system_limits)\\n",
    "gz_spoil = seq.make_trapezoid(channel='z', area=4*gz.area, duration=3e-3, system_limits=system_limits)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Step 4: Define Phase Encoding Plan\\n",
    "# High-level phase encoding scaling\\n",
    "pe_scaling = seq.calculate_pe_scaling(matrix_size)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Low-level custom phase encoding plan using numpy\\n",
    "pe_table = np.linspace(-1, 1, matrix_size[1])\\n",
    "gx_scaled = gx * pe_table[:, np.newaxis]\\n",
    "gy_scaled = gy * pe_table[:, np.newaxis]"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Step 5: Build the Scan Loop\\n",
    "for ky in range(matrix_size[1]):\\n",
    "    for kz in range(matrix_size[2]):\\n",
    "        seq.add_block(rf, gz, gx_scaled[ky], gy_scaled[kz], adc)\\n",
    "        seq.add_block(gx_rew, gy_rew, gz_spoil)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Step 6: Export the Sequence\\n",
    "seq.export('3d_gre.seq')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Wrapping the code into a function\\n",
    "def build_3d_gre_sequence(fov, matrix_size, flip_angle, tr, te, slice_thickness, bandwidth):\\n",
    "    system_limits = pp.Opts(max_grad=32, grad_unit='mT/m', max_slew=130, slew_unit='T/m/s',\\n",
    "                            rf_ringdown_time=20e-6, rf_dead_time=100e-6, adc_dead_time=10e-6)\\n",
    "    seq = MySequence(system_limits)\\n",
    "    \\n",
    "    rf, gz = seq.make_sinc_pulse(flip_angle=np.deg2rad(flip_angle), duration=3e-3, \\n",
    "                                 slice_thickness=slice_thickness, system_limits=system_limits)\\n",
    "    gz_refocus = seq.make_trapezoid(channel='z', area=-gz.area/2, duration=3e-3, system_limits=system_limits)\\n",
    "    \\n",
    "    gx, gy, gz_pe = seq.make_phase_encodings(fov=fov, matrix_size=matrix_size)\\n",
    "    gx_readout, adc = seq.make_readout(fov=fov[0], matrix_size=matrix_size[0], duration=3.2e-3, system_limits=system_limits)\\n",
    "    \\n",
    "    gx_rew = seq.make_trapezoid(channel='x', area=-gx.area/2, duration=2e-3, system_limits=system_limits)\\n",
    "    gy_rew = seq.make_trapezoid(channel='y', area=-gy.area/2, duration=2e-3, system_limits=system_limits)\\n",
    "    gz_spoil = seq.make_trapezoid(channel='z', area=4*gz.area, duration=3e-3, system_limits=system_limits)\\n",
    "    \\n",
    "    for ky in range(matrix_size[1]):\\n",
    "        for kz in range(matrix_size[2]):\\n",
    "            seq.add_block(rf, gz, gx_scaled[ky], gy_scaled[kz], adc)\\n",
    "            seq.add_block(gx_rew, gy_rew, gz_spoil)\\n",
    "    \\n",
    "    seq.export('3d_gre.seq')"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "version": "3.8"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 2
}
EOF

echo "Notebook generated as $NOTEBOOK_FILE"

