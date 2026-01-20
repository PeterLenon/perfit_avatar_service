# SMPL Model Files

This directory should contain the SMPL body model files required for body reconstruction and measurement extraction.

## Download Instructions

1. Register at [https://smpl.is.tue.mpg.de/](https://smpl.is.tue.mpg.de/)
2. Download the "SMPL for Python" package
3. Extract and copy the following files to this directory:
   - `SMPL_MALE.pkl`
   - `SMPL_FEMALE.pkl`
   - `SMPL_NEUTRAL.pkl`

## Expected Directory Structure

```
ml/smpl/models/
├── README.md (this file)
├── SMPL_MALE.pkl
├── SMPL_FEMALE.pkl
└── SMPL_NEUTRAL.pkl
```

## Note

These files are not included in the repository due to licensing restrictions.
The SMPL model is licensed by the Max Planck Institute for Intelligent Systems.
