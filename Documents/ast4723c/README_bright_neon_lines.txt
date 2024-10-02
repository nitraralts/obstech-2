README: Bright Neon Lines CSV

The purpose of this file is to store the distances between (visibly) bright neon lines without necessarily knowing the pixel/wavelength conversion ratio.
The first row and column are both the bright neon wavelengths. Below the first value in each data column, each line (corresponding to the row)'s distance to the line (corresponding to the column) is recorded, where a distance of 1 (or -1) is the closest line.

The purpose is to multiply a given column by the distance between a line suspected to correspond to that column's wavelength and the closest line to it, which, assuming a roughly linear fit, should make everything match with their true wavelengths.