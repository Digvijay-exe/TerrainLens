# TerrainLens Limitations & Scope of Applicability

## 1. Algorithmic Scope
1. **Classical Computer Vision Only:** TerrainLens deliberately executes on classical deterministic algorithms (OpenCV, SciPy, scikit-image). It does not employ deep learning or convolutional neural networks (e.g. U-Net, SAM), guaranteeing CPU compatibility without GPU requirements.
2. **Single-Band Elevation Constraints:** For GeoTIFF rasters, topographic processing is performed on Band 1. Multi-band orthomosaics or hyperspectral datasets are processed in Terrain Image mode.
3. **Planar Spatial Coordinate Assumption:** Slope and hillshade algorithms calculate gradients using Horn's planar 3x3 kernels. For geographic coordinate systems (latitude/longitude in degrees), metric cell sizes should ideally be projected (e.g. UTM) for true metric slope calculation.

## 2. Degradation Simulation Assumptions
1. **Additive Noise Model:** Additive Gaussian noise assumes sensor thermal noise. Poisson shot noise, speckle noise (SAR), and atmospheric turbulence are modeled indirectly.
2. **Resolution Restoration:** Upsampling back to the original dimension preserves spatial grid alignment for pixel-wise matrix comparisons, but introduces nearest-neighbor blockiness typical of pixel binning.

## 3. Data Integrity Limits
1. Maximum supported file size is 100 MB per image or DEM raster.
2. Arrays with $>50\%$ missing/NaN pixels are flagged as invalid to prevent spurious geomorphometric artifacts.
