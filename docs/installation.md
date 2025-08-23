# Installation Guide

Complete installation instructions for the Solar Differential Rotation Analysis Pipeline.

## 🎯 System Requirements

### Operating Systems
- **Linux** (Ubuntu 18.04+, CentOS 7+, etc.) - Recommended
- **macOS** (10.14+) - Fully supported
- **Windows** (10+) - Supported with notes below

### Hardware Requirements
- **RAM**: 4GB minimum, 8GB+ recommended for large datasets
- **Storage**: 1GB for software, additional space for data and results
- **CPU**: Multi-core recommended for faster processing

### Software Requirements
- **Python**: 3.8+ (3.9 or 3.10 recommended)
- **Git**: For cloning repository (optional)

## 🐍 Python Environment Setup

### Option 1: Using Conda (Recommended)

```bash
# Create dedicated environment
conda create -n solar-rotation python=3.10
conda activate solar-rotation

# Install main packages from conda-forge
conda install -c conda-forge numpy matplotlib scipy pandas astropy
conda install -c conda-forge opencv scikit-image

# Install SunPy (solar physics library)
conda install -c conda-forge sunpy

# Clone and install pipeline
git clone https://github.com/yourusername/solar-rotation-pipeline.git
cd solar-rotation-pipeline
pip install -r requirements.txt
```

### Option 2: Using Virtual Environment

```bash
# Create virtual environment
python -m venv solar-rotation-env

# Activate environment
# Linux/Mac:
source solar-rotation-env/bin/activate
# Windows:
solar-rotation-env\Scripts\activate

# Install requirements
pip install -r requirements.txt
```

### Option 3: System-wide Installation (Not Recommended)

```bash
# Direct installation (may conflict with other packages)
pip install -r requirements.txt
```

## 📦 Package Dependencies

### Core Requirements
```
numpy>=1.20.0          # Numerical computing
matplotlib>=3.3.0      # Plotting and visualization  
scipy>=1.6.0           # Scientific computing
pandas>=1.2.0          # Data manipulation
astropy>=4.2.0         # Astronomy computations
```

### Solar Physics & Image Processing
```
sunpy>=3.0.0           # Solar physics tools
opencv-python>=4.5.0   # Computer vision
scikit-image>=0.18.0   # Image processing
Pillow>=8.0.0          # Image I/O support
```

### Optional Dependencies
```
sunpy[net]             # For downloading solar data
jupyter>=1.0.0         # Notebook interface
pytest>=6.0.0         # Testing framework
```

## 🔧 Detailed Installation Steps

### Step 1: Python Installation

#### Linux (Ubuntu/Debian)
```bash
# Update package manager
sudo apt update

# Install Python and pip
sudo apt install python3.10 python3.10-venv python3-pip

# Install development tools
sudo apt install build-essential python3-dev
```

#### macOS
```bash
# Using Homebrew
brew install python@3.10

# Or download from python.org
```

#### Windows
1. Download Python from [python.org](https://python.org)
2. **Important**: Check "Add Python to PATH" during installation
3. Verify installation: `python --version`

### Step 2: Download Pipeline

#### Using Git (Recommended)
```bash
git clone https://github.com/yourusername/solar-rotation-pipeline.git
cd solar-rotation-pipeline
```

#### Manual Download
1. Download ZIP from GitHub
2. Extract to desired location
3. Navigate to extracted folder

### Step 3: Environment Setup

#### Create Isolated Environment
```bash
# Using venv
python -m venv solar-env
source solar-env/bin/activate  # Linux/Mac
# or
solar-env\Scripts\activate     # Windows

# Using conda
conda create -n solar-rotation python=3.10
conda activate solar-rotation
```

### Step 4: Install Dependencies

#### Standard Installation
```bash
pip install -r requirements.txt
```

#### Development Installation
```bash
# Install with development dependencies
pip install -r requirements.txt
pip install jupyter pytest black flake8
```

#### Verify Installation
```bash
python -c "import solar_pipeline; print('Installation successful!')"
```

## 🐧 Platform-Specific Notes

### Linux
- Most straightforward installation
- All features fully supported
- Recommended for server deployments

```bash
# Additional libraries for some distributions
sudo apt install libgl1-mesa-glx  # For OpenCV
sudo apt install libglib2.0-0     # For GUI features
```

### macOS
- Works well with Homebrew Python
- May need Xcode command line tools

```bash
# Install Xcode tools
xcode-select --install

# If using system Python, may need:
pip install --user -r requirements.txt
```

### Windows
- Use Anaconda for easiest setup
- Some packages may require Microsoft C++ Build Tools

```powershell
# If you encounter compilation errors:
# Install Microsoft C++ Build Tools from:
# https://visualstudio.microsoft.com/visual-cpp-build-tools/

# Alternative: Use Anaconda which provides pre-compiled packages
conda install -c conda-forge astropy sunpy opencv
```

## 🧪 Testing Installation

### Basic Functionality Test
```bash
# Test core imports
python -c "
import numpy as np
import matplotlib.pyplot as plt
import astropy
import sunpy
import cv2
from solar_pipeline import SolarAnalysisPipeline
print('✓ All core packages imported successfully')
"
```

### Run Example Analysis
```bash
# Create sample configuration
python solar_pipeline.py --create-config
echo "✓ Configuration template created"

# Test with sample data (if available)
python examples/sample_analysis.py
```

### Comprehensive Test Suite
```bash
# If pytest is installed
pytest tests/ -v
```

## 🚨 Common Installation Issues

### Issue 1: SunPy Installation Fails
```bash
# Solutions:
# 1. Use conda instead of pip
conda install -c conda-forge sunpy

# 2. Or install specific version
pip install sunpy==4.0.0

# 3. Install without optional dependencies first
pip install sunpy --no-deps
pip install -r requirements.txt
```

### Issue 2: OpenCV Issues
```bash
# Linux: Missing libraries
sudo apt install libopencv-dev python3-opencv

# Alternative installation
pip uninstall opencv-python
pip install opencv-python-headless  # Headless version
```

### Issue 3: Astropy Compilation Errors
```bash
# Use pre-compiled wheel
pip install --only-binary=astropy astropy

# Or use conda
conda install -c conda-forge astropy
```

### Issue 4: Permission Errors (Linux/Mac)
```bash
# Don't use sudo with pip, use virtual environment instead
python -m venv myenv
source myenv/bin/activate
pip install -r requirements.txt
```

### Issue 5: Windows Path Issues
```powershell
# Ensure Python is in PATH
python --version

# Use forward slashes or raw strings in paths
image_path = r"C:\data\solar\image.fits"
# or
image_path = "C:/data/solar/image.fits"
```

## ⚡ Performance Optimization

### For Large Datasets
```bash
# Install accelerated NumPy (if not already optimized)
pip uninstall numpy
pip install numpy[accelerate]

# Or use Intel MKL optimized packages
conda install -c intel numpy scipy
```

### Memory Management
```bash
# For memory-constrained systems, consider:
export OMP_NUM_THREADS=2  # Limit OpenMP threads
export OPENBLAS_NUM_THREADS=2  # Limit BLAS threads
```

## 🔄 Updating Installation

### Update Pipeline Code
```bash
# If installed from git
cd solar-rotation-pipeline
git pull origin main

# If manually downloaded, re-download latest version
```

### Update Dependencies
```bash
# Update all packages
pip install --upgrade -r requirements.txt

# Update specific package
pip install --upgrade sunpy
```

### Check for Issues After Update
```bash
# Re-run tests
python -c "from solar_pipeline import SolarAnalysisPipeline; print('Update successful')"
```

## 🌐 Alternative Installation Methods

### Docker Installation (Advanced)
```dockerfile
# Dockerfile
FROM python:3.10-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .
CMD ["python", "solar_pipeline.py", "--help"]
```

### Singularity Container (HPC)
```bash
# For HPC environments
singularity pull docker://python:3.10
singularity exec python_3.10.sif pip install -r requirements.txt
```

## 📋 Installation Checklist

- [ ] Python 3.8+ installed
- [ ] Virtual environment created and activated
- [ ] All required packages installed (`pip install -r requirements.txt`)
- [ ] Optional packages installed if needed (`sunpy[net]`)
- [ ] Pipeline code downloaded/cloned
- [ ] Basic import test passes
- [ ] Sample configuration created
- [ ] Test analysis runs (if sample data available)

## 🆘 Getting Help

If installation fails:

1. **Check Python version**: `python --version` (must be 3.8+)
2. **Check pip version**: `pip --version` (should be recent)
3. **Try conda instead**: Often resolves dependency conflicts
4. **Check error messages**: Look for specific package names
5. **Use virtual environment**: Avoid system-wide installations
6. **Search issues**: Check GitHub issues for similar problems

### Reporting Installation Problems

When reporting issues, include:
- Operating system and version
- Python version (`python --version`)
- Full error message
- Installation method attempted
- Output of `pip list` or `conda list`

## 🎉 Next Steps

After successful installation:

1. Read the [main README](../README.md)
2. Try the [examples](../examples/)
3. Review the [API documentation](api_reference.md)
4. Start with your own solar data!

---

**Installation complete!** 🌞 Ready to analyze solar rotation!
