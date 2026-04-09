## Setup

1. Install OpenCV:
   - **macOS:** `brew install opencv`
   - **Linux:** `sudo apt install libopencv-dev`

2. Build:
   ```bash
   mkdir build
   cd build && cmake .. && make
   ```

3. Run:
   ```bash
   ./inpaint
   ```

## Building on GHC / CMU AFS
```
Install OpenCV (first time only)
bashwget -O opencv.zip https://github.com/opencv/opencv/archive/4.9.0.zip
unzip opencv.zip && mkdir -p opencv-build && cd opencv-build
cmake ../opencv-4.9.0 \
  -DCMAKE_INSTALL_PREFIX=$HOME/local \
  -DBUILD_TESTS=OFF \
  -DBUILD_EXAMPLES=OFF \
  -DBUILD_DOCS=OFF
make -j$(nproc)
make install
```
cd ~ && rm -rf opencv-build opencv-4.9.0 opencv.zip

bashexport LD_LIBRARY_PATH=$HOME/local/lib:$LD_LIBRARY_PATH
Add this to ~/.bashrc to persist it.

## References

https://github.com/younesse-cv/PatchMatch
https://github.com/vacancy/PyPatchMatch