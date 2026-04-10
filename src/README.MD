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

## OpenCV Configuration on GHC / CMU AFS

Install OpenCV
```
wget -O opencv.zip https://github.com/opencv/opencv/archive/4.9.0.zip
unzip opencv.zip && mkdir -p opencv-build && cd opencv-build
cmake ../opencv-4.9.0 \
  -DCMAKE_INSTALL_PREFIX=$HOME/local \
  -DBUILD_TESTS=OFF \
  -DBUILD_EXAMPLES=OFF \
  -DBUILD_DOCS=OFF
make -j$(nproc)
make install
```

Finally delete all files created
``` cd ~ && rm -rf opencv-build opencv-4.9.0 opencv.zip ```

Add this to ~/.bashrc to persist it.
```export LD_LIBRARY_PATH=$HOME/local/lib:$LD_LIBRARY_PATH```


## References

https://github.com/younesse-cv/PatchMatch
https://github.com/vacancy/PyPatchMatch