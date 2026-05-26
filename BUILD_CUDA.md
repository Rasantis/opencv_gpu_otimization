# Compilando OpenCV com CUDA + cudacodec (NVDEC nativo)

O `python3-opencv` do Ubuntu **não** tem CUDA. Para o modo `opencv-cuda`
(decode NVDEC nativo via `cv2.cudacodec`, frame na GPU) é preciso compilar
o OpenCV do fonte. Foi o que fizemos nesta bancada (RTX 3050, CUDA 12.4,
Python 3.14, OpenCV 4.13.0).

## 1) Dependências

```bash
sudo apt install -y build-essential cmake git pkg-config python3.14-dev \
    nvidia-cuda-toolkit g++-13 \
    libgstreamer1.0-dev libgstreamer-plugins-base1.0-dev \
    libjpeg-dev libpng-dev libtiff-dev libavcodec-dev libavformat-dev libswscale-dev
```

> CUDA 12.4 (toolkit do Ubuntu) **exige gcc ≤ 13** — por isso `g++-13`.

## 2) Headers do Video Codec SDK (nvcuvid.h / cuviddec.h / nvEncodeAPI.h)

O toolkit do Ubuntu **não** inclui esses headers (só os `dynlink_*` do
`libffmpeg-nvenc-dev`, que o cudacodec não aceita). Pegue os oficiais do
repositório público da NVIDIA e instale em `/usr/include`:

```bash
base=https://raw.githubusercontent.com/NVIDIA/video-sdk-samples/master/Samples/NvCodec
curl -sL $base/NvDecoder/cuviddec.h     -o /tmp/cuviddec.h
curl -sL $base/NvDecoder/nvcuvid.h      -o /tmp/nvcuvid.h
curl -sL $base/NvEncoder/nvEncodeAPI.h  -o /tmp/nvEncodeAPI.h
sudo cp /tmp/cuviddec.h /tmp/nvcuvid.h /tmp/nvEncodeAPI.h /usr/include/
```

A biblioteca de runtime `libnvcuvid.so` já vem com o driver NVIDIA.

## 3) Fonte

```bash
mkdir -p ~/opencv_build && cd ~/opencv_build
curl -sL https://github.com/opencv/opencv/archive/refs/tags/4.13.0.tar.gz | tar xz
curl -sL https://github.com/opencv/opencv_contrib/archive/refs/tags/4.13.0.tar.gz | tar xz
```

## 4) Configuração (enxuta — só os módulos necessários)

```bash
cmake -S opencv-4.13.0 -B build \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_C_COMPILER=/usr/bin/gcc-13 -DCMAKE_CXX_COMPILER=/usr/bin/g++-13 \
  -DCUDA_HOST_COMPILER=/usr/bin/gcc-13 -DCMAKE_CUDA_HOST_COMPILER=/usr/bin/gcc-13 \
  -DOPENCV_EXTRA_MODULES_PATH=$PWD/opencv_contrib-4.13.0/modules \
  -DWITH_CUDA=ON -DCUDA_ARCH_BIN=8.6 -DCUDA_ARCH_PTX= \
  -DWITH_NVCUVID=ON -DWITH_NVCUVENC=ON -DBUILD_opencv_cudacodec=ON \
  -DCUDA_nvcuvid_LIBRARY=/usr/lib/x86_64-linux-gnu/libnvcuvid.so \
  -DWITH_GSTREAMER=ON -DWITH_FFMPEG=ON \
  -DCMAKE_EXE_LINKER_FLAGS="-Wl,--allow-shlib-undefined" \
  -DCMAKE_SHARED_LINKER_FLAGS="-Wl,--allow-shlib-undefined" \
  -DCMAKE_MODULE_LINKER_FLAGS="-Wl,--allow-shlib-undefined" \
  -DBUILD_opencv_python3=ON -DPYTHON3_EXECUTABLE=/usr/bin/python3.14 \
  -DBUILD_LIST=cudev,python3,cudacodec,cudaimgproc,cudawarping,cudaarithm,videoio,imgcodecs,highgui \
  -DWITH_PROTOBUF=OFF -DBUILD_PROTOBUF=OFF -DWITH_ADE=OFF \
  -DBUILD_JPEG=OFF -DBUILD_PNG=OFF -DBUILD_TIFF=OFF -DBUILD_ZLIB=OFF \
  -DBUILD_TESTS=OFF -DBUILD_PERF_TESTS=OFF -DBUILD_EXAMPLES=OFF
```

> `CUDA_ARCH_BIN=8.6` é a arquitetura do RTX 3050 (Ampere). Ajuste pra sua
> GPU. `BUILD_LIST` corta o resto pra acelerar muito a compilação.
>
> **Os `-Wl,--allow-shlib-undefined` são essenciais aqui:** o FFmpeg do sistema
> é 8.0 (avcodec 62) e puxa `libsrt` compilado com gcc-15, que referencia
> `__cxa_call_terminate@CXXABI_1.3.15` — símbolo ausente no libstdc++ do
> **gcc-13** (que somos obrigados a usar pelo CUDA 12.4). Sem o flag, o teste
> de link do FFmpeg falha e o OpenCV desabilita o FFmpeg (`FFMPEG: NO`). O
> símbolo resolve em runtime no libstdc++ do gcc-15 (o `.so` do sistema).
> Confirme no resumo do cmake: `FFMPEG: YES`, `GStreamer: YES`,
> `NVIDIA CUDA: YES (... NVCUVID NVCUVENC)`.

## 5) Compilar

```bash
cmake --build build --parallel 8     # ~20-40 min nesta máquina
```

## 6) Instalar como cv2 único do sistema

```bash
sudo cmake --install build      # instala em /usr/local
sudo ldconfig
# /usr/local/lib/python3.14/dist-packages tem precedência sobre o cv2 do apt
python3 -c "import cv2; print(cv2.__version__, hasattr(cv2,'cudacodec'))"
# -> 4.13.0 True   (FFmpeg + GStreamer + CUDA/NVDEC, tudo num cv2 só)
```

Isso **não remove** o `python3-opencv` do apt — só o sombreia para o `import`.
Para reverter, basta `sudo rm -rf /usr/local/lib/python3.14/dist-packages/cv2`.
