VERS_1.0 {
  global:
    extern "C" {
      GetPjrtApi;
      MosaicGpuCaptureDeviceless;
      MosaicGpuCompile;
      MosaicGpuUnload;
    };

  local:
    *;
};
