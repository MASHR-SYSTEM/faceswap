#!/usr/bin/env bash

_faceswap_neural_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

_faceswap_prepend_library_path() {
  local path="$1"
  if [[ -d "$path" ]]; then
    case ":${LD_LIBRARY_PATH:-}:" in
      *":$path:"*) ;;
      *) export LD_LIBRARY_PATH="$path${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" ;;
    esac
  fi
}

_faceswap_prepend_library_path "/usr/local/lib/ollama/cuda_v13"
_faceswap_prepend_library_path "/usr/local/lib/ollama/mlx_cuda_v13"
_faceswap_prepend_library_path "/usr/local/lib/ollama/cuda_v12"

for _faceswap_site_packages in "$_faceswap_neural_root"/.venv/lib/python*/site-packages; do
  [[ -d "$_faceswap_site_packages" ]] || continue
  _faceswap_prepend_library_path "$_faceswap_site_packages/nvidia/cublas/lib"
  _faceswap_prepend_library_path "$_faceswap_site_packages/nvidia/cuda_runtime/lib"
  _faceswap_prepend_library_path "$_faceswap_site_packages/nvidia/cudnn/lib"
  _faceswap_prepend_library_path "$_faceswap_site_packages/nvidia/cufft/lib"
  _faceswap_prepend_library_path "$_faceswap_site_packages/nvidia/curand/lib"
  _faceswap_prepend_library_path "$_faceswap_site_packages/nvidia/cusolver/lib"
  _faceswap_prepend_library_path "$_faceswap_site_packages/nvidia/cusparse/lib"
  _faceswap_prepend_library_path "$_faceswap_site_packages/nvidia/nccl/lib"
  _faceswap_prepend_library_path "$_faceswap_site_packages/nvidia/nvjitlink/lib"
  _faceswap_prepend_library_path "$_faceswap_site_packages/nvidia/nvtx/lib"
  _faceswap_prepend_library_path "$_faceswap_site_packages/nvidia/cu13/lib"
  _faceswap_prepend_library_path "$_faceswap_site_packages/tensorrt_libs"
done

unset -f _faceswap_prepend_library_path
unset _faceswap_site_packages
unset _faceswap_neural_root
