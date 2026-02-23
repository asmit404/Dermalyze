#!/usr/bin/env bash
set -euo pipefail

# Install PyTorch with automatic CUDA/ROCm channel selection.
# Supported channels: cu126, cu128, cu130, rocm7.1, cpu

readonly VALID_CHANNELS=("cu126" "cu128" "cu130" "rocm7.1" "cpu")

has_command() {
  command -v "$1" >/dev/null 2>&1
}

print_error() {
  echo "[install_pytorch.sh] ERROR: $*" >&2
}

validate_channel() {
  local ch="${1:-}"
  for valid in "${VALID_CHANNELS[@]}"; do
    if [[ "$ch" == "$valid" ]]; then
      return 0
    fi
  done
  return 1
}

version_to_code() {
  # Converts "12.8" -> 1208, "13.0" -> 1300
  local v="$1"
  local major="${v%%.*}"
  local minor="${v#*.}"
  if [[ -z "$major" || -z "$minor" || "$major" == "$v" ]]; then
    echo "0"
    return
  fi
  # Remove any non-digit suffix just in case.
  major="${major//[^0-9]/}"
  minor="${minor//[^0-9]/}"
  if [[ -z "$major" || -z "$minor" ]]; then
    echo "0"
    return
  fi
  printf "%d%02d\n" "$major" "$minor"
}

cuda_version_to_channel() {
  local cuda_version="$1"
  local code
  code="$(version_to_code "$cuda_version")"

  if (( code >= 1300 )); then
    echo "cu130"
  elif (( code >= 1208 )); then
    echo "cu128"
  elif (( code >= 1206 )); then
    echo "cu126"
  else
    echo "cpu"
  fi
}

rocm_version_to_channel() {
  local rocm_version="$1"
  local code
  code="$(version_to_code "$rocm_version")"

  if (( code >= 701 )); then
    echo "rocm7.1"
  else
    echo "cpu"
  fi
}

detect_cuda_version() {
  local detected=""

  if has_command nvidia-smi; then
    detected="$(nvidia-smi | sed -n 's/.*CUDA Version: \([0-9]\+\.[0-9]\+\).*/\1/p' | head -n1 || true)"
  fi

  if [[ -z "$detected" ]] && has_command nvcc; then
    detected="$(nvcc --version | sed -n 's/.*release \([0-9]\+\.[0-9]\+\).*/\1/p' | head -n1 || true)"
  fi

  echo "$detected"
}

detect_rocm_version() {
  local detected=""

  if has_command rocminfo; then
    detected="$(rocminfo 2>/dev/null | sed -n 's/.*ROCm Version: \([0-9]\+\.[0-9]\+\).*/\1/p' | head -n1 || true)"
  fi

  if [[ -z "$detected" ]] && has_command hipcc; then
    detected="$(hipcc --version 2>/dev/null | sed -n 's/.*HIP version: \([0-9]\+\.[0-9]\+\).*/\1/p' | head -n1 || true)"
  fi

  if [[ -z "$detected" ]] && [[ -f /opt/rocm/.info/version ]]; then
    detected="$(sed -n 's/^\([0-9]\+\.[0-9]\+\).*$/\1/p' /opt/rocm/.info/version | head -n1 || true)"
  fi

  echo "$detected"
}

choose_channel() {
  local override="${TORCH_CHANNEL:-}"
  if [[ -n "$override" ]]; then
    if ! validate_channel "$override"; then
      print_error "Invalid TORCH_CHANNEL='$override'. Valid values: cu126, cu128, cu130, rocm7.1, cpu"
      exit 1
    fi
    echo "$override"
    return
  fi

  if [[ "$(uname -s)" == "Darwin" ]]; then
    echo "cpu"
    return
  fi

  local cuda_version
  cuda_version="$(detect_cuda_version)"
  if [[ -z "$cuda_version" ]]; then
    local rocm_version
    rocm_version="$(detect_rocm_version)"
    if [[ -z "$rocm_version" ]]; then
      echo "cpu"
      return
    fi

    rocm_version_to_channel "$rocm_version"
    return
  fi

  cuda_version_to_channel "$cuda_version"
}

ensure_python() {
  if has_command python; then
    echo "python"
  elif has_command python3; then
    echo "python3"
  else
    print_error "Python is not available in PATH."
    exit 1
  fi
}

main() {
  local py_cmd
  py_cmd="$(ensure_python)"

  local channel
  channel="$(choose_channel)"

  echo "[install_pytorch.sh] Selected PyTorch channel: $channel"

  "$py_cmd" -m pip install --upgrade pip

  if [[ "$channel" == "cpu" ]]; then
    if [[ "$(uname -s)" == "Darwin" ]]; then
      # macOS wheels are served from PyPI (MPS-enabled builds where available).
      "$py_cmd" -m pip install torch torchvision torchaudio
    else
      "$py_cmd" -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
    fi
  else
    "$py_cmd" -m pip install torch torchvision torchaudio --index-url "https://download.pytorch.org/whl/${channel}"
  fi

  echo "[install_pytorch.sh] PyTorch installation complete."
  "$py_cmd" - <<'PY'
import torch
print(f"torch={torch.__version__}")
print(f"cuda_available={torch.cuda.is_available()}")
PY
}

main "$@"
