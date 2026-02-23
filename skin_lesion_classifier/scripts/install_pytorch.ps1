$ErrorActionPreference = 'Stop'

# Install PyTorch with automatic CUDA channel selection.
# Supported channels: cu126, cu128, cu130, cpu

$ValidChannels = @('cu126', 'cu128', 'cu130', 'cpu')

function Write-Err {
    param([string]$Message)
    Write-Error "[install_pytorch.ps1] $Message"
}

function Test-ValidChannel {
    param([string]$Channel)
    return $ValidChannels -contains $Channel
}

function Get-CudaVersion {
    $cudaVersion = $null

    $nvidiaSmi = Get-Command nvidia-smi -ErrorAction SilentlyContinue
    if ($nvidiaSmi) {
        $smiOutput = & nvidia-smi 2>$null | Out-String
        $match = [regex]::Match($smiOutput, 'CUDA Version:\s*([0-9]+\.[0-9]+)')
        if ($match.Success) {
            $cudaVersion = $match.Groups[1].Value
        }
    }

    if (-not $cudaVersion) {
        $nvcc = Get-Command nvcc -ErrorAction SilentlyContinue
        if ($nvcc) {
            $nvccOutput = & nvcc --version 2>$null | Out-String
            $match = [regex]::Match($nvccOutput, 'release\s*([0-9]+\.[0-9]+)')
            if ($match.Success) {
                $cudaVersion = $match.Groups[1].Value
            }
        }
    }

    return $cudaVersion
}

function Get-ChannelFromCudaVersion {
    param([string]$CudaVersion)

    $v = [version]$CudaVersion
    if ($v -ge [version]'13.0') { return 'cu130' }
    if ($v -ge [version]'12.8') { return 'cu128' }
    if ($v -ge [version]'12.6') { return 'cu126' }
    return 'cpu'
}

function Get-SelectedChannel {
    $override = $env:TORCH_CHANNEL
    if ($override) {
        if (-not (Test-ValidChannel -Channel $override)) {
            Write-Err "Invalid TORCH_CHANNEL='$override'. Valid values: cu126, cu128, cu130, cpu"
            exit 1
        }
        return $override
    }

    $cudaVersion = Get-CudaVersion
    if (-not $cudaVersion) {
        return 'cpu'
    }

    return Get-ChannelFromCudaVersion -CudaVersion $cudaVersion
}

function Get-PythonCommand {
    if (Get-Command py -ErrorAction SilentlyContinue) {
        return @('py', '-3')
    }
    if (Get-Command python -ErrorAction SilentlyContinue) {
        return @('python')
    }
    Write-Err 'Python is not available in PATH.'
    exit 1
}

function Invoke-Pip {
    param(
        [string[]]$PythonCmd,
        [string[]]$PipArgs
    )

    if ($PythonCmd.Count -gt 1) {
        & $PythonCmd[0] $PythonCmd[1..($PythonCmd.Count - 1)] -m pip @PipArgs
    }
    else {
        & $PythonCmd[0] -m pip @PipArgs
    }
}

$pythonCmd = Get-PythonCommand
$channel = Get-SelectedChannel

Write-Host "[install_pytorch.ps1] Selected PyTorch channel: $channel"

Invoke-Pip -PythonCmd $pythonCmd -PipArgs @('install', '--upgrade', 'pip')

if ($channel -eq 'cpu') {
    Invoke-Pip -PythonCmd $pythonCmd -PipArgs @('install', 'torch', 'torchvision', 'torchaudio', '--index-url', 'https://download.pytorch.org/whl/cpu')
}
else {
    Invoke-Pip -PythonCmd $pythonCmd -PipArgs @('install', 'torch', 'torchvision', 'torchaudio', '--index-url', "https://download.pytorch.org/whl/$channel")
}

Write-Host '[install_pytorch.ps1] PyTorch installation complete.'
if ($pythonCmd.Count -gt 1) {
    & $pythonCmd[0] $pythonCmd[1..($pythonCmd.Count - 1)] -c "import torch; print(f'torch={torch.__version__}'); print(f'cuda_available={torch.cuda.is_available()}')"
}
else {
    & $pythonCmd[0] -c "import torch; print(f'torch={torch.__version__}'); print(f'cuda_available={torch.cuda.is_available()}')"
}
