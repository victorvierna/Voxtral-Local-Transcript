param(
    [string]$Distro = "Ubuntu",
    [string]$InstallDir = "~/.voxtray",
    [string]$BundleDir = ""
)

$ErrorActionPreference = "Stop"

function Escape-BashSingleQuotes {
    param([string]$Value)
    return $Value -replace "'", "'\"'\"'"
}

function Convert-ToBashPath {
    param([string]$PathValue)
    if ($PathValue.StartsWith("~/")) {
        return '$HOME/' + $PathValue.Substring(2)
    }
    return $PathValue
}

function New-CmdLauncher {
    param(
        [Parameter(Mandatory = $true)][string]$Path,
        [Parameter(Mandatory = $true)][string]$DistroName,
        [Parameter(Mandatory = $true)][string]$BashCommand,
        [bool]$KeepOpen = $false
    )

    $lines = @(
        "@echo off",
        "wsl.exe -d `"$DistroName`" -- bash -lc `"$BashCommand`""
    )
    if ($KeepOpen) {
        $lines += ""
        $lines += "echo."
        $lines += "pause"
    }
    Set-Content -Path $Path -Value ($lines -join [Environment]::NewLine) -Encoding ASCII
}

function New-LnkShortcut {
    param(
        [Parameter(Mandatory = $true)][string]$ShortcutPath,
        [Parameter(Mandatory = $true)][string]$TargetPath,
        [Parameter(Mandatory = $true)][string]$WorkingDirectory,
        [string]$Description = ""
    )

    $shell = New-Object -ComObject WScript.Shell
    $shortcut = $shell.CreateShortcut($ShortcutPath)
    $shortcut.TargetPath = $TargetPath
    $shortcut.WorkingDirectory = $WorkingDirectory
    if ($Description) {
        $shortcut.Description = $Description
    }
    $shortcut.Save()
}

if (-not (Get-Command wsl.exe -ErrorAction SilentlyContinue)) {
    throw "wsl.exe is not available. Install WSL2 first."
}

if ([string]::IsNullOrWhiteSpace($BundleDir)) {
    $BundleDir = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
} else {
    $BundleDir = (Resolve-Path $BundleDir).Path
}

$distros = @(wsl.exe -l -q | ForEach-Object { $_.Trim() } | Where-Object { $_ -ne "" })
if (-not ($distros -contains $Distro)) {
    throw "WSL distro '$Distro' not found. Installed distros: $($distros -join ', ')"
}

$bundleWsl = (& wsl.exe -d $Distro -- wslpath -a "$BundleDir").Trim()
if ([string]::IsNullOrWhiteSpace($bundleWsl)) {
    throw "Could not convert bundle path to WSL path."
}

$installDirBash = Convert-ToBashPath -PathValue $InstallDir
$bundleWslEscaped = Escape-BashSingleQuotes -Value $bundleWsl
$installDirEscaped = Escape-BashSingleQuotes -Value $installDirBash
$installCommand = "cd '$bundleWslEscaped' && chmod +x scripts/install_wsl2.sh && scripts/install_wsl2.sh --install-dir '$installDirEscaped'"

Write-Host "Installing Voxtray into WSL distro '$Distro'..." -ForegroundColor Cyan
& wsl.exe -d $Distro -- bash -lc $installCommand

$launcherDir = Join-Path $env:LOCALAPPDATA "VoxtrayWSL"
New-Item -ItemType Directory -Path $launcherDir -Force | Out-Null

$voxtrayBin = "$installDirBash/.venv/bin/voxtray"

$toggleCmd = Join-Path $launcherDir "Voxtray-Toggle.cmd"
$warmOnCmd = Join-Path $launcherDir "Voxtray-Warm-On.cmd"
$warmOffCmd = Join-Path $launcherDir "Voxtray-Warm-Off.cmd"
$statusCmd = Join-Path $launcherDir "Voxtray-Status.cmd"
$logsCmd = Join-Path $launcherDir "Voxtray-Logs.cmd"

New-CmdLauncher -Path $toggleCmd -DistroName $Distro -BashCommand "$voxtrayBin record --toggle"
New-CmdLauncher -Path $warmOnCmd -DistroName $Distro -BashCommand "$voxtrayBin warm on"
New-CmdLauncher -Path $warmOffCmd -DistroName $Distro -BashCommand "$voxtrayBin warm off"
New-CmdLauncher -Path $statusCmd -DistroName $Distro -BashCommand "$voxtrayBin status && $voxtrayBin warm status" -KeepOpen $true
New-CmdLauncher -Path $logsCmd -DistroName $Distro -BashCommand "$voxtrayBin logs --target all --lines 200" -KeepOpen $true

$desktopDir = [Environment]::GetFolderPath("Desktop")
$startMenuDir = Join-Path $env:APPDATA "Microsoft\Windows\Start Menu\Programs\Voxtray WSL"
New-Item -ItemType Directory -Path $startMenuDir -Force | Out-Null

New-LnkShortcut -ShortcutPath (Join-Path $desktopDir "Voxtray Toggle.lnk") -TargetPath $toggleCmd -WorkingDirectory $launcherDir -Description "Toggle Voxtray recording in WSL"
New-LnkShortcut -ShortcutPath (Join-Path $desktopDir "Voxtray Status.lnk") -TargetPath $statusCmd -WorkingDirectory $launcherDir -Description "Show Voxtray status in WSL"

New-LnkShortcut -ShortcutPath (Join-Path $startMenuDir "Voxtray Toggle.lnk") -TargetPath $toggleCmd -WorkingDirectory $launcherDir -Description "Toggle Voxtray recording in WSL"
New-LnkShortcut -ShortcutPath (Join-Path $startMenuDir "Voxtray Warm On.lnk") -TargetPath $warmOnCmd -WorkingDirectory $launcherDir -Description "Enable warm mode in WSL"
New-LnkShortcut -ShortcutPath (Join-Path $startMenuDir "Voxtray Warm Off.lnk") -TargetPath $warmOffCmd -WorkingDirectory $launcherDir -Description "Disable warm mode in WSL"
New-LnkShortcut -ShortcutPath (Join-Path $startMenuDir "Voxtray Status.lnk") -TargetPath $statusCmd -WorkingDirectory $launcherDir -Description "Show Voxtray status in WSL"
New-LnkShortcut -ShortcutPath (Join-Path $startMenuDir "Voxtray Logs.lnk") -TargetPath $logsCmd -WorkingDirectory $launcherDir -Description "Show Voxtray logs in WSL"

Write-Host ""
Write-Host "Done. Launchers created in:" -ForegroundColor Green
Write-Host "  $launcherDir"
Write-Host ""
Write-Host "Desktop shortcuts created:"
Write-Host "  Voxtray Toggle"
Write-Host "  Voxtray Status"
Write-Host ""
Write-Host "You can pin those shortcuts to Start or Taskbar."
