# Install OpenBLAS and pkg-config on Windows for Meson builds.
# Called from both ci.yml and cibuildwheel before-all.

$ErrorActionPreference = 'Stop'

$OPENBLAS_VER = '0.3.28'
$INSTALL_DIR  = 'C:\openblas'

# 1. pkg-config (so Meson can parse .pc files)
Write-Host '--- Installing pkgconfiglite via Chocolatey ---'
choco install pkgconfiglite -y

# 2. Download OpenBLAS pre-built binaries
$zipUrl = "https://github.com/OpenMathLib/OpenBLAS/releases/download/v${OPENBLAS_VER}/OpenBLAS-${OPENBLAS_VER}-x64.zip"
$zipFile = "$env:TEMP\openblas.zip"
Write-Host "--- Downloading OpenBLAS ${OPENBLAS_VER} ---"
Invoke-WebRequest -Uri $zipUrl -OutFile $zipFile

# 3. Extract — the zip contains a single top-level folder
Write-Host '--- Extracting ---'
$tmpDir = "$env:TEMP\openblas-extract"
Expand-Archive $zipFile -DestinationPath $tmpDir -Force
$innerDir = Get-ChildItem $tmpDir | Select-Object -First 1
if (Test-Path $INSTALL_DIR) { Remove-Item $INSTALL_DIR -Recurse -Force }
Move-Item $innerDir.FullName $INSTALL_DIR

# 4. Create openblas.pc for dependency('openblas') in Meson
$pkgconfigDir = Join-Path $INSTALL_DIR 'lib\pkgconfig'
New-Item -ItemType Directory -Force -Path $pkgconfigDir | Out-Null

$pcContent = @'
prefix=C:/openblas
libdir=${prefix}/lib
includedir=${prefix}/include

Name: openblas
Description: OpenBLAS
Version: 0.3.28
Libs: -L${libdir} -lopenblas
Cflags: -I${includedir}
'@
Set-Content -Path (Join-Path $pkgconfigDir 'openblas.pc') -Value $pcContent -Encoding UTF8

Write-Host "--- OpenBLAS installed to $INSTALL_DIR ---"
Get-ChildItem $INSTALL_DIR -Recurse | Select-Object FullName | Format-Table -AutoSize
