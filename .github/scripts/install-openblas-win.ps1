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

# 4. Ensure lib/ and include/ directories exist
$libDir = Join-Path $INSTALL_DIR 'lib'
$incDir = Join-Path $INSTALL_DIR 'include'
$binDir = Join-Path $INSTALL_DIR 'bin'
New-Item -ItemType Directory -Force -Path $libDir | Out-Null
New-Item -ItemType Directory -Force -Path $incDir | Out-Null

# 5. Find the DLL wherever it landed and copy to standard locations
$dll = Get-ChildItem -Path $INSTALL_DIR -Recurse -Filter 'libopenblas.dll' | Select-Object -First 1
if (-not $dll) {
    Write-Error "libopenblas.dll not found in $INSTALL_DIR"
    exit 1
}
Write-Host "Found DLL at: $($dll.FullName)"

# Copy DLL to bin/ (for PATH / runtime) and to lib/ (for linker), skipping self-copies
New-Item -ItemType Directory -Force -Path $binDir | Out-Null
$binDest = Join-Path $binDir 'libopenblas.dll'
$libDest = Join-Path $libDir 'libopenblas.dll'
if ($dll.FullName -ne (Resolve-Path $binDest -ErrorAction SilentlyContinue)) {
    Copy-Item $dll.FullName $binDest -Force
}
if ($dll.FullName -ne (Resolve-Path $libDest -ErrorAction SilentlyContinue)) {
    Copy-Item $dll.FullName $libDest -Force
}

# 6. Generate MinGW import library (.dll.a) from the DLL
#    The pre-built OpenBLAS zip does not include a MinGW import library,
#    only the bare DLL.  We generate one with gendef + dlltool.
Write-Host '--- Generating MinGW import library ---'
Push-Location $INSTALL_DIR

# gendef creates a .def file listing all exported symbols
& gendef libopenblas.dll
if ($LASTEXITCODE -ne 0) {
    Write-Host 'gendef not found, trying dlltool directly with a minimal .def'
    # Fallback: create .def manually from dumpbin or just let dlltool try
    & dlltool -z libopenblas.def --export-all-symbols libopenblas.dll
}

# dlltool creates the import library
& dlltool -d libopenblas.def -l (Join-Path $libDir 'libopenblas.dll.a') -D libopenblas.dll
if ($LASTEXITCODE -ne 0) {
    Write-Error 'dlltool failed to create import library'
    exit 1
}
Pop-Location

Write-Host "Import library created: $(Join-Path $libDir 'libopenblas.dll.a')"

# 7. Create openblas.pc for dependency('openblas') in Meson
$pkgconfigDir = Join-Path $libDir 'pkgconfig'
New-Item -ItemType Directory -Force -Path $pkgconfigDir | Out-Null

$pcContent = @"
prefix=C:/openblas
libdir=`${prefix}/lib
includedir=`${prefix}/include

Name: openblas
Description: OpenBLAS
Version: $OPENBLAS_VER
Libs: -L`${libdir} -lopenblas
Cflags: -I`${includedir}
"@
Set-Content -Path (Join-Path $pkgconfigDir 'openblas.pc') -Value $pcContent -Encoding UTF8

Write-Host "--- OpenBLAS installed to $INSTALL_DIR ---"
Get-ChildItem $INSTALL_DIR -Recurse | Select-Object FullName | Format-Table -AutoSize
