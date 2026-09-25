$ErrorActionPreference = "Stop"
$project = Get-Content "$PSScriptRoot\..\pyproject.toml" -Raw
if ($project -notmatch '(?m)^version = "([^"]+)"') { throw "Could not read project version" }
$version = $Matches[1]
$baseVersion = ($version -split '-', 2)[0]
$numericParts = @($baseVersion -split '\.' | ForEach-Object { [int]$_ })
while ($numericParts.Count -lt 4) { $numericParts += 0 }
if ($version.Contains('-') -and $numericParts[3] -eq 0) { $numericParts[3] = 1 }
$numericVersion = ($numericParts[0..3] -join '.')
$iscc = Get-Command ISCC.exe -ErrorAction SilentlyContinue
if ($iscc) {
  $isccPath = $iscc.Source
} else {
  $candidate = "${env:ProgramFiles(x86)}\Inno Setup 6\ISCC.exe"
  if (-not (Test-Path $candidate)) { throw "Install Inno Setup 6, then run this script again." }
  $isccPath = $candidate
}
& $isccPath "/DMyAppVersion=$version" "/DMyAppNumericVersion=$numericVersion" "$PSScriptRoot\..\installer\FaceSwap.iss"
if ($LASTEXITCODE -ne 0) { throw "Inno Setup failed with exit code $LASTEXITCODE" }
Get-ChildItem "$PSScriptRoot\..\dist\installer\*.exe" | Get-FileHash -Algorithm SHA256 |
  ForEach-Object { "$($_.Hash.ToLower())  $($_.Path | Split-Path -Leaf)" } |
  Set-Content "$PSScriptRoot\..\dist\installer\SHA256SUMS.txt"
