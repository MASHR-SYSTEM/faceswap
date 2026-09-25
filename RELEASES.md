# Windows release policy

The public download at `https://face.mashr.ai/` must point to the same immutable
installer bytes published on the corresponding GitHub Release. Source archives
and portable test archives are labelled separately and are never presented as
the Windows installer.

Stable public release is gated on Authenticode signing. Early prereleases may be
published unsigned when they are prominently labelled on both the website and
GitHub Release, include a verified SHA-256 checksum, and explain SmartScreen and
Smart App Control limitations.

For signed releases, the project signs the project-owned `FaceSwap.exe`, builds the Inno Setup installer
from that signed tree, then signs the installer. The `Publish signed Windows release`
workflow rejects missing or invalid signatures and requires approval through the
`signed-production-release` GitHub environment.

The website download card should show version, file size, publisher, SHA-256,
the InsightFace research-use notice, links to source and support, and this sequence:

1. Download for Windows.
2. Verify the checksum, run the installer and launch from the Start menu.
3. Preview the camera immediately.
4. Accept the upstream model terms only if eligible, then let the app download
   and verify the optional models.

Do not promote a prerelease to stable until a clean Windows 11 install,
upgrade, uninstall, Microsoft Defender, CPU fallback, and supported NVIDIA test
have passed against the exact release bytes.
