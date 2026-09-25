# Windows release policy

The public download at `https://face.mashr.ai/` must point to the same immutable
installer bytes published on the corresponding GitHub Release. Source archives
and portable test archives are labelled separately and are never presented as
the Windows installer.

Public release is gated on SignPath Foundation approval and Authenticode signing.
The project signs the project-owned `FaceSwap.exe`, builds the Inno Setup installer
from that signed tree, then signs the installer. The `Publish signed Windows release`
workflow rejects missing or invalid signatures and requires approval through the
`signed-production-release` GitHub environment.

The website download card should show version, file size, publisher, SHA-256,
the InsightFace research-use notice, links to source and support, and this sequence:

1. Download for Windows.
2. Run the signed installer and launch from the Start menu.
3. Preview the camera immediately.
4. Accept the upstream model terms only if eligible, then let the app download
   and verify the optional models.

Do not enable the public download button until a clean Windows 11 install,
upgrade, uninstall, Microsoft Defender, CPU fallback, and supported NVIDIA test
have passed against the exact signed release bytes.
