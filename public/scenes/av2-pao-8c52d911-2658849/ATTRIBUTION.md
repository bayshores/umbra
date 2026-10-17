# Argoverse 2 attribution

This scene is made from the Argoverse 2 Sensor Dataset, log `8c52d911-fe34-3424-9864-d3fdfac38064`. © 2021 Argo AI, LLC.

- Publisher: https://www.argoverse.org/av2.html
- Terms of use: https://www.argoverse.org/about.html#terms-of-use
- License: [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/). You may share and adapt this data for non-commercial purposes only. Give credit to Argo AI, and release adaptations under the same license.

Every file in this folder is derived from that log: the camera image (`recorded.png`), the LiDAR points and measured surfaces (`.bin` files, `ground.json`, `tier-0-ground.json`) and the map-based geometry (`scene.json`, `suggestions.json`).

Changes made by umbra: Mapped geometry and annotations converted to umbra scene coordinates; other same-recording LiDAR sweeps registered into measured triangles. Scoring sweep excluded; holes retained. No endorsement implied.

The Argoverse terms forbid trying to identify people in the data and using Argo's name, logo or trademarks without written permission (attribution excepted). Argo AI does not endorse umbra. An Argoverse maintainer confirmed this hackathon use on [argoverse/av2-api#340](https://github.com/argoverse/av2-api/issues/340).

`scene.json` records the sha256 of the original local source record (`sourceHashes.license`, `dbe438e1…`). That record is not included because it names local folders; this file replaces it.
