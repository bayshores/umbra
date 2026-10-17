# Third-party notices

umbra's own code is licensed under the Apache License 2.0 ([LICENSE](LICENSE)). This file lists the third-party software, fonts, icons, models, services and data that umbra uses, with their licenses. Dataset terms are separate from the code license.

## Shipped in the built web app

This section covers the JavaScript, CSS and font files emitted into `dist/client/assets` by `npm run build`. Versions come from the installed packages, not dependency ranges. The offline audit was made on 2026-10-16.

| Package | Installed version | License (SPDX) | Copyright from the license | Files or code shipped | License text |
| --- | --- | --- | --- | --- | --- |
| `react` | `19.2.0` | `MIT` | Copyright (c) Meta Platforms, Inc. and affiliates. | UI code and JSX runtime | [Full text](licenses/MIT-react.txt) |
| `react-dom` | `19.2.0` | `MIT` | Copyright (c) Meta Platforms, Inc. and affiliates. | DOM renderer | [Full text](licenses/MIT-react.txt) |
| `scheduler` | `0.27.0` | `MIT` | Copyright (c) Meta Platforms, Inc. and affiliates. | React scheduling code; transitive dependency | [Full text](licenses/MIT-react.txt) |
| `three` | `0.186.1` | `MIT` | Copyright © 2010-2026 three.js authors | Three.js core and WebGL renderer | [Full text](licenses/MIT-three.txt) |
| `@phosphor-icons/react` | `2.1.10` | `MIT` | Copyright (c) 2020 Phosphor Icons | Icon definitions, React components and shared icon code | [Full text](licenses/MIT-phosphor-icons.txt) |
| `@fontsource/overpass-mono` | `5.3.0` | `OFL-1.1` | Copyright 2021 The Overpass Project Authors (https://github.com/RedHatOfficial/Overpass) | CSS and 30 WOFF/WOFF2 font files; weights 400, 600 and 700 | [Full text](licenses/OFL-1.1-overpass-mono.txt) |
| `vite` | `6.4.2` | `MIT` | Copyright (c) 2019-present, VoidZero Inc. and Vite contributors | Browser module-preload polyfill | [Full text](licenses/MIT-vite-core.txt) |
| `@rollup/plugin-commonjs` helper embedded in Vite | Bundled in `vite@6.4.2`; standalone version not verified locally | `MIT` | Copyright (c) 2019 RollupJS Plugin Contributors (https://github.com/rollup/plugins/graphs/contributors) | `commonjsHelpers.js`; not installed as a separate package | [Full text](licenses/MIT-rollup-commonjs-from-vite.txt) |

React, React DOM and scheduler contain byte-identical license files. They share one copy here. Vite's license document contains several separate notices. The two complete MIT license blocks needed by the browser helpers are copied byte-for-byte, including their original formatting. The independent CommonJS plugin version is not in the installed metadata; Vite's development dependency range is not an exact version.

The included Phosphor icon definitions are ArrowCounterClockwise, ArrowUp, CaretDown, CrosshairSimple, Cube, Eye, PersonSimple, Play, RoadHorizon, Ruler, Stop and XCircle. The font subsets are Cyrillic, Cyrillic Extended, Latin, Latin Extended and Vietnamese. All 30 emitted font files match the installed source files by SHA256.

The MIT notices and permission texts must accompany redistributed copies. The Overpass Mono fonts stay under OFL-1.1. That license requires the font copyright and license notice, restricts selling the fonts by themselves, and has rules for modified font names.

A build also copies `public/scenes`. It holds four tracked Argoverse 2 scenes (CC BY-NC-SA 4.0) and one tracked DrivingStereo scene (MIT); their data and model terms are covered below and in [public/scenes/README.md](public/scenes/README.md).

## Build and test tools

These tools run on the developer's machine. They are not application code emitted into the browser bundle. Node.js built-in test and standard-library modules are also used; they are supplied by the user's Node.js installation.

| Direct dependency not bundled | Installed version | License (SPDX) | Use |
| --- | --- | --- | --- |
| `@vitejs/plugin-react` | `5.0.4` | `MIT` | React build/development transform |

Vite is also a build and development-server tool. Its two browser helpers are listed above. Babel, Rollup, esbuild, React Refresh and their other build dependencies do not otherwise appear as installed-package modules in the production import graph.

`caniuse-lite` is browser-support data under **CC-BY-4.0**. It is used during the build. Its data is not emitted in this app's bundle. Its attribution requirements are separate from ordinary software MIT notices. Redistribution of that data requires the supplied attribution and license information, and a statement of changes. Its [full license text](licenses/CC-BY-4.0-caniuse-lite.txt) is included. No caniuse data was changed by this notices audit.

The next table was generated on 2026-10-16 by a script reading every `packages` entry in `package-lock.json`, and the corresponding installed `package.json` and license files when present. It includes the root entry so nothing is silently omitted. There are 117 dependency entries: 68 installed and 49 absent platform-specific optional packages. All 68 installed versions match the lockfile. The root entry declares Apache-2.0. Every dependency license field contains a standard SPDX identifier. Missing installed license texts are marked separately.

| Package | Locked version | License (SPDX, as declared) | Local evidence |
| --- | --- | --- | --- |
| umbra (root entry) | `0.0.0` | `Apache-2.0` | Project itself; not a third-party package; see [LICENSE](LICENSE) |
| `@babel/code-frame` | `7.29.7` | `MIT` | Installed; metadata and license text read |
| `@babel/compat-data` | `7.29.7` | `MIT` | Installed; metadata and license text read |
| `@babel/core` | `7.29.7` | `MIT` | Installed; metadata and license text read |
| `@babel/generator` | `7.29.7` | `MIT` | Installed; metadata and license text read |
| `@babel/helper-compilation-targets` | `7.29.7` | `MIT` | Installed; metadata and license text read |
| `@babel/helper-globals` | `7.29.7` | `MIT` | Installed; metadata and license text read |
| `@babel/helper-module-imports` | `7.29.7` | `MIT` | Installed; metadata and license text read |
| `@babel/helper-module-transforms` | `7.29.7` | `MIT` | Installed; metadata and license text read |
| `@babel/helper-plugin-utils` | `7.29.7` | `MIT` | Installed; metadata and license text read |
| `@babel/helper-string-parser` | `7.29.7` | `MIT` | Installed; metadata and license text read |
| `@babel/helper-validator-identifier` | `7.29.7` | `MIT` | Installed; metadata and license text read |
| `@babel/helper-validator-option` | `7.29.7` | `MIT` | Installed; metadata and license text read |
| `@babel/helpers` | `7.29.7` | `MIT` | Installed; metadata and license text read |
| `@babel/parser` | `7.29.7` | `MIT` | Installed; metadata and license text read |
| `@babel/plugin-transform-react-jsx-self` | `7.29.7` | `MIT` | Installed; metadata and license text read |
| `@babel/plugin-transform-react-jsx-source` | `7.29.7` | `MIT` | Installed; metadata and license text read |
| `@babel/template` | `7.29.7` | `MIT` | Installed; metadata and license text read |
| `@babel/traverse` | `7.29.7` | `MIT` | Installed; metadata and license text read |
| `@babel/types` | `7.29.7` | `MIT` | Installed; metadata and license text read |
| `@esbuild/aix-ppc64` | `0.25.12` | `MIT` | **Not installed**; lockfile declaration only; package text not verified locally |
| `@esbuild/android-arm` | `0.25.12` | `MIT` | **Not installed**; lockfile declaration only; package text not verified locally |
| `@esbuild/android-arm64` | `0.25.12` | `MIT` | **Not installed**; lockfile declaration only; package text not verified locally |
| `@esbuild/android-x64` | `0.25.12` | `MIT` | **Not installed**; lockfile declaration only; package text not verified locally |
| `@esbuild/darwin-arm64` | `0.25.12` | `MIT` | **Metadata only**; installed package has no license file |
| `@esbuild/darwin-x64` | `0.25.12` | `MIT` | **Not installed**; lockfile declaration only; package text not verified locally |
| `@esbuild/freebsd-arm64` | `0.25.12` | `MIT` | **Not installed**; lockfile declaration only; package text not verified locally |
| `@esbuild/freebsd-x64` | `0.25.12` | `MIT` | **Not installed**; lockfile declaration only; package text not verified locally |
| `@esbuild/linux-arm` | `0.25.12` | `MIT` | **Not installed**; lockfile declaration only; package text not verified locally |
| `@esbuild/linux-arm64` | `0.25.12` | `MIT` | **Not installed**; lockfile declaration only; package text not verified locally |
| `@esbuild/linux-ia32` | `0.25.12` | `MIT` | **Not installed**; lockfile declaration only; package text not verified locally |
| `@esbuild/linux-loong64` | `0.25.12` | `MIT` | **Not installed**; lockfile declaration only; package text not verified locally |
| `@esbuild/linux-mips64el` | `0.25.12` | `MIT` | **Not installed**; lockfile declaration only; package text not verified locally |
| `@esbuild/linux-ppc64` | `0.25.12` | `MIT` | **Not installed**; lockfile declaration only; package text not verified locally |
| `@esbuild/linux-riscv64` | `0.25.12` | `MIT` | **Not installed**; lockfile declaration only; package text not verified locally |
| `@esbuild/linux-s390x` | `0.25.12` | `MIT` | **Not installed**; lockfile declaration only; package text not verified locally |
| `@esbuild/linux-x64` | `0.25.12` | `MIT` | **Not installed**; lockfile declaration only; package text not verified locally |
| `@esbuild/netbsd-arm64` | `0.25.12` | `MIT` | **Not installed**; lockfile declaration only; package text not verified locally |
| `@esbuild/netbsd-x64` | `0.25.12` | `MIT` | **Not installed**; lockfile declaration only; package text not verified locally |
| `@esbuild/openbsd-arm64` | `0.25.12` | `MIT` | **Not installed**; lockfile declaration only; package text not verified locally |
| `@esbuild/openbsd-x64` | `0.25.12` | `MIT` | **Not installed**; lockfile declaration only; package text not verified locally |
| `@esbuild/openharmony-arm64` | `0.25.12` | `MIT` | **Not installed**; lockfile declaration only; package text not verified locally |
| `@esbuild/sunos-x64` | `0.25.12` | `MIT` | **Not installed**; lockfile declaration only; package text not verified locally |
| `@esbuild/win32-arm64` | `0.25.12` | `MIT` | **Not installed**; lockfile declaration only; package text not verified locally |
| `@esbuild/win32-ia32` | `0.25.12` | `MIT` | **Not installed**; lockfile declaration only; package text not verified locally |
| `@esbuild/win32-x64` | `0.25.12` | `MIT` | **Not installed**; lockfile declaration only; package text not verified locally |
| `@fontsource/overpass-mono` | `5.3.0` | `OFL-1.1` | Installed; metadata and license text read |
| `@jridgewell/gen-mapping` | `0.3.13` | `MIT` | Installed; metadata and license text read |
| `@jridgewell/remapping` | `2.3.5` | `MIT` | Installed; metadata and license text read |
| `@jridgewell/resolve-uri` | `3.1.2` | `MIT` | Installed; metadata and license text read |
| `@jridgewell/sourcemap-codec` | `1.5.5` | `MIT` | Installed; metadata and license text read |
| `@jridgewell/trace-mapping` | `0.3.31` | `MIT` | Installed; metadata and license text read |
| `@phosphor-icons/react` | `2.1.10` | `MIT` | Installed; metadata and license text read |
| `@rolldown/pluginutils` | `1.0.0-beta.38` | `MIT` | Installed; metadata and license text read |
| `@rollup/rollup-android-arm-eabi` | `4.62.2` | `MIT` | **Not installed**; lockfile declaration only; package text not verified locally |
| `@rollup/rollup-android-arm64` | `4.62.2` | `MIT` | **Not installed**; lockfile declaration only; package text not verified locally |
| `@rollup/rollup-darwin-arm64` | `4.62.2` | `MIT` | **Metadata only**; installed package has no license file |
| `@rollup/rollup-darwin-x64` | `4.62.2` | `MIT` | **Not installed**; lockfile declaration only; package text not verified locally |
| `@rollup/rollup-freebsd-arm64` | `4.62.2` | `MIT` | **Not installed**; lockfile declaration only; package text not verified locally |
| `@rollup/rollup-freebsd-x64` | `4.62.2` | `MIT` | **Not installed**; lockfile declaration only; package text not verified locally |
| `@rollup/rollup-linux-arm-gnueabihf` | `4.62.2` | `MIT` | **Not installed**; lockfile declaration only; package text not verified locally |
| `@rollup/rollup-linux-arm-musleabihf` | `4.62.2` | `MIT` | **Not installed**; lockfile declaration only; package text not verified locally |
| `@rollup/rollup-linux-arm64-gnu` | `4.62.2` | `MIT` | **Not installed**; lockfile declaration only; package text not verified locally |
| `@rollup/rollup-linux-arm64-musl` | `4.62.2` | `MIT` | **Not installed**; lockfile declaration only; package text not verified locally |
| `@rollup/rollup-linux-loong64-gnu` | `4.62.2` | `MIT` | **Not installed**; lockfile declaration only; package text not verified locally |
| `@rollup/rollup-linux-loong64-musl` | `4.62.2` | `MIT` | **Not installed**; lockfile declaration only; package text not verified locally |
| `@rollup/rollup-linux-ppc64-gnu` | `4.62.2` | `MIT` | **Not installed**; lockfile declaration only; package text not verified locally |
| `@rollup/rollup-linux-ppc64-musl` | `4.62.2` | `MIT` | **Not installed**; lockfile declaration only; package text not verified locally |
| `@rollup/rollup-linux-riscv64-gnu` | `4.62.2` | `MIT` | **Not installed**; lockfile declaration only; package text not verified locally |
| `@rollup/rollup-linux-riscv64-musl` | `4.62.2` | `MIT` | **Not installed**; lockfile declaration only; package text not verified locally |
| `@rollup/rollup-linux-s390x-gnu` | `4.62.2` | `MIT` | **Not installed**; lockfile declaration only; package text not verified locally |
| `@rollup/rollup-linux-x64-gnu` | `4.62.2` | `MIT` | **Not installed**; lockfile declaration only; package text not verified locally |
| `@rollup/rollup-linux-x64-musl` | `4.62.2` | `MIT` | **Not installed**; lockfile declaration only; package text not verified locally |
| `@rollup/rollup-openbsd-x64` | `4.62.2` | `MIT` | **Not installed**; lockfile declaration only; package text not verified locally |
| `@rollup/rollup-openharmony-arm64` | `4.62.2` | `MIT` | **Not installed**; lockfile declaration only; package text not verified locally |
| `@rollup/rollup-win32-arm64-msvc` | `4.62.2` | `MIT` | **Not installed**; lockfile declaration only; package text not verified locally |
| `@rollup/rollup-win32-ia32-msvc` | `4.62.2` | `MIT` | **Not installed**; lockfile declaration only; package text not verified locally |
| `@rollup/rollup-win32-x64-gnu` | `4.62.2` | `MIT` | **Not installed**; lockfile declaration only; package text not verified locally |
| `@rollup/rollup-win32-x64-msvc` | `4.62.2` | `MIT` | **Not installed**; lockfile declaration only; package text not verified locally |
| `@types/babel__core` | `7.20.5` | `MIT` | Installed; metadata and license text read |
| `@types/babel__generator` | `7.27.0` | `MIT` | Installed; metadata and license text read |
| `@types/babel__template` | `7.4.4` | `MIT` | Installed; metadata and license text read |
| `@types/babel__traverse` | `7.28.0` | `MIT` | Installed; metadata and license text read |
| `@types/estree` | `1.0.9` | `MIT` | Installed; metadata and license text read |
| `@vitejs/plugin-react` | `5.0.4` | `MIT` | Installed; metadata and license text read |
| `baseline-browser-mapping` | `2.10.42` | `Apache-2.0` | Installed; metadata and license text read |
| `browserslist` | `4.28.5` | `MIT` | Installed; metadata and license text read |
| `caniuse-lite` | `1.0.30001803` | `CC-BY-4.0` | Installed; metadata and license text read |
| `convert-source-map` | `2.0.0` | `MIT` | Installed; metadata and license text read |
| `debug` | `4.4.3` | `MIT` | Installed; metadata and license text read |
| `electron-to-chromium` | `1.5.389` | `ISC` | Installed; metadata and license text read |
| `esbuild` | `0.25.12` | `MIT` | Installed; metadata and license text read |
| `escalade` | `3.2.0` | `MIT` | Installed; metadata and license text read |
| `fdir` | `6.5.0` | `MIT` | Installed; metadata and license text read |
| `fsevents` | `2.3.3` | `MIT` | Installed; metadata and license text read |
| `gensync` | `1.0.0-beta.2` | `MIT` | Installed; metadata and license text read |
| `js-tokens` | `4.0.0` | `MIT` | Installed; metadata and license text read |
| `jsesc` | `3.1.0` | `MIT` | Installed; metadata and license text read |
| `json5` | `2.2.3` | `MIT` | Installed; metadata and license text read |
| `lru-cache` | `5.1.1` | `ISC` | Installed; metadata and license text read |
| `ms` | `2.1.3` | `MIT` | Installed; metadata and license text read |
| `nanoid` | `3.3.15` | `MIT` | Installed; metadata and license text read |
| `node-releases` | `2.0.50` | `MIT` | Installed; metadata and license text read |
| `picocolors` | `1.1.1` | `ISC` | Installed; metadata and license text read |
| `picomatch` | `4.0.5` | `MIT` | Installed; metadata and license text read |
| `postcss` | `8.5.16` | `MIT` | Installed; metadata and license text read |
| `react` | `19.2.0` | `MIT` | Installed; metadata and license text read |
| `react-dom` | `19.2.0` | `MIT` | Installed; metadata and license text read |
| `react-refresh` | `0.17.0` | `MIT` | Installed; metadata and license text read |
| `rollup` | `4.62.2` | `MIT` | Installed; metadata and license text read |
| `scheduler` | `0.27.0` | `MIT` | Installed; metadata and license text read |
| `semver` | `6.3.1` | `ISC` | Installed; metadata and license text read |
| `source-map-js` | `1.2.1` | `BSD-3-Clause` | Installed; metadata and license text read |
| `three` | `0.186.1` | `MIT` | Installed; metadata and license text read |
| `tinyglobby` | `0.2.17` | `MIT` | Installed; metadata and license text read |
| `update-browserslist-db` | `1.2.3` | `MIT` | Installed; metadata and license text read |
| `vite` | `6.4.2` | `MIT` | Installed; metadata and license text read |
| `yallist` | `3.1.1` | `ISC` | Installed; metadata and license text read |

The two installed native platform packages with metadata-only evidence are `@esbuild/darwin-arm64` and `@rollup/rollup-darwin-arm64`. Both declare MIT. Their own installed package directories contain no license file. The full Vite build-tool artifact also contains third-party notices in its installed `LICENSE.md`; only the complete MIT blocks for the helpers actually emitted to the browser are copied here.

## Python tools

These packages are not shipped in the web app. The user installs each tool's requirements into a separate virtual environment. The audit queried the preparation, detection and labels environments with each environment's own Python using `-I -B` and `importlib.metadata`. It read the `License`, `License-Expression` and license classifier fields, and the installed distribution's license or notice files. No packages were imported or installed.

The four files contain 70 pinned rows, or 67 distinct package/version pairs. There are 69 exact installed row matches (four of them, `opencv-python`, `onnx`, `protobuf` and `ml_dtypes`, in a fresh test environment made later the same day). One row, `numpy==2.5.3` in the model-inspection file, has no exact match. An exact version match is separate from finding a complete license text. Missing texts and notice-only evidence are marked below. The table shows the package's own declaration; bundled component notices can add other terms.

### [scripts/requirements-preparation.txt](scripts/requirements-preparation.txt)

| Package | Pinned version | License declaration or identified license | Verification |
| --- | --- | --- | --- |
| `numpy` | `2.3.1` | `BSD-3-Clause` | Exact version and license text checked in labels, preparation |
| `Pillow` | `11.3.0` | `MIT-CMU` | Exact version and license text checked in labels, preparation |

### [scripts/california/requirements-av2.txt](scripts/california/requirements-av2.txt)

| Package | Pinned version | License declaration or identified license | Verification |
| --- | --- | --- | --- |
| `numpy` | `2.3.1` | `BSD-3-Clause` | Exact version and license text checked in labels, preparation |
| `Pillow` | `11.3.0` | `MIT-CMU` | Exact version and license text checked in labels, preparation |
| `opencv-python` | `4.13.0.92` | Apache-2.0 (metadata says Apache 2.0); separate third-party license file for bundled components | Exact version and license text checked in a test environment |
| `pyarrow` | `20.0.0` | Apache-2.0 | Exact version and license text checked in labels |
| `scipy` | `1.16.0` | `BSD-3-Clause` | Exact version and license text checked in labels |

### [scripts/requirements-model-inspection.txt](scripts/requirements-model-inspection.txt)

| Package | Pinned version | License declaration or identified license | Verification |
| --- | --- | --- | --- |
| `onnx` | `1.23.1` | `Apache-2.0` | Exact version and license text checked in a test environment |
| `numpy` | `2.5.3` | Not verified locally; no license inferred | **not verified locally (installed 2.3.1 / 2.5.4)** |
| `protobuf` | `7.36.2` | BSD-3-Clause (metadata says 3-Clause BSD License) | Exact version and license text checked in a test environment |
| `typing_extensions` | `4.16.0` | `PSF-2.0` | Exact version and license text checked in detection |
| `ml_dtypes` | `0.6.0` | `Apache-2.0`; bundled Eigen license file | Exact version and license text checked in a test environment |

### [scripts/requirements-detection.txt](scripts/requirements-detection.txt)

| Package | Pinned version | License declaration or identified license | Verification |
| --- | --- | --- | --- |
| `annotated-doc` | `0.0.5` | `MIT` | Exact version and license text checked in detection |
| `annotated-types` | `0.8.0` | `MIT` | Exact version and license text checked in detection |
| `anyio` | `4.15.1` | `MIT` | Exact version and license text checked in detection |
| `av` | `19.0.1` | `BSD-3-Clause` | Exact version and license text checked in detection |
| `certifi` | `2026.7.22` | `MPL-2.0` | Exact version in detection; notice read; full MPL-2.0 terms not available locally |
| `charset-normalizer` | `3.5.2` | `MIT` | Exact version and license text checked in detection |
| `click` | `8.5.0` | `BSD-3-Clause` | Exact version and license text checked in detection |
| `contourpy` | `1.4.0` | `BSD-3-Clause` | Exact version and license text checked in detection |
| `cycler` | `0.12.1` | `BSD-3-Clause` | Exact version and license text checked in detection |
| `defusedxml` | `0.7.1` | PSF-2.0 (metadata says PSFL) | Exact version and license text checked in detection |
| `filelock` | `4.1.1` | `MIT` | Exact version and license text checked in detection |
| `fonttools` | `4.66.1` | `MIT` | Exact version and license text checked in detection |
| `fsspec` | `2026.9.0` | `BSD-3-Clause` | Exact version and license text checked in detection |
| `h11` | `0.16.0` | `MIT` | Exact version and license text checked in detection |
| `hf-xet` | `1.7.0` | `Apache-2.0` | Exact version and license text checked in detection |
| `httpcore2` | `2.13.1` | `BSD-3-Clause` | Exact version and license text checked in detection |
| `httpx2` | `2.13.1` | `BSD-3-Clause` | Exact version and license text checked in detection |
| `huggingface_hub` | `2.2.0` | `Apache-2.0` | Exact version and license text checked in detection |
| `idna` | `3.20` | `BSD-3-Clause` | Exact version and license text checked in detection |
| `Jinja2` | `3.1.6` | `BSD-3-Clause` | Exact version and license text checked in detection |
| `kiwisolver` | `1.5.1` | `BSD-3-Clause` | Exact version and license text checked in detection |
| `markdown-it-py` | `4.2.0` | `MIT` | Exact version and license text checked in detection |
| `MarkupSafe` | `3.0.4` | `BSD-3-Clause` | Exact version and license text checked in detection |
| `matplotlib` | `3.11.2` | **Custom / non-standard:** Matplotlib license agreement, versions 1.3.0 and later | Exact version and license text checked in detection |
| `mdurl` | `0.1.2` | `MIT` | Exact version and license text checked in detection |
| `mpmath` | `1.3.0` | BSD-3-Clause (metadata says BSD) | Exact version and license text checked in detection |
| `networkx` | `3.7` | `BSD-3-Clause` | Exact version and license text checked in detection |
| `numpy` | `2.5.4` | `BSD-3-Clause AND 0BSD AND MIT AND Zlib AND CC0-1.0` | Exact version and license text checked in detection |
| `packaging` | `26.3` | `Apache-2.0 OR BSD-2-Clause` | Exact version and license text checked in detection |
| `pillow` | `12.3.0` | `MIT-CMU` | Exact version and license text checked in detection |
| `pydantic` | `2.14.1` | `MIT` | Exact version and license text checked in detection |
| `pydantic_core` | `2.50.1` | `MIT` | Exact version and license text checked in detection |
| `pyDeprecate` | `0.9.0` | `Apache-2.0` | Exact version and license text checked in detection |
| `Pygments` | `2.21.0` | `BSD-2-Clause` | Exact version and license text checked in detection |
| `pyparsing` | `3.3.3` | `MIT` | Exact version and license text checked in detection |
| `python-dateutil` | `2.9.0.post0` | "Dual License" (metadata); Apache-2.0 and BSD-3-Clause texts | Exact version and license text checked in detection |
| `PyYAML` | `6.0.3` | `MIT` | Exact version and license text checked in detection |
| `regex` | `2026.9.29` | `Apache-2.0 AND CNRI-Python` | Exact version and license text checked in detection |
| `requests` | `2.34.2` | `Apache-2.0` | Exact version and license text checked in detection |
| `rfdetr` | `1.11.2` | Apache-2.0 | Exact version and license text checked in detection |
| `rich` | `15.0.0` | `MIT` | Exact version and license text checked in detection |
| `safetensors` | `0.8.0` | `Apache-2.0` | Exact version and license text checked in detection |
| `scipy` | `1.18.1` | `BSD-3-Clause` | Exact version and license text checked in detection |
| `setuptools` | `84.0.0` | `MIT` | Exact version and license text checked in detection |
| `shellingham` | `1.5.4` | ISC | Exact version and license text checked in detection |
| `six` | `1.17.0` | `MIT` | Exact version and license text checked in detection |
| `supervision` | `0.30.9` | `MIT` | Exact version and license text checked in detection |
| `sympy` | `1.14.0` | BSD-3-Clause; bundled MIT component (metadata says BSD) | Exact version and license text checked in detection |
| `tokenizers` | `0.23.3` | Apache Software License classifier only; **full license text not verified** | Exact version in detection; metadata only; license text not verified |
| `torch` | `2.14.1` | `Apache-2.0 AND Apache-2.0 WITH LLVM-exception AND BSD-2-Clause AND BSD-3-Clause AND BSL-1.0 AND MIT` | Exact version and license text checked in detection |
| `torchvision` | `0.29.1` | BSD-3-Clause (metadata says BSD) | Exact version and license text checked in detection |
| `tqdm` | `4.70.1` | `MPL-2.0 AND MIT` | Exact version in detection; notice read; full MPL-2.0 terms not available locally |
| `transformers` | `5.19.0` | Apache-2.0 | Exact version and license text checked in detection |
| `truststore` | `0.10.4` | `MIT` | Exact version and license text checked in detection |
| `typer` | `0.27.3` | `MIT` | Exact version and license text checked in detection |
| `typing-inspection` | `0.4.4` | `MIT` | Exact version and license text checked in detection |
| `typing_extensions` | `4.16.0` | `PSF-2.0` | Exact version and license text checked in detection |
| `urllib3` | `2.8.0` | `MIT` | Exact version and license text checked in detection |

### Optional tools mentioned by the documentation

The README's test setup installs these packages outside the four requirements files. Their metadata and license files were read in a fresh test environment on 2026-10-16.

| Tool | Version mentioned in the docs | License and local verification |
| --- | --- | --- |
| `onnxruntime` | `1.24.4` | MIT (metadata and license file) |
| `pytest` | `8.4.1` | MIT (metadata and license file) |
| `pandas` | `2.3.1` | BSD-3-Clause (metadata says BSD 3-Clause License; license file read) |

### Additional Python license terms and gaps

- **NumPy and SciPy:** BSD package terms coexist with bundled OpenBLAS/LAPACK notices, `GPL-3.0-or-later WITH GCC-exception-3.1` runtime terms, and `LGPL-2.1-or-later` libquadmath notices. The LGPL notices link to the terms; the full LGPL text was not found in those notices. Their wheel terms cannot be reduced to a BSD label.
- **certifi and tqdm:** the installed declarations include `MPL-2.0`. tqdm also declares MIT. Their files contain MPL notices, but not the complete MPL-2.0 terms. The license declarations were checked; the complete MPL texts were not verified locally.
- **setuptools:** its vendored autocommand 2.2.2 has LGPLv3 text. The top-level MIT declaration does not replace that component's terms.
- **matplotlib:** its own license is a custom Matplotlib agreement. Its broader PSF classifier does not replace that text. Bundled notices also cover OFL-1.1 fonts, FreeType (`FTL OR GPL-2.0-or-later`), BaKoMa, Bitstream Charter, Qhull, Yorick and other component terms. There is no single verified SPDX expression for the whole bundle.
- **fonttools:** MIT package terms coexist with an OFL-1.1 font license.
- **pyarrow:** Apache-2.0 package terms coexist with many component notices, including an LLVM exception, BSD, MIT, ZPL and custom terms.
- **Pillow:** the long bundled license contains native component terms. Its XZ notice distinguishes public-domain liblzma from GPL/LGPL tools and build files. This does not establish that those tools are linked into the wheel.
- **python-dateutil:** metadata says "Dual License". The installed file contains Apache-2.0 and BSD-3-Clause text and states that the BSD license applies to all code. It is not silently rewritten into an assumed SPDX choice.
- **PyAV (`av`):** the package declares BSD-3-Clause. Its metadata says FFmpeg is bundled, and installed file names include FFmpeg, x264 and x265 libraries. Their full native-component licensing is **not verified locally** by the package's license file.
- **tokenizers:** the exact installed version has an Apache license classifier, but no installed full license text was found. The full license is **not verified locally**.

These are local package-document checks. They do not establish complete terms for redistributing Python wheels or their native libraries. The Python tools are user-installed and are not part of the web bundle.

## Models and hosted services

| Model or service | Use and exact identity | License / terms | Evidence and verification |
| --- | --- | --- | --- |
| RF-DETR Medium | `rfdetr==1.11.2`; offline detector. Weights are not tracked in Git. Expected weights SHA256: `749ff6071828aaffac63e204c4f4135ed3d6cdae4d702e086c360edc3b5768c8`. | Apache-2.0 for the package; the installed package separately designates Medium weights Apache 2.0. | [Full package license](licenses/Apache-2.0-rfdetr.txt); [pinned requirements and weight identity](scripts/requirements-detection.txt). Package statements are quoted below. Weight bytes and the binary checksum were not re-verified in this audit. |
| NVIDIA C-FoundationStereo-S | Offline reconstruction; `fixed320x736 v2`; weights are not tracked in Git. | NVIDIA Open Model License (cited; not re-verified) | [Existing README citation](README.md#development-evaluation), [version and model contract](scripts/sponsors/README.md#nvidia), [model card](https://huggingface.co/nvidia/c-foundationstereo-s), [license URL already cited by the README](https://www.nvidia.com/en-us/agreements/enterprise-software/nvidia-open-model-license/). These links were not opened. |
| NVIDIA NV-Grounding-DINO | Prepared `scripts/cloud` tooling; the job never ran. No model outputs exist. | NVIDIA Open Model License (cited; not re-verified) | [Tracked license notes](scripts/cloud/README.md#one-nv-grounding-dino-job-tao-720-container). The tracked docs name the license; the same license URL appears in the main README citation above. No NV-Grounding-DINO-specific license URL is supplied by its tracked README section. |
| NVIDIA TAO Toolkit 7.2.0 container | Tooling prepared for NV-Grounding-DINO; not run or redistributed. | NVIDIA Software License Agreement and Product-Specific Terms (cited; not re-verified) | [Tracked license notes](scripts/cloud/README.md#one-nv-grounding-dino-job-tao-720-container). No direct terms URL appears in that section. |
| BERT (`google-bert/bert-base-uncased`) | Named dependency of the prepared NV-Grounding-DINO job; job never ran. | Apache-2.0 (cited; not re-verified) | [Tracked license notes](scripts/cloud/README.md#one-nv-grounding-dino-job-tao-720-container). No publisher license file was checked. |
| NVIDIA Nemotron 3.5 Lightning through Nebius Token Factory | Hosted API; model id `nvidia/Nemotron-3_5-Lightning`. No weights downloaded. | Model: OpenMDW License Agreement 1.1 ([license text](https://github.com/OpenMDW/OpenMDW/blob/main/1.1/LICENSE.OpenMDW-1.1)), per NVIDIA's NGC governing terms. Service: [Nebius Token Factory terms of service](https://docs.tokenfactory.nebius.com/legal/terms-of-service) (effective 2026-08-20). | Checked online on 2026-10-16. Token Factory terms: the customer owns Outputs (10(b)); the customer must follow the model's license (11(c)); Input and Output may be collected to train speculative-decoding models, with an opt-out (7); Output may not be used for military, surveillance or biometric purposes (5(f)). The terms' preamble says a different agreement covers some customers from 2026-09-28; which one covers this account was not checked. [Model constant](worker/nemotron.js), [application integration](README.md#nvidia-nemotron-assistant-nebius-token-factory), [API references](scripts/sponsors/README.md#primary-api-references). |
| Nebius AI Cloud | Offline-job tooling and saved inference workflow; no service call was made for this audit. | Not checked | [Tracked job tooling](scripts/cloud/README.md). Nebius AI Cloud's service terms were not read. |

RF-DETR's installed `rfdetr-1.11.2.dist-info/licenses/LICENSE` is the Apache 2.0 text with `Copyright 2025 Roboflow, Inc.`. That LICENSE file does not name Medium weights separately. The installed `METADATA` README body provides the additional model statement:

> The open-source `rfdetr` package and Apache-designated model weights are licensed under Apache License 2.0.

The statement is at `rfdetr-1.11.2.dist-info/METADATA:418`. Its model table at line 329 lists `M`, `RFDETRMedium`, `rfdetr-medium` and `Apache 2.0` together. The package's `rfdetr/config.py:737` declares `Apache-2.0`; its Medium configuration names `rf-detr-medium.pth`. The weight URL in `rfdetr/assets/model_weights.py:227` matches the tracked requirements comment. The expected SHA256 above comes from that tracked comment; the package's download catalog uses a different checksum format. This audit did not read or hash model weights.

The C-FoundationStereo, NV-Grounding-DINO, TAO and BERT rows repeat existing repository citations. They are not a fresh review of publisher terms. The Nemotron and Token Factory row was checked online; no license is guessed for Nebius AI Cloud.

## Data

### Argoverse 2

Attribution: **© 2021 Argo AI, LLC. Argoverse 2 Sensor Dataset.** License: **Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International (CC BY-NC-SA 4.0)**. See the [publisher](https://www.argoverse.org/av2.html), [dataset terms link](https://www.argoverse.org/about.html#terms-of-use) and [license](https://creativecommons.org/licenses/by-nc-sa/4.0/). The terms page was checked online on 2026-10-16. Besides the license, it says: use must be non-commercial (not primarily intended for commercial advantage or monetary compensation); do not try to identify people in the data; do not use Argo's name, logo or trademarks without written permission (attribution excepted); and Argo may end the license for use that breaks these terms. The av2 software API is MIT. The complete CC BY-NC-SA 4.0 text is at the license link; it is not copied here.

On the public issue [argoverse/av2-api#340](https://github.com/argoverse/av2-api/issues/340), this project described its use (processing selected AV2 images, LiDAR, calibration and maps into scene geometry and evaluation results; showing attributed images and derived geometry in a public demo and video; open-source application code; entering a cash-prize hackathon with the organizers' judging and promotional use). An Argoverse maintainer replied on 2026-10-05 that this use is fine under the Argoverse 2 terms, that the code may be MIT, Apache 2.0 or another license, and that only the data side is covered by ShareAlike. This clarifies the existing terms for that described use; it is not a waiver of the license conditions or a permission for other commercial use.

No raw AV2 file is in Git. Four prepared scenes derived from AV2 sensor data are tracked: `public/scenes/av2-pao-8c52d911-2658849`, `av2-pao-c780d53a-2069452`, `av2-pao-e1d68dde-5987490` and `av2-pao-fd5c6932-2659589`. Each holds a camera image, LiDAR points, measured ground surfaces and map-based geometry from one log, plus an `ATTRIBUTION.md` with the credit, license, change notes and log id. Each scene's `scene.json` records the sha256 of the original local source record (`sourceHashes.license`); that record names local folders, so `ATTRIBUTION.md` replaces it. These scenes, and the source-derived values in the tracked files below, remain under CC BY-NC-SA 4.0: retain attribution, use them non-commercially, share adaptations under the same license, and note changes. This covers the data-derived portions, including identifiers, source/derived-file hashes, measured results and transformed coordinates. It does not change the license of umbra's own code (Apache-2.0).

Changes made by umbra include selecting source observations, converting mapped geometry and annotations into scenario coordinates, registering LiDAR sweeps, preparing ground surfaces with explicit holes and calibration, deriving shape and visibility measurements, and reporting development/test results. No publisher endorsement is implied. Frozen legacy source strings keep their existing identifiers.

The tracked-file inventory below was made from AV2 log UUIDs, site identifiers, source hashes, coordinates and measured scalar values. Tests with analytic geometry can still contain real identifiers or measurements.

| Tracked file | Source-derived values or reason for review |
| --- | --- |
| [README.md](README.md) | Measured results, scene counts and calibration summaries. |
| [scripts/california/README.md](scripts/california/README.md) | A real scene identifier in the catalog example. |
| [public/scenes/index.json](public/scenes/index.json) | Scene identifiers and hashes of the derived scene files. |
| `.local/evaluation/` frozen method files (30 files; `git ls-files .local` lists them and `commit-map.txt`) | Log and site identifiers, timestamps, coordinates, hashes of derived artifacts and measured values in the frozen split, protocols, lock records and reviews. |
| [scripts/california/build_av2_scene.py](scripts/california/build_av2_scene.py) | Real dataset identifiers and observation metadata. |
| [scripts/california/car_shape.py](scripts/california/car_shape.py) | Real dataset identifiers and metadata hashes. |
| [scripts/california/measured_ground.py](scripts/california/measured_ground.py) | Real dataset identifiers. |
| [scripts/california/prepare_av2_ground.py](scripts/california/prepare_av2_ground.py) | Real dataset identifiers. |
| [scripts/evaluation/labels/build_validation.py](scripts/evaluation/labels/build_validation.py) | Real annotation identifiers and case metadata. |
| [scripts/evaluation/labels/common.py](scripts/evaluation/labels/common.py) | Real dataset identifiers. |
| [scripts/evaluation/labels/select_cases.py](scripts/evaluation/labels/select_cases.py) | Real dataset identifiers and timestamps. |
| [scripts/evaluation/m7/anchors-2/labels-manifest.json](scripts/evaluation/m7/anchors-2/labels-manifest.json) | Real dataset identifiers and hashes of derived artifacts. |
| [scripts/evaluation/m7/anchors-2/runs.json](scripts/evaluation/m7/anchors-2/runs.json) | Real dataset identifiers and hashes of derived artifacts. |
| [scripts/evaluation/m7/common.py](scripts/evaluation/m7/common.py) | A hash of site-selection metadata. |
| [scripts/evaluation/m7/run_heldout.py](scripts/evaluation/m7/run_heldout.py) | A measured uncertainty value and source metadata. |
| [scripts/evaluation/pao-baseline.mjs](scripts/evaluation/pao-baseline.mjs) | A real scene identifier and measured expected results. |
| [tests/av2-ground.test.mjs](tests/av2-ground.test.mjs) | Measured numeric test values and source/artifact hashes. |
| [tests/california/test_build_av2_scene.py](tests/california/test_build_av2_scene.py) | Real identifiers, timestamps and source/artifact hashes. |
| [tests/california/test_car_shape.py](tests/california/test_car_shape.py) | Real identifiers reused in an otherwise synthetic test. |
| [tests/california/test_measure_paint_edge.py](tests/california/test_measure_paint_edge.py) | Measured LiDAR radii. |
| [tests/california/test_prepare_av2_ground.py](tests/california/test_prepare_av2_ground.py) | Real identifiers and measured regression values. |
| [tests/california/test_suggest_av2_placements.py](tests/california/test_suggest_av2_placements.py) | Measured numeric test values and source/artifact hashes. |
| [tests/car-shape.test.mjs](tests/car-shape.test.mjs) | Measured vehicle dimensions. |
| [tests/evaluation/test_labels_coverage_v2.py](tests/evaluation/test_labels_coverage_v2.py) | A real annotation identifier prefix in a synthetic test. |
| [tests/evaluation/test_labels_tools.py](tests/evaluation/test_labels_tools.py) | A real log identifier prefix in a synthetic test. |
| [tests/evaluation/test_m7_child_preflight.py](tests/evaluation/test_m7_child_preflight.py) | Real log identifier prefixes in a synthetic test. |
| [tests/geometry.test.mjs](tests/geometry.test.mjs) | Real site identifiers, measured grid indices and scene results. |
| [tests/map-layers.test.mjs](tests/map-layers.test.mjs) | Measured map category counts. |
| [tests/metric-f.test.mjs](tests/metric-f.test.mjs) | Measured result counts. |
| [tests/pao-baseline.test.mjs](tests/pao-baseline.test.mjs) | Measured results and source/artifact hashes. |
| [tests/scenario-assistant.test.mjs](tests/scenario-assistant.test.mjs) | Measured facts, source pixel coordinates and result ranges. |

The following files are also included conservatively. The origin of their numeric literals, real-looking UUIDs or evaluation-metadata hashes could not be proved from local tracked text. Treat any source-derived portions under the same dataset terms until their origin is resolved; this does not relicense the surrounding code.

| Tracked file | Source-derived values or reason for review |
| --- | --- |
| [tests/driver-viewport.test.mjs](tests/driver-viewport.test.mjs) | Calibration and pixel values whose source is not recorded. |
| [tests/indexed-scene.test.mjs](tests/indexed-scene.test.mjs) | A real-looking log identifier in synthetic test geometry. |
| [tests/mapped-fixture.mjs](tests/mapped-fixture.mjs) | A real-looking log identifier and uncertainty value in a synthetic test. |
| [tests/metric-e-grid.test.mjs](tests/metric-e-grid.test.mjs) | An expected result from an optional real-scene test. |
| [tests/evaluation/test_metric_f_car_top.py](tests/evaluation/test_metric_f_car_top.py) | A synthetic regression linked to a real observed failure. |
| [scripts/evaluation/m7/anchors-2/labels-addendum-anchor.json](scripts/evaluation/m7/anchors-2/labels-addendum-anchor.json) | A hash of evaluation metadata; underlying contents were not inspected. |
| [scripts/evaluation/m7/anchors-2/pao-lock-anchor.json](scripts/evaluation/m7/anchors-2/pao-lock-anchor.json) | A hash of evaluation metadata; underlying contents were not inspected. |
| [scripts/evaluation/m7/anchors/labels-addendum-anchor.json](scripts/evaluation/m7/anchors/labels-addendum-anchor.json) | A hash of evaluation metadata; underlying contents were not inspected. |
| [scripts/evaluation/pao-lock-anchor.json](scripts/evaluation/pao-lock-anchor.json) | A hash of evaluation metadata; underlying contents were not inspected. |
| [scripts/evaluation/m7/build_scene.py](scripts/evaluation/m7/build_scene.py) | A hash of evaluation metadata; underlying contents were not inspected. |
| [tests/evaluation/test_m7_lock2.py](tests/evaluation/test_m7_lock2.py) | Evaluation metadata hashes and a fitted uncertainty value. |

### DrivingStereo

The local publisher license says exactly **MIT License** and **Copyright (c) 2019 drivingstereo-dataset**. Its complete text is copied byte-for-byte in [licenses/MIT-drivingstereo.txt](licenses/MIT-drivingstereo.txt). The source is the [official DrivingStereo dataset](https://drivingstereo-dataset.github.io/). Its website, checked online on 2026-10-16, says the dataset "is released under the MIT license" and asks users who use it in research to cite the CVPR 2019 paper by Yang et al. One prepared scene is tracked: `public/scenes/2018-10-19-09-30-39_2018-10-19-09-40-40-716`, with the MIT `LICENSE`, the citation and change notes in its `ATTRIBUTION.md`. Its `estimated*.bin` files are a depth prediction by NVIDIA C-FoundationStereo-S; no model weights are tracked. No other DrivingStereo file is tracked.

Changes include image/depth cropping and resizing, calibration processing, depth/point preparation, pixel annotations, reconstruction comparisons and measured summaries. The following tracked files contain DrivingStereo-derived values.

| Tracked file | Source-derived values or reason for review |
| --- | --- |
| [README.md](README.md) | Measured results, scene counts and calibration summaries. |
| [scripts/benchmark-comparison-path.mjs](scripts/benchmark-comparison-path.mjs) | A distance from a recorded scene. |
| [scripts/evaluate-epipolar-alignment.py](scripts/evaluate-epipolar-alignment.py) | Real frame identifiers and metadata. |
| [scripts/evaluate-stereo-baseline.py](scripts/evaluate-stereo-baseline.py) | Real frame identifiers and metadata. |
| [scripts/evaluation/README.md](scripts/evaluation/README.md) | Hashes of dataset-derived reports and artifacts. |
| [tests/geometry.test.mjs](tests/geometry.test.mjs) | Real site identifiers, measured grid indices and scene results. |
| [tests/scenario-assistant.test.mjs](tests/scenario-assistant.test.mjs) | Measured facts, source pixel coordinates and result ranges. |
| [tests/scene-input.mjs](tests/scene-input.mjs) | Pixel annotations on a real source image. |
| [tests/suggestions.test.mjs](tests/suggestions.test.mjs) | Source pixel coordinates and measured summaries. |
| [tests/test_attach_nvidia.py](tests/test_attach_nvidia.py) | A real frame identifier and saved-prediction hash. |
| [tests/test_detect_vehicles_boxes.py](tests/test_detect_vehicles_boxes.py) | Real frame identifiers and metadata. |
| [tests/test_scene_preparation.py](tests/test_scene_preparation.py) | Real frame identifiers and possible source calibration values. |
| [tests/test_suggest_placements.py](tests/test_suggest_placements.py) | A measured crossing point and real frame metadata. |

The following files are included conservatively because their precise numeric origin is not established by the tracked text.

| Tracked file | Source-derived values or reason for review |
| --- | --- |
| [scripts/evaluate-stereo-offset.py](scripts/evaluate-stereo-offset.py) | An alignment constant whose source is not recorded. |
| [tests/driver-viewport.test.mjs](tests/driver-viewport.test.mjs) | Calibration and pixel values whose source is not recorded. |
| [scripts/cloud/prepare_nvdino_job.py](scripts/cloud/prepare_nvdino_job.py) | Recorded-image and chunk byte sizes and hashes; dataset identity not verified. |
| [scripts/cloud/README.md](scripts/cloud/README.md) | An actual image chunk size; dataset identity not verified. |
| [tests/cloud/test_nvdino_job.py](tests/cloud/test_nvdino_job.py) | Imported real image/chunk hashes; dataset identity and test-box origins not verified. |

## How this list was made

Date: **2026-10-16**. The package audit was offline; the AV2, Nemotron and Token Factory terms were checked online afterwards. It used `npm run build`, Vite's local build API with `write:false` to inspect emitted modules, SHA256 comparisons of emitted font files, `package-lock.json` plus installed npm metadata/license files, isolated `python -I -B` calls to `importlib.metadata`, and tracked-text searches with `git ls-files` and `git grep`. AV2 searches included UUID log ids and site ids from `tests/` and `scripts/`. DrivingStereo searches included frame ids, calibration values, image hashes and measured results. Every lockfile row was generated by a script; the Python tables were generated from the pinned requirements and local metadata inventory.

Not verified: the 49 absent optional npm artifacts; the two installed native npm artifacts' own missing license texts; the independent version of Vite's embedded CommonJS helper; one unavailable exact Python pin (numpy 2.5.3); OpenCV's bundled third-party component terms; tokenizers' full license text; the complete MPL texts for certifi/tqdm; full LGPL terms linked by scientific-package notices; PyAV's bundled FFmpeg/codec terms; complete wheel/native-library redistribution terms; the actual RF-DETR weight bytes/checksum; current C-FoundationStereo, NV-Grounding-DINO, TAO, BERT and Nebius AI Cloud terms; and the origin of the conservatively listed data literals. The detailed tables identify each unavailable package, pin and file. Text search cannot prove the origin of every unattributed scalar. Historical Git versions and ignored data artifacts were not inventoried. No sensor file was opened for inspection. The package audit made no install, model call or network request.
