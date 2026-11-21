<!---
Copyright 2020 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
-->

# What's in `docs/`

| Directory | Audience | Built & published? |
| --- | --- | --- |
| [`source/`](./source) | **Users** — installation, hardware guides, per-policy guides, tutorials | Yes — built by `doc-builder`, published to <https://huggingface.co/docs/lerobot>. The rest of this file explains how to build it. |
| [`intro/`](./intro/README.md) | **Contributors** — a map of the codebase: architecture, module-by-module walkthroughs, flowcharts of the runtime | No — read on GitHub |
| [`issue-deep-dives/`](./issue-deep-dives/README.md) | **Contributors** — real defects analysed end to end: root cause, fix, regression test | No — read on GitHub |

**"How does LeRobot work, and where do I change it?"** →
[`intro/README.md`](./intro/README.md). Twelve documents, heavy on Mermaid flowcharts, every
claim anchored to a file and symbol in `src/lerobot/`:

```
01 Project Overview        — mission, the five pillars, repo layout, the CLI surface
02 Architecture            — layers, dependency rules, the two universal data currencies
03 Configuration & CLI     — draccus, ChoiceRegistry, the two-phase --policy.path resolution
04 Robots & Teleoperators  — the Robot/Teleoperator contracts, calibration lifecycle
05 Cameras                 — the background reader thread pattern and its hazards
06 Motors & Calibration    — MotorsBus, normalization, the calibrate/write/read-back loop
07 LeRobotDataset          — the v3 format, delta timestamps, writing, streaming
08 Processor Pipeline      — EnvTransition, ProcessorStep, train/deploy parity
09 Policies                — PreTrainedPolicy, action chunking, the factory, PEFT
10 Training                — lerobot-train end to end, accelerate, checkpoints, the Hub
11 Inference & Rollout     — lerobot-eval, lerobot-rollout, RTC, remote inference
12 Reinforcement Learning  — HIL-SERL actor/learner, replay buffer, reward models
```

**"Show me how this code actually breaks."** →
[`issue-deep-dives/README.md`](./issue-deep-dives/README.md) — two hard concurrency bugs in
the camera layer and four medium contract bugs across motors, robots, the processor
pipeline and the config system, each with a reproduction and the regression test that pins
it.

`intro/` and `issue-deep-dives/` are plain GitHub-flavoured Markdown with Mermaid diagrams.
They are **not** part of the `doc-builder` build, so nothing in them can break the published
site, and they do not need an entry in `source/_toctree.yml`.

# Generating the documentation

To generate the documentation, you first have to build it. Several packages are necessary to build the doc,
you can install them with the following command, at the root of the code repository:

```bash
pip install -e . -r docs-requirements.txt
```

You will also need `nodejs`. Please refer to their [installation page](https://nodejs.org/en/download)

---

**NOTE**

You only need to generate the documentation to inspect it locally (if you're planning changes and want to
check how they look before committing for instance). You don't have to `git commit` the built documentation.

---

## Building the documentation

Once you have setup the `doc-builder` and additional packages, you can generate the documentation by
typing the following command:

```bash
doc-builder build lerobot docs/source/ --build_dir ~/tmp/test-build
```

You can adapt the `--build_dir` to set any temporary folder that you prefer. This command will create it and generate
the MDX files that will be rendered as the documentation on the main website. You can inspect them in your favorite
Markdown editor.

## Previewing the documentation

To preview the docs, first install the `watchdog` module with:

```bash
pip install watchdog
```

Then run the following command:

```bash
doc-builder preview lerobot docs/source/
```

The docs will be viewable at [http://localhost:3000](http://localhost:3000). You can also preview the docs once you have opened a PR. You will see a bot add a comment to a link where the documentation with your changes lives.

---

**NOTE**

The `preview` command only works with existing doc files. When you add a completely new file, you need to update `_toctree.yml` & restart `preview` command (`ctrl-c` to stop it & call `doc-builder preview ...` again).

---

## Adding a new element to the navigation bar

Accepted files are Markdown (.md).

Create a file with its extension and put it in the source directory. You can then link it to the toc-tree by putting
the filename without the extension in the [`_toctree.yml`](https://github.com/huggingface/lerobot/blob/main/docs/source/_toctree.yml) file.

## Renaming section headers and moving sections

It helps to keep the old links working when renaming the section header and/or moving sections from one document to another. This is because the old links are likely to be used in Issues, Forums, and Social media and it'd make for a much more superior user experience if users reading those months later could still easily navigate to the originally intended information.

Therefore, we simply keep a little map of moved sections at the end of the document where the original section was. The key is to preserve the original anchor.

So if you renamed a section from: "Section A" to "Section B", then you can add at the end of the file:

```
Sections that were moved:

[ <a href="#section-b">Section A</a><a id="section-a"></a> ]
```

and of course, if you moved it to another file, then:

```
Sections that were moved:

[ <a href="../new-file#section-b">Section A</a><a id="section-a"></a> ]
```

Use the relative style to link to the new file so that the versioned docs continue to work.

For an example of a rich moved sections set please see the very end of [the transformers Trainer doc](https://github.com/huggingface/transformers/blob/main/docs/source/en/main_classes/trainer.md).

### Adding a new tutorial

Adding a new tutorial or section is done in two steps:

- Add a new file under `./source`. This file can either be ReStructuredText (.rst) or Markdown (.md).
- Link that file in `./source/_toctree.yml` on the correct toc-tree.

Make sure to put your new file under the proper section. If you have a doubt, feel free to ask in a Github Issue or PR.

### Writing source documentation

Values that should be put in `code` should either be surrounded by backticks: \`like so\`. Note that argument names
and objects like True, None or any strings should usually be put in `code`.

#### Writing a multi-line code block

Multi-line code blocks can be useful for displaying examples. They are done between two lines of three backticks as usual in Markdown:

````
```
# first line of code
# second line
# etc
```
````

#### Adding an image

Due to the rapidly growing repository, it is important to make sure that no files that would significantly weigh down the repository are added. This includes images, videos, and other non-text files. We prefer to leverage a hf.co hosted `dataset` like
the ones hosted on [`hf-internal-testing`](https://huggingface.co/hf-internal-testing) in which to place these files and reference
them by URL. We recommend putting them in the following dataset: [huggingface/documentation-images](https://huggingface.co/datasets/huggingface/documentation-images).
If an external contribution, feel free to add the images to your PR and ask a Hugging Face member to migrate your images
to this dataset.
