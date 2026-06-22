# WeChat Cover Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Generate and deliver one text-free WeChat article cover that communicates semantic-preserving medical image augmentation.

**Architecture:** Use the built-in image generation workflow to create a horizontal scientific editorial illustration from the approved design. Inspect the result for medical context, soft-mask clarity, semantic consistency, cropping safety, and accidental text, then copy the accepted PNG into the workspace.

**Tech Stack:** Built-in image generation tool, local image inspection, PowerShell file copy.

---

### Task 1: Generate the cover

**Files:**
- Create: `output/imagegen/wechat-cover-semantic-medical-augmentation.png`

- [ ] **Step 1: Generate one horizontal cover candidate**

Use the approved “semantic guardian” prompt: central medical image, precise translucent soft mask around the diagnostic region, several visually varied but semantically consistent augmented samples, cool cyan/navy palette with restrained coral highlights, no text, no logos, no watermark.

- [ ] **Step 2: Inspect the generated candidate**

Verify that the image is recognizably medical, the soft mask is visually distinct from a generic glow, key content stays in the central crop-safe region, and no accidental text or UI labels appear.

- [ ] **Step 3: Iterate only if a concrete defect is found**

If validation fails, regenerate with one targeted correction while preserving all approved constraints.

### Task 2: Save and verify the deliverable

**Files:**
- Create: `output/imagegen/wechat-cover-semantic-medical-augmentation.png`

- [ ] **Step 1: Copy the accepted image into the workspace**

Create `output/imagegen/` if needed and copy the accepted built-in output to the stable filename above without overwriting any unrelated asset.

- [ ] **Step 2: Re-open the workspace copy**

Confirm that the final file opens, retains the expected aspect ratio and composition, and contains no accidental text, logo, watermark, or visible artifact.

- [ ] **Step 3: Report the path and generation prompt**

Provide the absolute workspace path, an inline preview, the final prompt, and confirm that the built-in image generation mode was used.

### Task 3: Generate the two approved alternate directions

**Files:**
- Create: `output/imagegen/wechat-cover-three-step-evolution.png`
- Create: `output/imagegen/wechat-cover-distribution-search.png`

- [x] **Step 1: Generate the “three-step evolution” cover**

Create a left-to-right scientific visual sequence of original image, soft semantic mask extraction, augmented samples, and stabilized distribution, with no labels or text.

- [x] **Step 2: Generate the “distribution search” cover**

Create a medical-image-centered composition with feature particle clouds, constrained search trajectories, and a visible convergence region, while keeping the medical context dominant.

- [x] **Step 3: Crop, save, and inspect both outputs**

Save both as 900 × 383 PNG files, verify central safe-zone composition and confirm there is no text, logo, watermark, or generic robot/DNA/circuit imagery.
