from pathlib import Path

root = Path(
    "/data1/nakaue/MAGI-MD/dataset/MVMD/original/mvmd_test_2"
)

skip_names = {
    "confidence_by_scene_frame_unique",
}

for scene_dir in sorted(root.iterdir()):
    if not scene_dir.is_dir():
        continue

    if scene_dir.name in skip_names:
        continue

    pair_dir = scene_dir / "pair"

    if not pair_dir.exists():
        print(f"[SKIP] {scene_dir.name}: pair directory not found")
        continue

    pairs = sorted(
        p for p in pair_dir.iterdir()
        if p.is_dir() and p.name.startswith("pair_")
    )

    if len(pairs) == 0:
        print(f"[SKIP] {scene_dir.name}: no pairs found")
        continue

    selected_pair = pairs[0]

    images_dir = selected_pair / "JPEGImages_pair"

    images = sorted(images_dir.glob("*.jpg"))

    if len(images) != 2:
        print(
            f"[SKIP] {scene_dir.name}: "
            f"{len(images)} images found in {selected_pair.name}"
        )
        continue

    print(
        scene_dir.name,
        selected_pair.name,
        images[0].name,
        images[1].name
    )