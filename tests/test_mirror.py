import cv2
import numpy as np

from swimsync.dataset.mirror import mirror_dataset, mirror_line


def test_mirror_line_flips_x_and_swaps_her_left_and_right():
    # 4 classes: left_leg, right_leg, left_arm, right_arm
    assert mirror_line("0 0.2 0.5 0.1 0.3") == "1 0.800000 0.500000 0.100000 0.300000"
    assert mirror_line("1 0.7 0.4 0.2 0.2") == "0 0.300000 0.400000 0.200000 0.200000"
    assert mirror_line("2 0.5 0.5 0.1 0.1").startswith("3 ")
    assert mirror_line("3 0.5 0.5 0.1 0.1").startswith("2 ")


def test_mirror_dataset_adds_flipped_training_copies_only(tmp_path):
    src, out = tmp_path / "src", tmp_path / "out"
    img = np.zeros((10, 20, 3), np.uint8)
    img[:, :5] = 255  # white stripe on the left
    for split in ("train", "val"):
        (src / "images" / split).mkdir(parents=True)
        (src / "labels" / split).mkdir(parents=True)
        cv2.imwrite(str(src / "images" / split / "a.jpg"), img)
        (src / "labels" / split / "a.txt").write_text("0 0.2 0.5 0.1 0.3")
    (src / "data.yaml").write_text(f"path: {src}\ntrain: images/train\nval: images/val\nnames:\n  0: left_leg\n")

    mirror_dataset(src, out)

    assert sorted(p.name for p in (out / "images/train").iterdir()) == ["a.jpg", "a_mirror.jpg"]
    assert sorted(p.name for p in (out / "images/val").iterdir()) == ["a.jpg"]
    flipped = cv2.imread(str(out / "images/train/a_mirror.jpg"))
    assert flipped[:, -5:].mean() > 200 and flipped[:, :5].mean() < 50  # stripe now on the right
    assert (out / "labels/train/a_mirror.txt").read_text() == "1 0.800000 0.500000 0.100000 0.300000"
    assert f"path: {out.resolve()}" in (out / "data.yaml").read_text()
