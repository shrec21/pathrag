from PIL import Image

from pathrag import data


def img(color):
    return Image.new("RGB", (8, 8), color)


ROWS = [
    ("train", {"image": img("red"), "question": "What is present?", "answer": "hyaline arteriolosclerosis"}),
    ("train", {"image": img("red"), "question": "Is this the kidney?", "answer": "yes"}),
    ("train", {"image": img("red"), "question": "Is this the  liver?", "answer": "no"}),
    ("train", {"image": img("red"), "question": "Is this the kidney?", "answer": "yes"}),
    ("test", {"image": img("blue"), "question": "What does this show?", "answer": "Amyloidosis"}),
    ("test", {"image": img("red"), "question": "Where is this?", "answer": "kidney"}),
    ("validation", {"image": img("green"), "question": "Is there necrosis?", "answer": "no"}),
]


def test_same_pixels_same_id():
    assert data.image_id(img("red")) == data.image_id(img("red"))
    assert data.image_id(img("red")) != data.image_id(img("blue"))


def test_load_saves_each_image_once(tmp_path):
    recs = data.load_rows(ROWS, image_dir=tmp_path)
    assert len(recs) == 7
    assert len(list(tmp_path.glob("*.png"))) == 3
    assert recs[2].question == "Is this the liver?"


def test_split_report_finds_shared_image():
    recs = data.load_rows(ROWS)
    rep = data.split_report(recs)
    red = data.image_id(img("red"))
    assert rep["shared_images"] == {red: ["test", "train"]}
    assert rep["counts"]["train"] == {"qa": 4, "images": 1, "closed": 3}


def test_shared_image_goes_to_train_and_leaves_eval():
    recs = data.load_rows(ROWS)
    kept = data.drop_cross_split(recs)
    red = data.image_id(img("red"))
    assert all(r.split == "train" for r in kept if r.image_id == red)
    assert len(kept) == 6


def test_description_order_and_dedupe():
    recs = data.load_rows(ROWS)
    descs = data.build_descriptions(recs)
    red = descs[data.image_id(img("red"))]
    assert red.split == "train"
    assert red.facts == [
        "What is present: hyaline arteriolosclerosis.",
        "Where is this: kidney.",
        "Is this the kidney: yes.",
        "Is this the liver: no.",
    ]
    no_neg = data.build_descriptions(recs, include_no=False)
    assert "Is this the liver: no." not in no_neg[red.image_id].facts


def test_jsonl_round_trip(tmp_path):
    recs = data.load_rows(ROWS)
    descs = data.build_descriptions(recs)
    data.write_jsonl(tmp_path / "qa.jsonl", recs)
    data.write_jsonl(tmp_path / "d.jsonl", descs.values())
    assert data.read_qa(tmp_path / "qa.jsonl") == recs
    back = data.read_descriptions(tmp_path / "d.jsonl")
    assert {k: v.facts for k, v in back.items()} == {k: v.facts for k, v in descs.items()}
