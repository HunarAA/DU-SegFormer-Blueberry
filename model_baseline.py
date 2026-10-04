from transformers import SegformerForSemanticSegmentation


def build_model(model_name="nvidia/mit-b0", num_classes=2):
    """Plain, unmodified SegFormer-B0 baseline."""
    return SegformerForSemanticSegmentation.from_pretrained(
        model_name,
        num_labels=num_classes,
        id2label={0: "background", 1: "blueberry"},
        label2id={"background": 0, "blueberry": 1},
    )
