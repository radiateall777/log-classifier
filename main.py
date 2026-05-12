def main():
    import torch

    payload = torch.load(
        "outputs/teacher/ce_unixcoder_seed42/logits/dev_teacher_logits.pt",
        map_location="cpu",
    )

    print(payload.keys())
    print(payload["logits"].shape)
    print(payload["labels"].shape)
    print(payload["probs"][0])
    print(payload["probs"][0].sum())


if __name__ == "__main__":
    main()
