import torch, os, glob
from transformers import AutoTokenizer, AutoConfig, AutoModelForSeq2SeqLM
from huggingface_hub import snapshot_download
for m in ["Salesforce/codet5p-220m", "Salesforce/codet5p-770m"]:
    src = snapshot_download(m); name = m.split("/")[1]
    out = "$WORKSPACE/baselines/hf_local/" + name
    if os.path.exists(out + "/model.safetensors"):
        print("exists", out); continue
    cfg = AutoConfig.from_pretrained(src)
    model = AutoModelForSeq2SeqLM.from_config(cfg)
    sd = torch.load(glob.glob(src + "/pytorch_model*.bin")[0], map_location="cpu", weights_only=True)
    missing, unexpected = model.load_state_dict(sd, strict=False)
    print(m, "missing", len(missing), list(missing)[:3], "unexpected", len(unexpected), list(unexpected)[:3])
    model.save_pretrained(out, safe_serialization=True); AutoTokenizer.from_pretrained(src).save_pretrained(out)
    print("EFFECT: converted", out, os.path.getsize(out + "/model.safetensors") // 2**20, "MB")
