 42
 43
 44
 45
 46
 47
 48
 49
 50
 51
 52
 53
 54
 55
 56
 57
 58
 59
 60
 61
 62
 63
 64
 65
 66
 67
 68
 69
 70
 71
 72
 73
 74
 75
 76
 77
 78
 79
 80
 81
 82
 83
 84
 85
 86
 87
 88
 89
 90
 91
 92
 93
 94
 95
 96
 97
 98
#!/usr/bin/env python3

def gen_key_hex(length_bytes=32):
    return secrets.token_hex(length_bytes)

def write_kv_v2(client, path_suffix, data):
    # path_suffix: like "model_sign/<key_id>"
    full_path = f"{VAULT_MOUNT}/data/aegis/keys/{path_suffix}"
    client.secrets.kv.v2.create_or_update_secret(
        path="/".join(["aegis","keys",path_suffix]),
        secret={"value": data},
        mount_point=VAULT_MOUNT
    )

def read_active(client):
    try:
        res = client.secrets.kv.v2.read_secret_version(path="aegis/keys/model_sign/active", mount_point=VAULT_MOUNT)
        return res["data"]["data"]["value"]
    except Exception:
        return None

def main(dry_run=True):
    client = hvac_client()
    key_id = str(uuid.uuid4())
    key_value = gen_key_hex(32)
    ts = datetime.utcnow().isoformat() + "Z"

    # store new key
    payload = {
        "key_id": key_id,
        "key": key_value,
        "created_at": ts,
        "created_by": os.environ.get("ROTATED_BY", "automation"),
        "status": "active"
    }

    print("New key id:", key_id)
    print("Preview payload:", json.dumps(payload) if dry_run else "(hidden)")

    if dry_run:
        print("DRY RUN: not writing to Vault")
        return payload

    # write key artifact
    write_kv_v2(client, f"model_sign/{key_id}", payload)

    # update active pointer (store active_key_id + created_at)
    active_payload = {"active_key_id": key_id, "activated_at": ts}
    write_kv_v2(client, "model_sign/active", active_payload)

    print("Wrote key and updated active pointer in Vault.")
    return payload

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--preview", action="store_true", help="preview only (no writes)")
    args = p.parse_args()
    main(dry_run=args.preview)
