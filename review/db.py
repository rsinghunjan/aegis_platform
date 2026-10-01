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
 99
100
101
102
103
104
105
106
107
108
109
110
111
112
113
114
115
116
117
118
119
120
121
122
123
124
125
"""
        INSERT OR REPLACE INTO review_queue (id, request_id, created_at, flagged, reason, text_snippet, metadata, status)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    """, (nid, request_id, created_at, int(bool(flagged)), reason[:128], text_snippet[:512], metadata_json, "pending"))
    conn.commit()
    conn.close()
    return nid

def list_pending(limit: int = 100) -> List[Dict]:
    init_db()
    conn = _get_conn()
    c = conn.cursor()
    c.execute("SELECT id, request_id, created_at, flagged, reason, text_snippet, metadata FROM review_queue WHERE status = ? ORDER BY created_at ASC LIMIT ?", ("pending", limit))
    rows = c.fetchall()
    conn.close()
    items = []
    for r in rows:
        items.append({
            "id": r[0],
            "request_id": r[1],
            "created_at": r[2],
            "flagged": bool(r[3]),
            "reason": r[4],
            "text_snippet": r[5],
            "metadata": json.loads(r[6] or "{}")
        })
    return items

def get_item(item_id: str) -> Optional[Dict]:
    init_db()
    conn = _get_conn()
    c = conn.cursor()
    c.execute("SELECT id, request_id, created_at, flagged, reason, text_snippet, metadata, status, reviewer, review_at, review_notes FROM review_queue WHERE id = ?", (item_id,))
    row = c.fetchone()
    conn.close()
    if not row:
        return None
    return {
        "id": row[0],
        "request_id": row[1],
        "created_at": row[2],
        "flagged": bool(row[3]),
        "reason": row[4],
        "text_snippet": row[5],
        "metadata": json.loads(row[6] or "{}"),
        "status": row[7],
        "reviewer": row[8],
        "review_at": row[9],
        "review_notes": row[10],
    }

def set_review(item_id: str, reviewer: str, verdict: str, notes: Optional[str] = None) -> bool:
    """
    verdict: 'allow' | 'dismiss' | 'block' -> we store status accordingly.
    """
    init_db()
    conn = _get_conn()
    c = conn.cursor()
    review_at = time.time()
    status = "reviewed"
    c.execute("UPDATE review_queue SET status=?, reviewer=?, review_at=?, review_notes=? WHERE id = ?", (status, reviewer[:64], review_at, (notes or "")[:2000], item_id))
    conn.commit()
    changed = c.rowcount > 0
    conn.close()
    return changed

def export_pending(out_path: str = "logs/pending_export.json"):
    items = list_pending(limit=10000)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(items, f, ensure_ascii=False, indent=2)
    return out_path
