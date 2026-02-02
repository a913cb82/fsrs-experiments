def decode_varint(data: bytes, pos: int) -> tuple[int, int]:
    """Decodes a Protobuf varint from bytes."""
    res = 0
    shift = 0
    while True:
        b = data[pos]
        res |= (b & 0x7F) << shift
        pos += 1
        if not (b & 0x80):
            break
        shift += 7
    return res, pos


def get_field_from_proto(blob: bytes, field_no: int) -> int | None:
    """Extracts a varint field from a Protobuf blob."""
    if not blob:
        return None
    pos = 0
    while pos < len(blob):
        try:
            tag, pos = decode_varint(blob, pos)
            f_num = tag >> 3
            w_type = tag & 0x07
            if f_num == field_no and w_type == 0:
                val, pos = decode_varint(blob, pos)
                return val
            # Skip field
            if w_type == 0:
                _, pos = decode_varint(blob, pos)
            elif w_type == 1:
                pos += 8
            elif w_type == 2:
                length, pos = decode_varint(blob, pos)
                pos += length
            elif w_type == 5:
                pos += 4
            else:
                break
        except Exception:
            break
    return None


def get_deck_config_id(common_blob: bytes, kind_blob: bytes) -> int:
    """
    Extracts config_id from Anki's DeckCommon or NormalDeck blobs.
    Defaults to 1 if not found.
    """
    # 1. Try NormalDeck.config_id (field 1 of sub-message in field 1 of kind blob)
    if kind_blob:
        pos = 0
        while pos < len(kind_blob):
            try:
                tag, pos = decode_varint(kind_blob, pos)
                f_num = tag >> 3
                w_type = tag & 0x07
                if f_num == 1 and w_type == 2:  # normal field
                    length, pos = decode_varint(kind_blob, pos)
                    normal_deck_blob = kind_blob[pos : pos + length]
                    cid = get_field_from_proto(normal_deck_blob, 1)
                    if cid is not None:
                        return cid
                    break
                # Skip other fields in kind
                if w_type == 0:
                    _, pos = decode_varint(kind_blob, pos)
                elif w_type == 1:
                    pos += 8
                elif w_type == 2:
                    length, pos = decode_varint(kind_blob, pos)
                    pos += length
                elif w_type == 5:
                    pos += 4
                else:
                    break
            except Exception:
                break

    # 2. Try DeckCommon.config_id (field 1 of common blob)
    cid = get_field_from_proto(common_blob, 1)
    if cid is not None:
        return cid

    return 1
