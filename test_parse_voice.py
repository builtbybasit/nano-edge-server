from server import parse_voice

assert parse_voice("af_heart") == [("af_heart", 1.0)]
assert parse_voice("af_heart,af_bella") == [("af_heart", 0.5), ("af_bella", 0.5)]
assert parse_voice(" af_heart:3 , am_michael:1 ") == [("af_heart", 0.75), ("am_michael", 0.25)]
for bad in ("", "af_heart:0", "af_heart:-1", "af_heart:x", "af_heart,,af_bella"):
    try:
        parse_voice(bad)
    except ValueError:
        continue
    raise AssertionError(f"accepted {bad!r}")
print("parse_voice ok")
