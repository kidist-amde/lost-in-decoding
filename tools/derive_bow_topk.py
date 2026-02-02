#!/usr/bin/env python3
"""Derive top-k bow files by truncating an existing docid_to_tokenids.json.

Assumes input is a JSON object mapping docid strings to arrays of token ids,
with arrays sorted by score desc.
"""
import argparse
import json
import os
from typing import TextIO


class BufReader:
    def __init__(self, fp: TextIO, chunk_size: int = 1 << 20):
        self.fp = fp
        self.chunk_size = chunk_size
        self.buf = ""
        self.pos = 0
        self.eof = False

    def _fill(self, min_needed: int = 1):
        while not self.eof and len(self.buf) - self.pos < min_needed:
            chunk = self.fp.read(self.chunk_size)
            if not chunk:
                self.eof = True
                break
            if self.pos == 0:
                self.buf += chunk
            else:
                self.buf = self.buf[self.pos :] + chunk
                self.pos = 0

    def peek(self) -> str:
        self._fill(1)
        if self.pos >= len(self.buf):
            return ""
        return self.buf[self.pos]

    def getch(self) -> str:
        self._fill(1)
        if self.pos >= len(self.buf):
            return ""
        ch = self.buf[self.pos]
        self.pos += 1
        return ch

    def skip_ws(self):
        while True:
            ch = self.peek()
            if ch and ch in " \t\n\r":
                self.pos += 1
                continue
            break


def read_json_string(br: BufReader) -> str:
    ch = br.getch()
    if ch != '"':
        raise ValueError(f"Expected '\"' at start of string, got {ch!r}")
    out = []
    while True:
        ch = br.getch()
        if ch == "":
            raise ValueError("Unexpected EOF in string")
        if ch == '"':
            break
        if ch == "\\":
            esc = br.getch()
            if esc == "u":
                hexs = "".join(br.getch() for _ in range(4))
                out.append(chr(int(hexs, 16)))
            else:
                escapes = {
                    '"': '"',
                    "\\": "\\",
                    "/": "/",
                    "b": "\b",
                    "f": "\f",
                    "n": "\n",
                    "r": "\r",
                    "t": "\t",
                }
                out.append(escapes.get(esc, esc))
        else:
            out.append(ch)
    return "".join(out)


def read_json_array_text(br: BufReader) -> str:
    ch = br.getch()
    if ch != "[":
        raise ValueError(f"Expected '[' at start of array, got {ch!r}")
    depth = 1
    parts = ["["]
    while depth > 0:
        ch = br.getch()
        if ch == "":
            raise ValueError("Unexpected EOF in array")
        parts.append(ch)
        if ch == "[":
            depth += 1
        elif ch == "]":
            depth -= 1
    return "".join(parts)


def stream_truncate(input_path: str, out_path: str, topk: int):
    with open(input_path, "r", encoding="utf-8") as f_in, open(
        out_path, "w", encoding="utf-8"
    ) as f_out:
        br = BufReader(f_in)
        br.skip_ws()
        if br.getch() != "{":
            raise ValueError("Input is not a JSON object")
        f_out.write("{")
        first = True
        while True:
            br.skip_ws()
            ch = br.peek()
            if ch == "":
                raise ValueError("Unexpected EOF while reading object")
            if ch == "}":
                br.getch()
                break
            if ch == ",":
                br.getch()
                br.skip_ws()
            key = read_json_string(br)
            br.skip_ws()
            if br.getch() != ":":
                raise ValueError("Expected ':' after key")
            br.skip_ws()
            arr_text = read_json_array_text(br)
            arr = json.loads(arr_text)
            if not isinstance(arr, list):
                raise ValueError("Array value is not a list")
            arr = arr[:topk]
            if not first:
                f_out.write(",")
            first = False
            f_out.write(json.dumps(key))
            f_out.write(":")
            f_out.write(json.dumps(arr, separators=(",", ":")))
        f_out.write("}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--topk", type=int, required=True)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=False)
    out_json = os.path.join(args.out_dir, "docid_to_tokenids.json")
    out_meta = os.path.join(args.out_dir, "meta_data.json")

    stream_truncate(args.input, out_json, args.topk)
    with open(out_meta, "w", encoding="utf-8") as f:
        json.dump({"topk": args.topk}, f)


if __name__ == "__main__":
    main()
