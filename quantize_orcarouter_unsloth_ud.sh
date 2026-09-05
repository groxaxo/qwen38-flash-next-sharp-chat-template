#!/usr/bin/env bash
# Apply Unsloth Dynamic tensor map to orcarouter/Qwen3.8-Flash-Next-Uncensored.
#
# Does NOT copy Unsloth weights. Reads types from the Unsloth GGUF header
# (HF range requests, ~few MB) and feeds llama-quantize --tensor-type-file.
#
#   ./quantize_orcarouter_unsloth_ud.sh
#
# Env:
#   QUANT        Unsloth folder           default: UD-Q4_K_XL
#                also useful: UD-IQ4_XS (93.7 GB class)
#   UNSLOTH_REPO                          default: unsloth/Qwen3.8-Flash-Next-GGUF
#   SRC          OrcaRouter HF id/dir     default: orcarouter/Qwen3.8-Flash-Next-Uncensored
#   OUT          output prefix            default: ./out/Qwen38FN-Unc-$QUANT
#   LLAMA_CPP                             default: ./llama.cpp
#   MAP          existing type file       default: extract from Unsloth
#   IMATRIX      imatrix.gguf             default: AtomicChat's (or set RECOMPUTE_IMATRIX=1)
#   CALIB        text file                required if RECOMPUTE_IMATRIX=1
#   TEMPLATE     chat_template.jinja      default: ./chat_template.jinja
#   SKIP_CONVERT=1  SRC is already a GGUF master
#   OUTTYPE      convert outtype          default: bf16  (q8_0 from FP8)
#   MAP_ONLY=1   write the type file and exit
#   DRY_RUN=1    print llama-quantize argv and exit
set -euo pipefail

QUANT="${QUANT:-UD-Q4_K_XL}"
UNSLOTH_REPO="${UNSLOTH_REPO:-unsloth/Qwen3.8-Flash-Next-GGUF}"
SRC="${SRC:-orcarouter/Qwen3.8-Flash-Next-Uncensored}"
OUT="${OUT:-./out/Qwen38FN-Unc-${QUANT}}"
LLAMA_CPP="${LLAMA_CPP:-./llama.cpp}"
TEMPLATE="${TEMPLATE:-./chat_template.jinja}"
OUTTYPE="${OUTTYPE:-bf16}"
SPLIT_MAX="${SPLIT_MAX:-2G}"
IMATRIX_HF="${IMATRIX_HF:-AtomicChat/Qwen3.8-Flash-Next-GGUF}"
HERE="$(cd "$(dirname "$0")" && pwd)"

need() { command -v "$1" >/dev/null || { echo "missing $1"; exit 1; }; }
bin() {
  local n="$1"
  if [[ -x "$LLAMA_CPP/build/bin/$n" ]]; then echo "$LLAMA_CPP/build/bin/$n"
  elif command -v "$n" >/dev/null; then command -v "$n"
  else echo "missing $n (build llama.cpp --target $n)" >&2; exit 1
  fi
}

need python3
mkdir -p "$(dirname "$OUT")" "$(dirname "${MAP:-./maps/${QUANT}.tensor-types}")"

MAP="${MAP:-./maps/${QUANT}.tensor-types}"
if [[ ! -f "$MAP" ]]; then
  echo "== extract $UNSLOTH_REPO:$QUANT -> $MAP"
  python3 "$HERE/extract_gguf_tensor_map.py" "$UNSLOTH_REPO:$QUANT" -o "$MAP"
fi
[[ -s "$MAP" ]] || { echo "empty map $MAP"; exit 1; }
echo "== map $(grep -c '=' "$MAP") overrides"

if [[ "${MAP_ONLY:-0}" == "1" ]]; then
  echo "MAP_ONLY done: $MAP"
  exit 0
fi

Q="$(bin llama-quantize)"
S="$(bin llama-gguf-split)"
META=""
if [[ -x "$LLAMA_CPP/build/bin/gguf-new-metadata" ]]; then
  META="$LLAMA_CPP/build/bin/gguf-new-metadata"
elif command -v gguf-new-metadata >/dev/null; then
  META="$(command -v gguf-new-metadata)"
fi

MASTER="$OUT.master.gguf"
QUANT_OUT="$OUT.unsplit.gguf"

if [[ "${SKIP_CONVERT:-0}" == "1" ]]; then
  MASTER="$SRC"
else
  CONV="$LLAMA_CPP/convert_hf_to_gguf.py"
  [[ -f "$CONV" ]] || { echo "missing $CONV"; exit 1; }
  if [[ -d "$SRC" ]]; then
    LOCAL="$SRC"
  else
    need hf
    LOCAL="./hf-src-orca"
    hf download "$SRC" --local-dir "$LOCAL"
  fi
  echo "== convert $LOCAL -> $MASTER ($OUTTYPE)"
  python3 "$CONV" "$LOCAL" --outfile "$MASTER" --outtype "$OUTTYPE"
fi

IMATRIX="${IMATRIX:-}"
if [[ "${RECOMPUTE_IMATRIX:-0}" == "1" ]]; then
  [[ -n "${CALIB:-}" && -f "$CALIB" ]] || { echo "RECOMPUTE_IMATRIX=1 needs CALIB=file"; exit 1; }
  IMATRIX="$OUT.imatrix.gguf"
  echo "== llama-imatrix on OrcaRouter master"
  "$(bin llama-imatrix)" -m "$MASTER" -f "$CALIB" --chunk 512 --output-tensor "$IMATRIX" -ngl 99
elif [[ -z "$IMATRIX" ]]; then
  IMATRIX="./imatrix.gguf"
  if [[ ! -f "$IMATRIX" ]]; then
    need hf
    echo "== fetch AtomicChat imatrix (Unsloth map + stock imatrix; residual writers differ)"
    hf download "$IMATRIX_HF" imatrix.gguf --local-dir .
  fi
fi

# Do NOT pass --token-embedding-type: it steals per_layer_token_embd.
cmd=("$Q" --imatrix "$IMATRIX" --tensor-type-file "$MAP" "$MASTER" "$QUANT_OUT" q8_0)
echo "== ${cmd[*]}"
if [[ "${DRY_RUN:-0}" == "1" ]]; then
  exit 0
fi
"${cmd[@]}"

if [[ -f "$TEMPLATE" && -n "$META" && -x "$META" ]]; then
  echo "== bake $TEMPLATE"
  "$META" --chat-template-file "$TEMPLATE" "$QUANT_OUT" "$QUANT_OUT.tmpl"
  mv "$QUANT_OUT.tmpl" "$QUANT_OUT"
fi

echo "== split $SPLIT_MAX"
"$S" --split --split-max-size "$SPLIT_MAX" "$QUANT_OUT" "$OUT"
echo "done. load ${OUT}-00001-of-*.gguf"
echo "llama-server -m ${OUT}-00001-of-*.gguf --jinja -fit off -ngl 99 --reasoning-format deepseek"
