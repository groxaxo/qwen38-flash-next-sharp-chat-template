#!/usr/bin/env bash
# AtomicChat AD-4.27bpw ("Q4.5") recipe for orcarouter/Qwen3.8-Flash-Next-Uncensored.
# Needs llama.cpp with qwen4exp (PR #27742+): convert_hf_to_gguf.py, llama-quantize,
# llama-gguf-split, llama-imatrix, gguf-new-metadata.
#
#   ./quantize_orcarouter_ad427.sh
#
# Env:
#   SRC          HF id or local dir of BF16/FP8 weights
#                default: orcarouter/Qwen3.8-Flash-Next-Uncensored
#   OUT          output prefix              default: ./out/Qwen38FN-Unc-AD-4.27bpw
#   LLAMA_CPP    llama.cpp root             default: ./llama.cpp
#   IMATRIX      existing imatrix.gguf      default: download AtomicChat's
#   RECOMPUTE_IMATRIX=1   rebuild imatrix on THIS checkpoint (better; slow)
#   CALIB        text file for imatrix      required if RECOMPUTE_IMATRIX=1
#   TEMPLATE     chat_template.jinja        default: ./chat_template.jinja
#   SKIP_CONVERT=1   SRC is already a GGUF master
#   OUTTYPE      convert outtype            default: bf16  (use q8_0 from FP8)
set -euo pipefail

SRC="${SRC:-orcarouter/Qwen3.8-Flash-Next-Uncensored}"
OUT="${OUT:-./out/Qwen38FN-Unc-AD-4.27bpw}"
LLAMA_CPP="${LLAMA_CPP:-./llama.cpp}"
TEMPLATE="${TEMPLATE:-./chat_template.jinja}"
OUTTYPE="${OUTTYPE:-bf16}"
SPLIT_MAX="${SPLIT_MAX:-2G}"
IMATRIX_HF="${IMATRIX_HF:-AtomicChat/Qwen3.8-Flash-Next-GGUF}"

need() { command -v "$1" >/dev/null || { echo "missing $1"; exit 1; }; }
bin() {
  local n="$1"
  if [[ -x "$LLAMA_CPP/build/bin/$n" ]]; then echo "$LLAMA_CPP/build/bin/$n"
  elif command -v "$n" >/dev/null; then command -v "$n"
  else echo "missing $n (build llama.cpp --target $n)" >&2; exit 1
  fi
}

need python3
mkdir -p "$(dirname "$OUT")"
Q="$(bin llama-quantize)"
S="$(bin llama-gguf-split)"
META=""
if [[ -x "$LLAMA_CPP/build/bin/gguf-new-metadata" ]]; then
  META="$LLAMA_CPP/build/bin/gguf-new-metadata"
elif command -v gguf-new-metadata >/dev/null; then
  META="$(command -v gguf-new-metadata)"
fi

MASTER="$OUT.master.gguf"
QUANT="$OUT.unsplit.gguf"

if [[ "${SKIP_CONVERT:-0}" == "1" ]]; then
  MASTER="$SRC"
else
  CONV="$LLAMA_CPP/convert_hf_to_gguf.py"
  [[ -f "$CONV" ]] || { echo "missing $CONV"; exit 1; }
  if [[ -d "$SRC" ]]; then
    LOCAL="$SRC"
  else
    need hf
    LOCAL="./hf-src"
    hf download "$SRC" --local-dir "$LOCAL"
  fi
  echo "== convert $LOCAL -> $MASTER ($OUTTYPE)"
  python3 "$CONV" "$LOCAL" --outfile "$MASTER" --outtype "$OUTTYPE"
fi

IMATRIX="${IMATRIX:-}"
if [[ "${RECOMPUTE_IMATRIX:-0}" == "1" ]]; then
  [[ -n "${CALIB:-}" && -f "$CALIB" ]] || { echo "RECOMPUTE_IMATRIX=1 needs CALIB=file"; exit 1; }
  IMATRIX="$OUT.imatrix.gguf"
  echo "== llama-imatrix on this checkpoint"
  "$(bin llama-imatrix)" -m "$MASTER" -f "$CALIB" --chunk 512 --output-tensor "$IMATRIX" -ngl 99
elif [[ -z "$IMATRIX" ]]; then
  IMATRIX="./imatrix.gguf"
  if [[ ! -f "$IMATRIX" ]]; then
    need hf
    echo "== fetch AtomicChat imatrix (experts/PLE untouched by abliteration; residual writers differ)"
    hf download "$IMATRIX_HF" imatrix.gguf --local-dir .
  fi
fi

echo "== quantize AD-4.27bpw"
# ffn_down_exps ncols=640 → only block-32 types (iq4_nl). PLE ncols=160 → q5_1.
# Band IQ3_S on blk 0-3 and 40-47 from AtomicChat imatrix energy ranking.
"$Q" --imatrix "$IMATRIX" \
  --tensor-type 'blk\.([0-3]|4[0-7])\.ffn_(gate|up)_exps=iq3_s' \
  --tensor-type 'ffn_down_exps=iq4_nl' \
  --tensor-type 'ffn_gate_exps=iq2_s' \
  --tensor-type 'ffn_up_exps=iq2_s' \
  --tensor-type 'per_layer_token_embd=q5_1' \
  "$MASTER" "$QUANT" q8_0

if [[ -f "$TEMPLATE" && -x "${META:-}" ]]; then
  echo "== bake $TEMPLATE"
  "$META" --chat-template-file "$TEMPLATE" "$QUANT" "$QUANT.tmpl"
  mv "$QUANT.tmpl" "$QUANT"
elif [[ -f "$TEMPLATE" ]]; then
  echo "warn: gguf-new-metadata missing; bake later with --chat-template-file"
fi

echo "== split $SPLIT_MAX (isolates PLE into its own shard — required for Metal mmap)"
"$S" --split --split-max-size "$SPLIT_MAX" "$QUANT" "$OUT"

echo "done. load ${OUT}-00001-of-*.gguf"
echo "llama-server -m ${OUT}-00001-of-*.gguf --jinja -fit off -ngl 99 --reasoning-format deepseek"
echo "traps: keep mmap on; no --override-tensor for PLE; mxfp4 on ffn_down_exps drops the imatrix"
