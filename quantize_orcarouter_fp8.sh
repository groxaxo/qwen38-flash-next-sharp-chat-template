#!/usr/bin/env bash
# Full pipeline: orcarouter/Qwen3.8-Flash-Next-Uncensored-FP8 -> GGUF
#
#   RECIPE=unsloth QUANT=UD-Q4_K_XL ./quantize_orcarouter_fp8.sh
#   RECIPE=ad427                       ./quantize_orcarouter_fp8.sh
#   RECIPE=unsloth QUANT=UD-IQ4_XS     ./quantize_orcarouter_fp8.sh
#
# Steps:
#   1. hf download gated FP8 (~186 GB) unless SRC is a local dir
#   2. convert_hf_to_gguf.py --outtype q8_0 --fp8-as-q8  -> Q8 master (~176 GB)
#   3. optional llama-imatrix on that master
#   4. llama-quantize with Unsloth per-tensor map and/or AtomicChat AD-4.27
#   5. bake chat_template.jinja
#   6. llama-gguf-split --split-max-size 2G  (PLE lands in its own shard)
#
# Needs: llama.cpp with qwen4exp (PR #27742+), python3, hf, torch for convert.
# FP8 is gated — `hf auth login` and accept the repo terms first.
#
# Env:
#   SRC              default: orcarouter/Qwen3.8-Flash-Next-Uncensored-FP8
#   RECIPE           unsloth | ad427 | unsloth+ad427   default: unsloth
#   QUANT            Unsloth map source                default: UD-Q4_K_XL
#   UNSLOTH_REPO     default: unsloth/Qwen3.8-Flash-Next-GGUF
#   OUT              default: ./out/Qwen38FN-Unc-FP8-${RECIPE}-${QUANT}
#   LLAMA_CPP        default: ./llama.cpp
#   OUTTYPE          convert outtype                   default: q8_0
#   FP8_AS_Q8        1 (default) adds --fp8-as-q8
#   SKIP_CONVERT=1   SRC is already a GGUF master
#   MAP              pre-extracted .tensor-types
#   MAP_ONLY=1       write the Unsloth map and exit
#   IMATRIX          existing imatrix.gguf
#   RECOMPUTE_IMATRIX=1  + CALIB=file.txt
#   IMATRIX_HF       default: AtomicChat/Qwen3.8-Flash-Next-GGUF
#   TEMPLATE         default: ./chat_template.jinja next to this script
#   SPLIT_MAX        default: 2G
#   CONVERT_MTP=1    also emit MTP draft if converter supports --mtp
#   DRY_RUN=1
set -euo pipefail

SRC="${SRC:-orcarouter/Qwen3.8-Flash-Next-Uncensored-FP8}"
RECIPE="${RECIPE:-unsloth}"
QUANT="${QUANT:-UD-Q4_K_XL}"
UNSLOTH_REPO="${UNSLOTH_REPO:-unsloth/Qwen3.8-Flash-Next-GGUF}"
LLAMA_CPP="${LLAMA_CPP:-./llama.cpp}"
OUTTYPE="${OUTTYPE:-q8_0}"
FP8_AS_Q8="${FP8_AS_Q8:-1}"
SPLIT_MAX="${SPLIT_MAX:-2G}"
IMATRIX_HF="${IMATRIX_HF:-AtomicChat/Qwen3.8-Flash-Next-GGUF}"
HERE="$(cd "$(dirname "$0")" && pwd)"
TEMPLATE="${TEMPLATE:-$HERE/chat_template.jinja}"
SAFE_RECIPE="${RECIPE//+/-}"
OUT="${OUT:-./out/Qwen38FN-Unc-FP8-${SAFE_RECIPE}-${QUANT}}"

need() { command -v "$1" >/dev/null || { echo "missing $1" >&2; exit 1; }; }
bin() {
  local n="$1"
  if [[ -x "$LLAMA_CPP/build/bin/$n" ]]; then echo "$LLAMA_CPP/build/bin/$n"
  elif command -v "$n" >/dev/null; then command -v "$n"
  else echo "missing $n — build llama.cpp (qwen4exp) --target $n" >&2; exit 1
  fi
}
die() { echo "error: $*" >&2; exit 1; }

echo "== recipe=$RECIPE quant=$QUANT src=$SRC out=$OUT"

need python3
mkdir -p "$(dirname "$OUT")" ./maps ./hf-src-fp8

MAP="${MAP:-./maps/${QUANT}.tensor-types}"
if [[ "$RECIPE" == *unsloth* ]]; then
  EXTRACT="$HERE/extract_gguf_tensor_map.py"
  [[ -f "$EXTRACT" ]] || die "need $EXTRACT next to this script"
  if [[ ! -s "$MAP" ]]; then
    echo "== extract $UNSLOTH_REPO:$QUANT -> $MAP"
    python3 "$EXTRACT" "$UNSLOTH_REPO:$QUANT" -o "$MAP"
  fi
  [[ -s "$MAP" ]] || die "empty map $MAP"
  echo "== map $(grep -c '=' "$MAP") overrides"
  if ! grep -q 'per_layer_token_embd' "$MAP"; then
    echo "warn: map has no per_layer_token_embd — PLE will follow base q8_0" >&2
  fi
fi
if [[ "${MAP_ONLY:-0}" == "1" ]]; then
  echo "MAP_ONLY $MAP"; exit 0
fi

Q="$(bin llama-quantize)"
S="$(bin llama-gguf-split)"
META=""
if [[ -x "$LLAMA_CPP/build/bin/gguf-new-metadata" ]]; then
  META="$LLAMA_CPP/build/bin/gguf-new-metadata"
elif command -v gguf-new-metadata >/dev/null; then
  META="$(command -v gguf-new-metadata)"
fi

MASTER="$OUT.master.q8_0.gguf"
QUANT_OUT="$OUT.unsplit.gguf"

if [[ "${SKIP_CONVERT:-0}" == "1" ]]; then
  MASTER="$SRC"
  [[ -f "$MASTER" ]] || die "SKIP_CONVERT=1 but SRC is not a file: $SRC"
else
  CONV="$LLAMA_CPP/convert_hf_to_gguf.py"
  [[ -f "$CONV" ]] || die "missing $CONV"
  if [[ -d "$SRC" ]]; then
    LOCAL="$SRC"
  else
    need hf
    LOCAL="./hf-src-fp8"
    if [[ ! -f "$LOCAL/config.json" ]]; then
      echo "== hf download $SRC  (~186 GB, gated — accept terms + hf auth login)"
      hf download "$SRC" --local-dir "$LOCAL"
    else
      echo "== reuse $LOCAL"
    fi
  fi
  [[ -f "$LOCAL/config.json" ]] || die "no config.json under $LOCAL"

  CONV_ARGS=("$CONV" "$LOCAL" --outfile "$MASTER" --outtype "$OUTTYPE")
  if [[ "$FP8_AS_Q8" == "1" ]]; then
    if python3 "$CONV" --help 2>/dev/null | grep -q fp8-as-q8; then
      CONV_ARGS+=(--fp8-as-q8)
    else
      echo "warn: converter has no --fp8-as-q8; --outtype $OUTTYPE only"
    fi
  fi
  if [[ "${CONVERT_MTP:-0}" == "1" ]] && python3 "$CONV" --help 2>/dev/null | grep -q -- '--mtp'; then
    CONV_ARGS+=(--mtp)
  fi

  echo "== convert FP8 -> $MASTER"
  echo "   ${CONV_ARGS[*]}"
  if [[ "${DRY_RUN:-0}" != "1" ]]; then
    python3 "${CONV_ARGS[@]}" || die "convert failed. Need llama.cpp after PR #27742; optional marknx/flash-next-gguf-tools converter-fp8-fixes.patch"
  fi
fi

IMATRIX="${IMATRIX:-}"
if [[ "${RECOMPUTE_IMATRIX:-0}" == "1" ]]; then
  [[ -n "${CALIB:-}" && -f "${CALIB:-}" ]] || die "RECOMPUTE_IMATRIX=1 needs CALIB=file"
  IMATRIX="$OUT.imatrix.gguf"
  echo "== llama-imatrix $MASTER"
  if [[ "${DRY_RUN:-0}" != "1" ]]; then
    "$(bin llama-imatrix)" -m "$MASTER" -f "$CALIB" --chunk 512 --output-tensor "$IMATRIX" -ngl 99
  fi
elif [[ -z "$IMATRIX" ]]; then
  IMATRIX="./imatrix.gguf"
  if [[ ! -f "$IMATRIX" ]]; then
    need hf
    echo "== fetch $IMATRIX_HF imatrix.gguf"
    hf download "$IMATRIX_HF" imatrix.gguf --local-dir .
  fi
fi
[[ "${DRY_RUN:-0}" == "1" || -f "$IMATRIX" ]] || die "no imatrix $IMATRIX"

cmd=("$Q" --imatrix "$IMATRIX")
case "$RECIPE" in
  unsloth)
    cmd+=(--tensor-type-file "$MAP")
    ;;
  ad427)
    cmd+=(
      --tensor-type 'blk\.([0-3]|4[0-7])\.ffn_(gate|up)_exps=iq3_s'
      --tensor-type 'ffn_down_exps=iq4_nl'
      --tensor-type 'ffn_gate_exps=iq2_s'
      --tensor-type 'ffn_up_exps=iq2_s'
      --tensor-type 'per_layer_token_embd=q5_1'
    )
    ;;
  unsloth+ad427|ad427+unsloth)
    cmd+=(--tensor-type-file "$MAP")
    cmd+=(--tensor-type 'ffn_down_exps=iq4_nl')
    cmd+=(--tensor-type 'per_layer_token_embd=q5_1')
    ;;
  *)
    die "RECIPE must be unsloth | ad427 | unsloth+ad427"
    ;;
esac
cmd+=("$MASTER" "$QUANT_OUT" q8_0)
echo "== quantize"
echo "   ${cmd[*]}"
if [[ "${DRY_RUN:-0}" == "1" ]]; then
  exit 0
fi
"${cmd[@]}"

if [[ -f "$TEMPLATE" && -n "$META" && -x "$META" ]]; then
  echo "== bake $TEMPLATE"
  "$META" --chat-template-file "$TEMPLATE" "$QUANT_OUT" "$QUANT_OUT.tmpl"
  mv "$QUANT_OUT.tmpl" "$QUANT_OUT"
else
  echo "warn: skip template bake"
fi

echo "== split $SPLIT_MAX"
"$S" --split --split-max-size "$SPLIT_MAX" "$QUANT_OUT" "$OUT"

echo
echo "done."
echo "  master   $MASTER"
echo "  shards   ${OUT}-00001-of-*.gguf"
echo
echo "llama-server -m ${OUT}-00001-of-*.gguf \\"
echo "  --mmproj mmproj-Qwen3.8-Flash-Next-Uncensored-F16.gguf \\"
echo "  -ngl 99 -c 32768 --jinja -fit off --reasoning-format deepseek"
echo
echo "traps: mmap on; no --override-tensor for PLE; -fit off; repo gated; no MLX-4bit."
