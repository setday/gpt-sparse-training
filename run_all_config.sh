#!/usr/bin/env bash
set -euo pipefail

export WANDB_MODE="offline"

CONFIG="config/train_shakespeare_char.py"
LOGFILE="eval_results.log"

# очистить предыдущий лог
: > "${LOGFILE}"

# --- списки значений ---
RATIOS=(0.0)
TYPES=("none")
ACTIVATIONS=("gelu" "relu" "relu^2")
L1_TARGETS=("none" "input" "output" "weight")

run_once () {
  local stype="$1"
  local sratio="$2"
  local activation="$3"
  local l1_target="$4"
  local run_name="${stype}-${sratio}"
  local out_dir="out-${activation}-${l1_target}"

  echo "=========================="
  echo "Run: ${run_name}"
  echo "Out dir: ${out_dir}"
  echo "=========================="

  # TRAIN
  python train.py "${CONFIG}" \
    --out_dir="${out_dir}" \
    --sparsity_ratio="${sratio}" \
    --sparsity_type="${stype}" \
    --activation_function="${activation}" \
    --l1_target="${l1_target}" \
    --wandb_run_name="${run_name}"

  # EVAL (resume)
  tmpfile="$(mktemp)"
  python train.py "${CONFIG}" \
    --run_mode='eval' \
    --init_from='resume' \
    --out_dir="${out_dir}" \
    --sparsity_ratio="${sratio}" \
    --sparsity_type="${stype}" \
    --activation_function="${activation}" \
    --l1_target="${l1_target}" \
    --wandb_run_name="${run_name}" | tee "${tmpfile}"

  # --- парсинг результатов -------------------------------------------------
  # 1) step / train / val losses
  step_line="$(grep -E 'step[[:space:]]+[0-9]+:' "${tmpfile}" | tail -n1 || true)"
  if [[ -n "${step_line}" && "${step_line}" =~ step[[:space:]]+([0-9]+):[[:space:]]+train[[:space:]]+loss[[:space:]]+([0-9.]+),[[:space:]]+val[[:space:]]+loss[[:space:]]+([0-9.]+) ]]; then
    step_val="${BASH_REMATCH[1]}"
    train_loss="${BASH_REMATCH[2]}"
    val_loss="${BASH_REMATCH[3]}"
  else
    step_val="N/A"
    train_loss="N/A"
    val_loss="N/A"
  fi

  # 2) perplexity
  ppl_line="$(grep -E 'Strict perplexity over full val\.bin:' "${tmpfile}" | tail -n1 || true)"
  if [[ -n "${ppl_line}" && "${ppl_line}" =~ Strict\ perplexity\ over\ full\ val\.bin:\ ([0-9.]+) ]]; then
    ppl="${BASH_REMATCH[1]}"
  else
    ppl="N/A"
  fi

  rm -f "${tmpfile}"

  # --- записываем в общий лог ----------------------------------------------
  {
    echo "${run_name}:"
    printf '\tstep: %s\n'        "${step_val}"
    printf '\ttrain loss: %s\n'  "${train_loss}"
    printf '\tval loss: %s\n'    "${val_loss}"
    printf '\tppl: %s\n'         "${ppl}"
    echo
  } | tee -a "${LOGFILE}"
}

# базовый запуск


# сетка
for stype in "${TYPES[@]}"; do
  for sratio in "${RATIOS[@]}"; do
    for activation in "${ACTIVATIONS[@]}"; do
      for l1_target in "${L1_TARGETS[@]}"; do
        run_once "${stype}" "${sratio}" "${activation}" "${l1_target}"
      done
    done
  done
done

run_once "orig" "0.0"

echo "Все эксперименты завершены. Итоги см. в ${LOGFILE}"
