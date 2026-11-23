# Version 2 variants, in the order of Section 41 of the notebook
BASELINE_VARIANTS=(b_control b_clip b_grid b_ic b_missing b_huber)
GPU_VARIANTS=(control huber clip clip_month robust select_clip select_ic rank missing mlp gated
              dropout gradclip schedule earlystop joint50 joint75 combined)
ALL_VARIANTS=("${GPU_VARIANTS[@]}" "${BASELINE_VARIANTS[@]}")

is_variant() {  # is_variant <name>
    local name
    for name in "${ALL_VARIANTS[@]}"; do [ "$name" = "$1" ] && return 0; done
    return 1
}
is_baseline() {
    local name
    for name in "${BASELINE_VARIANTS[@]}"; do [ "$name" = "$1" ] && return 0; done
    return 1
}
