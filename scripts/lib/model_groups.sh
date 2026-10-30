#!/bin/bash
# =============================================================================
# model_groups.sh - the model export groups: which tiers run them, which Triton
# models they load, which files prove they ran, which models /health requires
# =============================================================================
# Sourced by model_setup.sh. Tables only; no side effects.

[[ -n "${_MODEL_GROUPS_SH_LOADED:-}" ]] && return 0
_MODEL_GROUPS_SH_LOADED=1

MODEL_SETUP_ALL_GROUPS=(preflight base yolo mobileclip faces ocr pe)

# model_setup_groups_for_tiers "core curation ..." -> ordered group names
model_setup_groups_for_tiers() {
    local tiers=" $1 " g
    for g in "${MODEL_SETUP_ALL_GROUPS[@]}"; do
        case "$g" in
            pe) [[ "$tiers" == *" curation "* ]] && echo "$g" ;;
            *) [[ "$tiers" == *" core "* ]] && echo "$g" ;;
        esac
    done
    return 0
}

# model_setup_group_models NAME -> Triton models loaded after the group
model_setup_group_models() {
    case "$1" in
        yolo) echo "yolov11_small_trt_end2end" ;;
        mobileclip) echo "mobileclip2_s2_image_encoder mobileclip2_s2_text_encoder" ;;
        faces) echo "scrfd_10g_bnkps arcface_w600k_r50" ;;
        ocr) echo "paddleocr_det_trt paddleocr_rec_trt ocr_pipeline" ;;
        pe) echo "pe_image_encoder pe_text_encoder" ;;
        *) echo "" ;;
    esac
}

# model_setup_group_outputs NAME -> host paths that prove the group ran
model_setup_group_outputs() {
    local m="${OP_DIR:?OP_DIR not set}/models"
    case "$1" in
        yolo) echo "$m/yolov11_small_trt_end2end/1/model.plan" ;;
        mobileclip) echo "$m/mobileclip2_s2_image_encoder/1/model.plan $m/mobileclip2_s2_text_encoder/1/model.plan" ;;
        faces) echo "$m/scrfd_10g_bnkps/1/model.plan $m/arcface_w600k_r50/1/model.plan" ;;
        ocr) echo "$m/paddleocr_det_trt/1/model.plan $m/paddleocr_rec_trt/1/model.plan" ;;
        pe) echo "$m/pe_image_encoder/1/model.plan $m/pe_text_encoder/1/model.onnx" ;;
        *) echo "" ;;
    esac
}

# model_setup_required_models "core curation ..." -> comma list for OP_REQUIRED_MODELS, the
# models the API's /health waits on. OCR is skipped by the minimal profile and the PE text
# tower has a CPU fallback, so neither gates readiness; the PE image encoder has none.
model_setup_required_models() {
    local g m out=()
    for g in $(model_setup_groups_for_tiers "$1"); do
        [[ "$g" == ocr ]] && continue
        for m in $(model_setup_group_models "$g"); do
            [[ "$m" == pe_text_encoder ]] || out+=("$m")
        done
    done
    (IFS=,; echo "${out[*]}")
}
