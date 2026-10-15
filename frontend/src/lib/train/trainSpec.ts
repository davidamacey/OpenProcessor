/**
 * Builds the `/train/preflight`, `/train/start` and `/train/start_campaign`
 * request bodies from the training form's values. Pure: the form owns the
 * state and passes a snapshot in.
 */
import type {
  AugmentationSpec,
  CampaignRunSpec,
  ModelSize,
  ProfileName,
  TrainCampaignSpec,
  TrainJobSpec,
} from '$lib/types_train';

export const SIZES: ModelSize[] = ['n', 's', 'm', 'l', 'x'];
export const PROFILE_NAMES: ProfileName[] = [
  'probe',
  'nano',
  'small',
  'medium',
  'large',
  'xlarge',
];

/** Hyperparameter overrides, keyed by Ultralytics arg name. `null` =
 *  "use the profile default". */
export interface HyperparamOverrides {
  epochs: number | null;
  batch: number | null;
  imgsz: number | null;
  lr0: number | null;
  lrf: number | null;
  momentum: number | null;
  weight_decay: number | null;
  mosaic: number | null;
  mixup: number | null;
  copy_paste: number | null;
  patience: number | null;
}

/** The overrides a profile's served `defaults` seed. */
export function profileHyperparams(def: Record<string, unknown>): HyperparamOverrides {
  return {
    epochs: (def.epochs as number | undefined) ?? null,
    batch: (def.batch as number | undefined) ?? null,
    imgsz: (def.imgsz as number | undefined) ?? null,
    lr0: (def.lr0 as number | undefined) ?? null,
    lrf: (def.lrf as number | undefined) ?? null,
    momentum: (def.momentum as number | undefined) ?? null,
    weight_decay: (def.weight_decay as number | undefined) ?? null,
    mosaic: (def.mosaic as number | undefined) ?? null,
    mixup: (def.mixup as number | undefined) ?? null,
    copy_paste: (def.copy_paste as number | undefined) ?? null,
    patience: (def.patience as number | undefined) ?? null,
  };
}

export interface TrainFormValues {
  datasetExportDir: string;
  modelSize: ModelSize;
  profileName: ProfileName;
  cudaDevices: string;
  selectedClasses: number[] | null;
  singleCls: boolean;
  singleClassExport: boolean;
  hp: HyperparamOverrides;
  augmentation: AugmentationSpec | null;
  autoQuantizeBakeoff: boolean;
  campaignSizes: ModelSize[];
  autoPromoteBest: boolean;
  stopWhenMap50: number | null;
}

// Only set hyperparameter fields the user actually overrode (anything
// still null is left to profile defaults inside the trainer container).
function buildHyperparameters(overrides: HyperparamOverrides): Record<string, unknown> {
  const hp: Record<string, unknown> = {
    // optimizer is fixed for YOLO26 (preflight rejects 'auto').
    optimizer: 'MuSGD',
  };
  if (overrides.epochs != null) hp.epochs = overrides.epochs;
  if (overrides.batch != null) hp.batch = overrides.batch;
  if (overrides.imgsz != null) hp.imgsz = overrides.imgsz;
  if (overrides.lr0 != null) hp.lr0 = overrides.lr0;
  if (overrides.lrf != null) hp.lrf = overrides.lrf;
  if (overrides.momentum != null) hp.momentum = overrides.momentum;
  if (overrides.weight_decay != null) hp.weight_decay = overrides.weight_decay;
  if (overrides.mosaic != null) hp.mosaic = overrides.mosaic;
  if (overrides.mixup != null) hp.mixup = overrides.mixup;
  if (overrides.copy_paste != null) hp.copy_paste = overrides.copy_paste;
  if (overrides.patience != null) hp.patience = overrides.patience;
  return hp;
}

export function buildSpec(form: TrainFormValues): TrainJobSpec {
  const {
    datasetExportDir,
    modelSize,
    profileName,
    cudaDevices,
    singleClassExport,
    selectedClasses,
    singleCls,
    augmentation,
    autoQuantizeBakeoff,
  } = form;
  return {
    dataset_export_dir: datasetExportDir,
    model_family: 'yolo26',
    model_size: modelSize,
    profile: profileName,
    cuda_visible_devices: cudaDevices || undefined,
    // A single-class export is already a class-0 dataset, so never filter
    // it by the multi-class registry ids — that drops every label. Send
    // no class subset for single-class runs.
    include_classes: singleClassExport ? null : selectedClasses,
    single_cls: singleClassExport ? true : singleCls,
    hyperparameters: buildHyperparameters(form.hp),
    augmentation: augmentation && augmentation.enabled ? augmentation : null,
    auto_quantize_bakeoff: autoQuantizeBakeoff,
  };
}

export function buildCampaign(form: TrainFormValues): TrainCampaignSpec {
  const {
    datasetExportDir,
    profileName,
    cudaDevices,
    singleClassExport,
    selectedClasses,
    singleCls,
    augmentation,
    campaignSizes,
    autoPromoteBest,
    stopWhenMap50,
  } = form;
  const runs: CampaignRunSpec[] = campaignSizes.map((size) => ({
    // For now we re-use the currently picked profile for every size —
    // a power-user can still tweak each size's run after submit by
    // editing the queued job.json. Could expand to per-size pickers
    // later if it proves useful.
    profile: profileName,
    model_size: size,
  }));
  const stopWhen =
    stopWhenMap50 != null && stopWhenMap50 > 0 ? { map50_at_least: stopWhenMap50 } : null;
  return {
    dataset_export_dir: datasetExportDir,
    include_classes: singleClassExport ? null : selectedClasses,
    single_cls: singleClassExport ? true : singleCls,
    cuda_visible_devices: cudaDevices || undefined,
    augmentation: augmentation && augmentation.enabled ? augmentation : null,
    runs,
    stop_when: stopWhen,
    auto_promote_best: autoPromoteBest,
  };
}
