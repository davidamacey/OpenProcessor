import { describe, expect, it } from 'vitest';
import {
  buildCampaign,
  buildSpec,
  profileHyperparams,
  type TrainFormValues,
} from './trainSpec';

function form(over: Partial<TrainFormValues> = {}): TrainFormValues {
  return {
    datasetExportDir: '/exports/yolo/current',
    modelSize: 'm',
    profileName: 'medium',
    cudaDevices: '0',
    selectedClasses: [1, 2],
    singleCls: false,
    singleClassExport: false,
    hp: profileHyperparams({}),
    augmentation: null,
    autoQuantizeBakeoff: false,
    campaignSizes: ['n', 's'],
    autoPromoteBest: false,
    stopWhenMap50: 0.9,
    ...over,
  };
}

describe('trainSpec', () => {
  it('sends only the overridden hyperparameters next to the fixed optimizer', () => {
    const hp = { ...profileHyperparams({ epochs: 50, batch: -1 }), mixup: 0 };
    expect(buildSpec(form({ hp })).hyperparameters).toEqual({
      optimizer: 'MuSGD',
      epochs: 50,
      batch: -1,
      mixup: 0,
    });
  });

  it('a single-class export sends no class subset and single_cls, for a run and a campaign', () => {
    const f = form({ singleClassExport: true });
    expect(buildSpec(f)).toMatchObject({ include_classes: null, single_cls: true });
    expect(buildCampaign(f)).toMatchObject({ include_classes: null, single_cls: true });
  });

  it('builds one campaign run per size, and no stop rule for a blank or zero mAP target', () => {
    const c = buildCampaign(form({ stopWhenMap50: 0 }));
    expect(c.runs).toEqual([
      { profile: 'medium', model_size: 'n' },
      { profile: 'medium', model_size: 's' },
    ]);
    expect(c.stop_when).toBeNull();
    expect(buildCampaign(form()).stop_when).toEqual({ map50_at_least: 0.9 });
  });

  it('omits an empty GPU claim so the backend picks from its allowlist', () => {
    expect(buildSpec(form({ cudaDevices: '' })).cuda_visible_devices).toBeUndefined();
  });
});
