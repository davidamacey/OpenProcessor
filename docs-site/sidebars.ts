import type {SidebarsConfig} from '@docusaurus/plugin-content-docs';

const sidebars: SidebarsConfig = {
  docsSidebar: [
    {
      type: 'category',
      label: 'Getting Started',
      items: [
        'getting-started/introduction',
        'getting-started/quick-start',
        'getting-started/installer',
        'getting-started/model-export',
        'getting-started/compose-profiles',
        'getting-started/second-stack',
      ],
    },
    'architecture/overview',
    {
      type: 'category',
      label: 'Guides',
      items: [
        'guides/projects',
        'guides/settings-and-config',
        'guides/use-your-own-domain',
        'guides/wheel-example',
        'guides/keymaps',
        'guides/vlm-selection',
        'guides/prompt-packs',
        'guides/region-profiles',
        'guides/multi-box-regions',
        'guides/dataset-import',
        'guides/reprocess',
        'guides/combine-projects',
        'guides/test-on-crop',
        'guides/open-vocabulary',
      ],
    },
    {
      type: 'category',
      label: 'API Reference',
      items: ['api-reference/overview', 'api-reference/core', 'api-reference/curation'],
    },
    {
      type: 'category',
      label: 'Configuration',
      items: ['configuration/basic', 'configuration/advanced'],
    },
    {
      type: 'category',
      label: 'Deployment',
      items: ['deployment/gpu-sizing', 'deployment/sizing-and-storage', 'deployment/security'],
    },
    {
      type: 'category',
      label: 'Operations',
      items: [
        'operations/curation-workflow',
        'operations/health-and-stalls',
        'operations/workers',
        'operations/exports-and-retention',
        'operations/training-and-promote',
        'operations/monitoring',
      ],
    },
    {
      type: 'category',
      label: 'About',
      items: ['about/vision-and-goals'],
    },
    {
      type: 'category',
      label: 'Reference',
      items: [
        'reference/licenses',
        'developer-guide/screenshots',
        'developer-guide/restart-after-update',
      ],
    },
  ],
};

export default sidebars;
