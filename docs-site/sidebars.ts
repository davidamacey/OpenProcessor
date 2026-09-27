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
      items: ['deployment/gpu-sizing', 'deployment/security'],
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
      label: 'Reference',
      items: ['reference/licenses', 'developer-guide/screenshots'],
    },
  ],
};

export default sidebars;
