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
        'operations/release-acceptance',
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
  // The Cropwright labeling UI: its own tab in the navbar, same docs plugin and site.
  cropwrightSidebar: [
    {
      type: 'category',
      label: 'Getting Started',
      items: [
        'cropwright/getting-started/introduction',
        'cropwright/getting-started/quick-start',
        'cropwright/getting-started/architecture-overview',
        'cropwright/getting-started/architecture-diagrams',
      ],
    },
    {
      type: 'category',
      label: 'User Guide',
      items: [
        'cropwright/user-guide/projects',
        'cropwright/user-guide/combine-projects',
        'cropwright/user-guide/dashboard',
        'cropwright/user-guide/ingest',
        'cropwright/user-guide/ingest-policy',
        'cropwright/user-guide/embedding-state',
        'cropwright/user-guide/dataset-import',
        'cropwright/user-guide/clusters',
        'cropwright/user-guide/item-filter',
        'cropwright/user-guide/review',
        'cropwright/user-guide/classes',
        'cropwright/user-guide/export',
        'cropwright/user-guide/train',
        'cropwright/user-guide/bakeoff',
        'cropwright/user-guide/models',
        'cropwright/user-guide/settings',
        'cropwright/user-guide/prompt-packs',
        'cropwright/user-guide/region-profiles',
        'cropwright/user-guide/open-vocabulary',
        'cropwright/user-guide/vlm-models',
        'cropwright/user-guide/multi-box-regions',
        'cropwright/user-guide/keyboard-shortcuts',
      ],
    },
    {
      type: 'category',
      label: 'Configuration',
      items: [
        'cropwright/configuration/environment-variables',
        'cropwright/configuration/backend-feature-flags',
        'cropwright/configuration/runtime-settings',
        'cropwright/configuration/annotation-profiles',
        'cropwright/configuration/second-instance',
      ],
    },
    {
      type: 'category',
      label: 'Operations',
      items: [
        'cropwright/operations/deployment',
        'cropwright/operations/project-administration',
        'cropwright/operations/security',
        'cropwright/operations/upgrading',
        'cropwright/operations/troubleshooting',
      ],
    },
    {
      type: 'category',
      label: 'Developer Guide',
      items: [
        'cropwright/developer-guide/development-setup',
        'cropwright/developer-guide/testing',
        'cropwright/developer-guide/api-contract',
        'cropwright/developer-guide/screenshots',
        'cropwright/developer-guide/contributing',
        'cropwright/developer-guide/releasing',
      ],
    },
    'cropwright/faq',
  ],
};

export default sidebars;
