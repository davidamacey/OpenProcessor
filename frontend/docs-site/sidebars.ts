import type {SidebarsConfig} from '@docusaurus/plugin-content-docs';

const sidebars: SidebarsConfig = {
  docsSidebar: [
    {
      type: 'category',
      label: 'Getting Started',
      items: [
        'getting-started/introduction',
        'getting-started/quick-start',
        'getting-started/architecture-overview',
      ],
    },
    {
      type: 'category',
      label: 'User Guide',
      items: [
        'user-guide/projects',
        'user-guide/combine-projects',
        'user-guide/dashboard',
        'user-guide/ingest',
        'user-guide/dataset-import',
        'user-guide/clusters',
        'user-guide/review',
        'user-guide/classes',
        'user-guide/export',
        'user-guide/train',
        'user-guide/bakeoff',
        'user-guide/models',
        'user-guide/settings',
        'user-guide/prompt-packs',
        'user-guide/region-profiles',
        'user-guide/vlm-models',
        'user-guide/multi-box-regions',
        'user-guide/keyboard-shortcuts',
      ],
    },
    {
      type: 'category',
      label: 'Configuration',
      items: [
        'configuration/environment-variables',
        'configuration/backend-feature-flags',
        'configuration/runtime-settings',
        'configuration/annotation-profiles',
        'configuration/second-instance',
      ],
    },
    {
      type: 'category',
      label: 'Operations',
      items: [
        'operations/deployment',
        'operations/project-administration',
        'operations/security',
        'operations/upgrading',
        'operations/troubleshooting',
      ],
    },
    {
      type: 'category',
      label: 'Developer Guide',
      items: [
        'developer-guide/development-setup',
        'developer-guide/testing',
        'developer-guide/api-contract',
        'developer-guide/screenshots',
        'developer-guide/contributing',
        'developer-guide/releasing',
      ],
    },
    'faq',
  ],
};

export default sidebars;
