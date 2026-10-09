/**
 * Single site-identity module.
 *
 * Every project-specific value (name, repo, URLs, nav/footer links, sibling
 * cross-links, landing-page quick start) lives here so this whole docs-site/
 * directory stays a clone of the shared docs framework: a sibling project
 * edits ONLY this file, the data JSONs under `src/data/`, `docs/**`, and
 * `static/img/**`.
 *
 * Nothing here is client-only code (no browser APIs, no JSX) so it can be
 * imported by both `docusaurus.config.ts` (Node.js, build time) and React
 * components (browser, via `@site/site.config`).
 */

export type SiteLink = {label: string; to?: string; href?: string};
export type SiteLinkGroup = {title: string; items: SiteLink[]};

const githubRepo = 'https://github.com/davidamacey/OpenProcessor';

export const siteConfig = {
  title: 'OpenProcessor',
  tagline:
    'Self-hosted, Triton-backed visual AI: detection, faces, embeddings, OCR, and an active-learning curation backend',
  favicon: 'img/favicon.svg',
  logo: 'img/logo.svg',

  // GitHub Pages deployment target.
  organizationName: new URL(githubRepo).pathname.split('/')[1],
  projectName: 'OpenProcessor',
  url: 'https://davidamacey.github.io',
  baseUrl: '/OpenProcessor/',

  githubRepo,
  editUrlBase: `${githubRepo}/tree/main/docs-site/`,

  license: 'AGPL-3.0-or-later',
  copyrightHolder: 'OpenProcessor Contributors',

  // Hero walkthrough GIF, built by scripts/docs/create-workflow-gif.sh from a
  // public-sample-data stack. Set to null to hide it.
  heroDemo: {
    src: '/img/openprocessor-workflow.gif',
    alt: 'OpenProcessor walkthrough: API calls in a terminal, the Swagger API docs, Triton model status, live Grafana dashboards, Prometheus targets, MLflow training runs and a run comparison, OpenSearch Dashboards, then Cropwright, the labeling UI for this API',
  } as {src: string; alt: string} | null,

  // Cross-links to sibling projects sharing this docs framework / product family.
  siblingProjects: [
    {
      label: 'OpenTranscribe',
      href: 'https://github.com/attevon-llc/OpenTranscribe',
      description: 'A sibling self-hosted AI project for audio/video transcription.',
    },
  ],

  announcementBar: {
    id: 'no-auth-warning',
    content:
      'OpenProcessor has <b>no request authentication</b> on any route — never expose it ' +
      'to the public internet. ' +
      '<a target="_blank" rel="noopener" href="/OpenProcessor/docs/deployment/security">Read the security notes</a>.',
    backgroundColor: '#2a1a1a',
    textColor: '#f5b5b5',
  },

  // Landing-page quick start (rendered by src/components/QuickStart).
  quickStart: {
    intro: 'Clone, run the setup script, and the core API is up on port 4603.',
    command:
      `git clone ${githubRepo}.git && cd OpenProcessor\n` +
      './scripts/setup.sh --yes\n' +
      'curl http://localhost:4603/health\n' +
      '# opt-in curation workers\n' +
      'docker compose --profile curation up -d',
    guideLabel: 'Full quick-start guide',
    guidePath: '/docs/getting-started/quick-start',
  },

  // The "How it works" section's closing link.
  howItWorksMore: {
    prefix: 'Walk through the whole loop in the',
    label: 'curation workflow guide',
    to: '/docs/operations/curation-workflow',
  },

  // Page copy for the generic /architecture and /roadmap pages.
  architecturePage: {
    description:
      'OpenProcessor architecture diagrams: services, backend modules, data model, inference, ' +
      'curation, region cascade and training/promote flows.',
    intro:
      'How OpenProcessor fits together, drawn from the current code. Each diagram is ' +
      'interactive: switch views with the tabs above the canvas, and zoom or open it full screen.',
  },
  roadmapPage: {
    description: "What's shipped in OpenProcessor and what's planned next.",
  },

  navbar: {
    title: 'OpenProcessor',
    items: [
      {to: '/docs/getting-started/introduction', position: 'left' as const, label: 'Docs'},
      {
        type: 'docSidebar' as const,
        sidebarId: 'cropwrightSidebar',
        position: 'left' as const,
        label: 'Cropwright',
      },
      {to: '/docs/api-reference/overview', position: 'left' as const, label: 'API'},
      {to: '/architecture', position: 'left' as const, label: 'Architecture'},
      {to: '/roadmap', position: 'left' as const, label: 'Roadmap'},
      {href: githubRepo, position: 'right' as const, label: 'GitHub'},
    ],
  },

  footerLinkGroups: [
    {
      title: 'Documentation',
      items: [
        {label: 'Getting Started', to: '/docs/getting-started/introduction'},
        {label: 'API Reference', to: '/docs/api-reference/overview'},
        {label: 'Configuration', to: '/docs/configuration/basic'},
        {label: 'Operations', to: '/docs/operations/health-and-stalls'},
        {label: 'Cropwright (labeling UI)', to: '/docs/cropwright/getting-started/introduction'},
      ],
    },
    {
      title: 'Project',
      items: [
        {label: 'Architecture', to: '/architecture'},
        {label: 'Roadmap', to: '/roadmap'},
        {label: 'GitHub Issues', href: `${githubRepo}/issues`},
        {label: 'GitHub Repository', href: githubRepo},
      ],
    },
    {
      title: 'Related projects',
      items: [
        {label: 'OpenTranscribe', href: 'https://github.com/attevon-llc/OpenTranscribe'},
      ],
    },
  ] satisfies SiteLinkGroup[],
};

export type SiteConfig = typeof siteConfig;
