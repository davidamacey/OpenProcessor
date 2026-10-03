/**
 * Single site-identity module.
 *
 * Every project-specific value (name, repo, URLs, nav/footer links, sibling
 * cross-links) lives here so this whole docs-site/ directory can be cloned
 * for a sibling project (OpenProcessor, and this project already borrows
 * the structure of a sister project's own docs-site) by editing ONLY this
 * file, the data JSONs under `src/data/`, `docs/**`, and `static/img/**`.
 * See `TEMPLATE.md` for the exact clone checklist.
 *
 * Nothing here is client-only code (no browser APIs, no JSX) so it can be
 * imported by both `docusaurus.config.ts` (Node.js, build time) and React
 * components (browser, via `@site/site.config`).
 */

export type SiteLink = {label: string; to?: string; href?: string};
export type SiteLinkGroup = {title: string; items: SiteLink[]};

export const siteConfig = {
  title: 'Cropwright',
  tagline: 'From raw images to a trained detector, without labeling one box at a time',
  // Supporting line under the hero tagline.
  heroSubtitle:
    'Clusters and a vision-language model do the bulk labeling; you confirm at keyboard speed. Then export, train, compare and promote models in the same app, for any domain. The labeling frontend for OpenProcessor.',
  favicon: 'img/favicon.svg',
  logo: 'img/favicon.svg',

  // GitHub Pages deployment target.
  organizationName: 'example-org',
  projectName: 'cropwright',
  url: 'http://localhost:5184',
  baseUrl: '/cropwright/',

  githubRepo: 'https://github.com/davidamacey/OpenProcessor',
  // No public Cropwright repository before the OpenProcessor 0.5.0 monorepo
  // release, so there is no "Edit this page" target yet.
  editUrlBase: undefined,

  license: 'AGPL-3.0-only',
  copyrightHolder: 'example-org LLC',

  // Animated walkthrough under the hero title; built from the committed
  // screenshots by scripts/create-workflow-gif.sh. Set to null to omit.
  heroDemo: {
    src: '/img/cropwright-workflow.gif',
    alt: 'Cropwright walkthrough: dashboard, ingest, clusters, review, classes, export, train, bake-off and settings',
  } as {src: string; alt: string} | null,

  // Cross-links to sibling projects sharing this docs framework / product family.
  siblingProjects: [
    {
      label: 'OpenProcessor',
      href: 'https://github.com/davidamacey/OpenProcessor',
      description: 'The data, clustering, VLM and training backend Cropwright is a frontend for.',
    },
  ],

  announcementBar: {
    id: 'no-auth-warning',
    content:
      'The curation API has <b>no request authentication</b> — never expose Cropwright or its ' +
      'backend to the public internet. ' +
      '<a target="_blank" rel="noopener" href="/cropwright/docs/operations/security">Read the security notes</a>.',
    backgroundColor: '#2a1a1a',
    textColor: '#f5b5b5',
  },

  navbar: {
    title: 'Cropwright',
    items: [
      {to: '/docs/getting-started/introduction', position: 'left' as const, label: 'Docs'},
      {to: '/architecture', position: 'left' as const, label: 'Architecture'},
      {to: '/roadmap', position: 'left' as const, label: 'Roadmap'},
      {href: 'https://github.com/davidamacey/OpenProcessor', position: 'right' as const, label: 'GitHub'},
    ],
  },

  footerLinkGroups: [
    {
      title: 'Documentation',
      items: [
        {label: 'Getting Started', to: '/docs/getting-started/introduction'},
        {label: 'User Guide', to: '/docs/user-guide/dashboard'},
        {label: 'FAQ', to: '/docs/faq'},
      ],
    },
    {
      title: 'Project',
      items: [
        {label: 'Architecture', to: '/architecture'},
        {label: 'Roadmap', to: '/roadmap'},
        {label: 'GitHub Issues', href: 'https://github.com/davidamacey/OpenProcessor/issues'},
        {label: 'GitHub Repository', href: 'https://github.com/davidamacey/OpenProcessor'},
      ],
    },
    {
      title: 'Related projects',
      items: [{label: 'OpenProcessor (backend)', href: 'https://github.com/davidamacey/OpenProcessor'}],
    },
  ] satisfies SiteLinkGroup[],
};

export type SiteConfig = typeof siteConfig;
